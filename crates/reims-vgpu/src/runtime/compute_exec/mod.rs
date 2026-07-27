//! Product-path compute bind/dispatch for `SEGMENT_TYPE_COMPUTE`.
//!
//! Executable surface:
//! - `0xd0` set compute pipeline (type-7 → kernel function MTLB + optional stage-input)
//! - `0xcb` / `0xd9` set buffers (+ optional attribute stride for dynamic stage-input layouts)
//! - `0xcf` / `0xda` set buffer offset (+ optional attribute stride)
//! - `0xce` set textures (type-2/3 GVA + type-11; sample vs storage via reflection)
//! - `0xcc` / `0xcd` set samplers (+ optional LOD clamp)
//! - `0xd1` direct stage-in region / `0xd2` indirect stage-in region (guest buffer args)
//! - `0xd3` threadgroup memory length
//! - `0xd8` imageblock dimensions
//! - `0xc8`/`0xca` direct dispatch; `0xc9`/`0xe6` indirect (guest args → direct encode)
//! - `0xdb` dispatch type (serial/concurrent)
//!
//! Fences: stream walk (`fence_exec`). Control-flow (`0xdc`–`0xe2`) encodes
//! host Metal SPI on a multi-record [`crate::runtime::compute_session`] (same
//! encoder for the segment). ICB (`0xe4`/`0xe5`) materializes type-7 `0x36` and
//! executes filled host command slots (CPU fill via [`crate::runtime::icb`];
//! stream fill opcodes remain unknown). Nested dispatches on an open session
//! encode onto that encoder (inside SPI); writeback runs after session commit.
//! Barriers and compressed-texture flush are ordered no-ops.
//!
//! One-shot encode uses [`crate::backend::metal::compute::compute_core`]; nested
//! encode uses `compute_encode_on_encoder`. Buffer and storage-image writeback
//! is GVA / type-11 staged.

use crate::contract::endian::ld32;
use crate::contract::pixel_format;
use crate::model::DeviceState;
use crate::runtime::decode::compute::{
    BufferBinding, Command as ComputeCommand, Kind, RefBinding, SamplerBinding,
};
use crate::runtime::decode::resource::{
    decode_function_descriptor, decode_texture_descriptor, decode_type7_descriptor,
    texture_type8_opcode, ComputeStageInputDescriptor, Descriptor as ResourceDescriptor,
    HEAP_TEXTURE_DESCRIPTOR, HEAP_TEXTURE_HEAP_REF, HEAP_TEXTURE_LEN, HEAP_TEXTURE_OFFSET,
    HEAP_TEXTURE_OPCODE, HEAP_TEXTURE_USE_OFFSET, OBJECT_TYPE_BUFFER, OBJECT_TYPE_FUNCTION,
    OBJECT_TYPE_TEXTURE, OBJECT_TYPE_TEXTURE_VARIANT, OBJECT_TYPE_TEXTURE_VIEW, OBJECT_TYPE_TYPE7,
    TEXTURE_VIEW_OPCODE_BUFFER_TEXTURE,
};
#[cfg(all(feature = "backend-metal", target_os = "macos"))]
use crate::runtime::decode::resource::{
    decode_sampler_descriptor, OBJECT_TYPE_IOSURFACE, TYPE7_OBJECT_SAMPLER,
};
use crate::runtime::gva_mem;
use crate::runtime::host::{HostMemory, HostOps};
use crate::runtime::mapper;
use crate::runtime::mapping_write;
use crate::runtime::metal_draw::host_alloc_len;
use crate::runtime::objects;

/// Cap on Metal compute buffer slots (matches backend `REIMS_VGPU_METAL_MAX_BUFFERS`).
pub const MAX_COMPUTE_BUFFER_SLOTS: u32 = 31;
/// Cap on compute texture stream indices (Metal bind = 32 + index).
pub const MAX_COMPUTE_TEXTURE_SLOTS: u32 = 31;
/// Cap on compute sampler stream indices (Metal bind = 64 + index).
pub const MAX_COMPUTE_SAMPLER_SLOTS: u32 = 16;
/// Cap on threadgroup-memory indices (plan `REIMS_VGPU_COMPUTE_PLAN_MAX_THREADGROUP_MEMORY`).
pub const MAX_THREADGROUP_MEMORY_SLOTS: u32 = 16;
/// `MTLDispatchThreadgroupsIndirectArguments` = three `uint32_t` (12 bytes).
pub const INDIRECT_THREADGROUPS_ARGS_LEN: usize = 12;
/// `MTLDispatchThreadsIndirectArguments` = six `uint32_t` (24 bytes).
pub const INDIRECT_THREADS_ARGS_LEN: usize = 24;
/// `MTLStageInRegionIndirectArguments` = six `uint32_t` (24 bytes).
pub const STAGE_IN_INDIRECT_ARGS_LEN: usize = 24;

/// Fail-visible, deduped record of a compute resource bind dropped because its
/// slot index exceeds the argument-table cap. The guest bound a real resource
/// (`ref != 0`, or a non-empty threadgroup allocation) at a slot we cannot
/// represent, so the dispatch runs *missing that bind* — wrong compute output
/// with no other symptom, previously silent. Runs on the drain worker (off the
/// QEMU main core). Deduped per `(table, index)` so a repeating dispatch cannot
/// flood, and a healthy guest — which binds within the Metal argument-table caps —
/// never fires it. The cap comparison is exclusive (`index >= MAX_*`) to match the
/// backend, which sizes its argument-table arrays to exactly these counts
/// (`[false; REIMS_VGPU_METAL_MAX_BUFFERS]`) and guards `idx >= REIMS_VGPU_METAL_MAX_*` before
/// indexing — so slot `MAX` is out of range and a bind there is a genuine drop, not
/// a boundary the accum should have accepted.
fn note_compute_bind_overflow(table: &'static str, index: u32, resource_ref: u32, cap: u32) {
    use std::collections::HashSet;
    use std::sync::{Mutex, OnceLock};
    static SEEN: OnceLock<Mutex<HashSet<(&'static str, u32)>>> = OnceLock::new();
    let seen = SEEN.get_or_init(|| Mutex::new(HashSet::new()));
    if seen
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .insert((table, index))
    {
        crate::observe::fail(format!(
            "compute_bind_overflow reason={table}_index_overflow index={index} \
             arg={resource_ref} cap={cap} (bind dropped; dispatch runs without it)"
        ));
    }
}

#[derive(Clone, Debug, Default)]
pub struct ComputeBufferBind {
    pub index: u32,
    pub buffer_ref: u32,
    pub offset: u64,
    pub attribute_stride: u64,
    pub has_attribute_stride: bool,
}

#[derive(Clone, Debug, Default)]
pub struct ComputeTextureBind {
    /// Stream texture index (`0xce first + i`); Metal bind = 32 + index.
    pub index: u32,
    pub texture_ref: u32,
}

#[derive(Clone, Debug, Default)]
pub struct ComputeSamplerBind {
    /// Stream sampler index; Metal bind = 64 + index.
    pub index: u32,
    pub sampler_ref: u32,
    pub lod_min_bits: u32,
    pub lod_max_bits: u32,
    pub has_lod_clamp: bool,
}

#[derive(Clone, Debug, Default)]
pub struct ThreadgroupMemoryBind {
    pub index: u32,
    pub length: u64,
}

#[derive(Clone, Debug, Default)]
pub struct StageInRegion {
    pub origin_x: u64,
    pub origin_y: u64,
    pub origin_z: u64,
    pub size_x: u64,
    pub size_y: u64,
    pub size_z: u64,
}

#[derive(Clone, Debug, Default)]
pub struct StageInRegionIndirect {
    pub buffer_ref: u32,
    pub buffer_offset: u64,
}

#[derive(Clone, Debug, Default)]
pub struct ImageblockDimensions {
    pub width: u32,
    pub height: u32,
}

#[derive(Clone, Debug, Default)]
pub struct ComputeAccum {
    pub pipeline_ref: u32,
    pub buffers: Vec<ComputeBufferBind>,
    pub textures: Vec<ComputeTextureBind>,
    pub samplers: Vec<ComputeSamplerBind>,
    pub threadgroup_memory: Vec<ThreadgroupMemoryBind>,
    /// Last direct `0xd1` stage-in region (cleared by `0xd2`).
    pub stage_in_region: Option<StageInRegion>,
    /// Last `0xd2` indirect stage-in (clears direct region).
    pub stage_in_region_indirect: Option<StageInRegionIndirect>,
    /// Last `0xd8` imageblock dimensions.
    pub imageblock: Option<ImageblockDimensions>,
    /// Last decoded `0xdb` dispatch type (Metal serial/concurrent); 0 = serial.
    pub dispatch_type: u32,
}

impl ComputeAccum {
    pub fn set_pipeline(&mut self, pipeline_ref: u32) {
        if pipeline_ref != 0 {
            self.pipeline_ref = pipeline_ref;
        }
    }

    pub fn bind_buffers(&mut self, first: u32, entries: &[BufferBinding]) {
        for (i, e) in entries.iter().enumerate() {
            let index = first.saturating_add(i as u32);
            if e.ref_ == 0 {
                continue; // unbind this slot: expected control flow, stay silent.
            }
            if index >= MAX_COMPUTE_BUFFER_SLOTS {
                note_compute_bind_overflow("buffer", index, e.ref_, MAX_COMPUTE_BUFFER_SLOTS);
                continue;
            }
            let bind = ComputeBufferBind {
                index,
                buffer_ref: e.ref_,
                offset: e.offset,
                attribute_stride: e.attribute_stride,
                has_attribute_stride: e.has_attribute_stride,
            };
            if let Some(slot) = self.buffers.iter_mut().find(|b| b.index == index) {
                *slot = bind;
            } else {
                self.buffers.push(bind);
            }
        }
    }

    pub fn set_buffer_offset(&mut self, index: u32, offset: u64, attribute_stride: Option<u64>) {
        if let Some(slot) = self.buffers.iter_mut().find(|b| b.index == index) {
            slot.offset = offset;
            if let Some(s) = attribute_stride {
                slot.attribute_stride = s;
                slot.has_attribute_stride = true;
            }
        }
    }

    pub fn bind_textures(&mut self, first: u32, entries: &[RefBinding]) {
        for (i, e) in entries.iter().enumerate() {
            let index = first.saturating_add(i as u32);
            if e.ref_ == 0 {
                continue; // unbind this slot: expected control flow, stay silent.
            }
            if index >= MAX_COMPUTE_TEXTURE_SLOTS {
                note_compute_bind_overflow("texture", index, e.ref_, MAX_COMPUTE_TEXTURE_SLOTS);
                continue;
            }
            let bind = ComputeTextureBind {
                index,
                texture_ref: e.ref_,
            };
            if let Some(slot) = self.textures.iter_mut().find(|t| t.index == index) {
                *slot = bind;
            } else {
                self.textures.push(bind);
            }
        }
    }

    pub fn bind_samplers(&mut self, first: u32, entries: &[SamplerBinding]) {
        for (i, e) in entries.iter().enumerate() {
            let index = first.saturating_add(i as u32);
            if e.ref_ == 0 {
                continue; // unbind this slot: expected control flow, stay silent.
            }
            if index >= MAX_COMPUTE_SAMPLER_SLOTS {
                note_compute_bind_overflow("sampler", index, e.ref_, MAX_COMPUTE_SAMPLER_SLOTS);
                continue;
            }
            let bind = ComputeSamplerBind {
                index,
                sampler_ref: e.ref_,
                lod_min_bits: e.lod_min_bits,
                lod_max_bits: e.lod_max_bits,
                has_lod_clamp: e.has_lod_clamp,
            };
            if let Some(slot) = self.samplers.iter_mut().find(|s| s.index == index) {
                *slot = bind;
            } else {
                self.samplers.push(bind);
            }
        }
    }

    pub fn set_threadgroup_memory(&mut self, index: u32, length: u64) {
        if index >= MAX_THREADGROUP_MEMORY_SLOTS {
            // A non-empty allocation at an over-cap slot is a genuine dropped bind
            // (the kernel expects threadgroup memory here); a zero length is an
            // unbind, expected control flow. `arg` carries the requested length.
            if length != 0 {
                note_compute_bind_overflow(
                    "threadgroup",
                    index,
                    length.min(u32::MAX as u64) as u32,
                    MAX_THREADGROUP_MEMORY_SLOTS,
                );
            }
            return;
        }
        let bind = ThreadgroupMemoryBind { index, length };
        if let Some(slot) = self
            .threadgroup_memory
            .iter_mut()
            .find(|t| t.index == index)
        {
            *slot = bind;
        } else {
            self.threadgroup_memory.push(bind);
        }
    }

    pub fn set_stage_in_region(&mut self, region: StageInRegion) {
        self.stage_in_region_indirect = None;
        self.stage_in_region = Some(region);
    }

    pub fn set_stage_in_region_indirect(&mut self, buffer_ref: u32, buffer_offset: u64) {
        if buffer_ref == 0 {
            return;
        }
        self.stage_in_region = None;
        self.stage_in_region_indirect = Some(StageInRegionIndirect {
            buffer_ref,
            buffer_offset,
        });
    }

    pub fn set_imageblock(&mut self, width: u32, height: u32) {
        if width == 0 || height == 0 {
            return;
        }
        self.imageblock = Some(ImageblockDimensions { width, height });
    }
}

/// The compute rail's refusal vocabulary.
///
/// Every refusing variant carries the **registered slug of the check that
/// refused**, not just its class. Before that payload existed, nine of these
/// variants were payload-free and 129 construction sites collapsed into them —
/// `MetalFailed` alone spoke for 38 checks, `MissingTexture` for 25 — so a live
/// `compute_dispatches_fail` counter told you a dispatch died and nothing else.
/// The slug is what makes the class greppable; the class is what decides the
/// caller's recovery.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ComputeStatus {
    Ok,
    #[cfg(all(feature = "backend-metal", target_os = "macos"))]
    MetalBackend(crate::backend::metal::error::Status),
    MissingPipeline(&'static str),
    MissingMtlb(&'static str),
    MissingBuffer(&'static str),
    MissingTexture(&'static str),
    MissingSampler(&'static str),
    BadGrid(&'static str),
    GuestIo(&'static str),
    MetalFailed(&'static str),
    NoMetal(&'static str),
    Unsupported(&'static str),
}

impl crate::observe::Refusal for ComputeStatus {
    fn refusal(&self) -> Option<&'static str> {
        match self {
            // The only non-refusal. Keeping it in the same enum is what makes
            // `Emit::refusal` unable to log a success by accident.
            Self::Ok => None,
            #[cfg(all(feature = "backend-metal", target_os = "macos"))]
            Self::MetalBackend(status) => status.refusal(),
            Self::MissingPipeline(slug)
            | Self::MissingMtlb(slug)
            | Self::MissingBuffer(slug)
            | Self::MissingTexture(slug)
            | Self::MissingSampler(slug)
            | Self::BadGrid(slug)
            | Self::GuestIo(slug)
            | Self::MetalFailed(slug)
            | Self::NoMetal(slug)
            | Self::Unsupported(slug) => Some(slug),
        }
    }

    fn fields(&self) -> Vec<(&'static str, String)> {
        // The class next to the reason: `MissingTexture` vs `MetalFailed` is
        // what the caller acted on, and a reader correlating a log line with a
        // recovery path needs both.
        #[cfg(all(feature = "backend-metal", target_os = "macos"))]
        if let Self::MetalBackend(status) = self {
            let mut fields = crate::observe::Refusal::fields(status);
            fields.push(("recovery", "metal_failed".to_string()));
            return fields;
        }
        vec![("class", self.class().to_string())]
    }
}

impl ComputeStatus {
    /// The variant name, for the `class=` field and for call sites that render
    /// their own line.
    pub fn class(&self) -> &'static str {
        match self {
            Self::Ok => "ok",
            #[cfg(all(feature = "backend-metal", target_os = "macos"))]
            Self::MetalBackend(status) => {
                if status.is_args() {
                    "metal_args"
                } else {
                    "metal_execute"
                }
            }
            Self::MissingPipeline(_) => "missing_pipeline",
            Self::MissingMtlb(_) => "missing_mtlb",
            Self::MissingBuffer(_) => "missing_buffer",
            Self::MissingTexture(_) => "missing_texture",
            Self::MissingSampler(_) => "missing_sampler",
            Self::BadGrid(_) => "bad_grid",
            Self::GuestIo(_) => "guest_io",
            Self::MetalFailed(_) => "metal_failed",
            Self::NoMetal(_) => "no_metal",
            Self::Unsupported(_) => "unsupported",
        }
    }

    /// The registered slug this status carries, or `"ok"` when it is not a
    /// refusal. For sites that render a `reason=` into a longer line of their
    /// own rather than building one with [`crate::observe::Emit`].
    pub fn reason(&self) -> &'static str {
        use crate::observe::Refusal as _;
        self.refusal().unwrap_or("ok")
    }
}

/// A malformed translated kernel module before descriptor reflection/execution.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ComputeSpirvDecline {
    HeaderTooShort { len: usize, minimum: usize },
    LengthMisaligned { len: usize, alignment: usize },
}

impl crate::observe::Decline for ComputeSpirvDecline {
    fn slug(&self) -> &'static str {
        match self {
            Self::HeaderTooShort { .. } => "compute_spirv_header_too_short",
            Self::LengthMisaligned { .. } => "compute_spirv_length_misaligned",
        }
    }

    fn fields(&self) -> Vec<(&'static str, String)> {
        match self {
            Self::HeaderTooShort { len, minimum } => {
                vec![("len", len.to_string()), ("minimum", minimum.to_string())]
            }
            Self::LengthMisaligned { len, alignment } => vec![
                ("len", len.to_string()),
                ("alignment", alignment.to_string()),
            ],
        }
    }
}

impl std::fmt::Display for ComputeSpirvDecline {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        use crate::observe::Decline as _;
        write!(f, "reason={}", self.slug())?;
        for (key, value) in self.fields() {
            write!(f, " {key}={value}")?;
        }
        Ok(())
    }
}

impl std::error::Error for ComputeSpirvDecline {}

/// Apply one decoded compute command to accum, or run a dispatch / sequencing op.
///
/// `session` / `block` are the per-segment multi-record encoder and latched
/// sequencing failure (ICB / control encode error). Pass `None` from unit tests
/// that only exercise binds / one-shot dispatch.
pub fn apply_record<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    task_id: u32,
    cmd: &ComputeCommand,
    acc: &mut ComputeAccum,
    session: &mut Option<crate::runtime::compute_session::ComputeSession>,
    block: &mut Option<crate::runtime::compute_session::SequencingBlock>,
) -> Option<ComputeStatus> {
    match cmd.kind {
        Kind::Pipeline => {
            acc.set_pipeline(cmd.pipeline_ref);
            None
        }
        Kind::BufferBind | Kind::BufferBindAttributeStride => {
            acc.bind_buffers(cmd.first, &cmd.buffers);
            None
        }
        Kind::BufferOffset => {
            acc.set_buffer_offset(cmd.first, cmd.buffer_offset, None);
            None
        }
        Kind::BufferOffsetAttributeStride => {
            acc.set_buffer_offset(cmd.first, cmd.buffer_offset, Some(cmd.attribute_stride));
            None
        }
        Kind::TextureBind => {
            acc.bind_textures(cmd.first, &cmd.textures);
            None
        }
        Kind::SamplerBind | Kind::SamplerLod => {
            acc.bind_samplers(cmd.first, &cmd.samplers);
            None
        }
        Kind::DispatchType => {
            acc.dispatch_type = cmd.dispatch_type;
            None
        }
        Kind::StageInRegion => {
            acc.set_stage_in_region(StageInRegion {
                origin_x: cmd.stage_in_region.origin.x,
                origin_y: cmd.stage_in_region.origin.y,
                origin_z: cmd.stage_in_region.origin.z,
                size_x: cmd.stage_in_region.size.x,
                size_y: cmd.stage_in_region.size.y,
                size_z: cmd.stage_in_region.size.z,
            });
            None
        }
        Kind::StageInRegionIndirect => {
            acc.set_stage_in_region_indirect(
                cmd.stage_in_indirect_buffer_ref,
                cmd.stage_in_indirect_buffer_offset,
            );
            None
        }
        Kind::ThreadgroupMemory => {
            acc.set_threadgroup_memory(cmd.threadgroup_memory_index, cmd.threadgroup_memory_length);
            None
        }
        Kind::ImageblockDimensions => {
            acc.set_imageblock(cmd.imageblock_width, cmd.imageblock_height);
            None
        }
        Kind::DispatchThreadgroups
        | Kind::DispatchThreads
        | Kind::DispatchThreadgroupsIndirect
        | Kind::DispatchThreadsIndirect => {
            if block.is_some() {
                return Some(ComputeStatus::Unsupported("dispatch_in_sequencing_block"));
            }
            // Open multi-record session (control-flow SPI): encode on that encoder.
            if let Some(sess) = session.as_mut() {
                return Some(execute_dispatch_nested(
                    state, host, task_id, acc, cmd, sess,
                ));
            }
            Some(execute_dispatch(state, host, task_id, acc, cmd))
        }
        Kind::UpdateFence | Kind::WaitFence => None,
        // Ordered no-ops at the product one-dispatch boundary.
        Kind::BarrierResources
        | Kind::BarrierScope
        | Kind::UseHeaps
        | Kind::UseResources
        | Kind::CompressedTextureFlush => None,
        Kind::ControlStartDoWhile
        | Kind::ControlEndDoWhile
        | Kind::ControlStartWhile
        | Kind::ControlEndWhile
        | Kind::ControlStartIf
        | Kind::ControlStartElse
        | Kind::ControlEndIf
        | Kind::ExecuteCommandsInBuffer
        | Kind::ExecuteCommandsInBufferIndirect => {
            Some(crate::runtime::compute_session::apply_sequencing(
                state, host, task_id, cmd, acc, session, block,
            ))
        }
        Kind::Unknown => None,
    }
}

pub(crate) fn load_mtlb<M: HostMemory + HostOps>(
    state: &DeviceState,
    host: &M,
    task_id: u32,
    func_ref: u32,
) -> Option<Vec<u8>> {
    // ref==0 is "no function bound" (legitimate, e.g. no fragment stage) — stay
    // silent. Every other None is a bound function that failed to materialize,
    // collapsing into the caller's coarse MissingMtlb; log the reason (audit).
    if func_ref == 0 {
        return None;
    }
    let miss = |reason: &str, detail: String| -> Option<Vec<u8>> {
        crate::observe::fail(format!(
            "compute_load_mtlb fail reason={reason} task={task_id} func_ref={func_ref} {detail}"
        ));
        None
    };
    let Some(entry) = objects::lookup_list_entry(state, host, task_id, func_ref) else {
        return miss("no_entry", String::new());
    };
    if entry.object_type != OBJECT_TYPE_FUNCTION {
        return miss("wrong_type", format!("ot={}", entry.object_type));
    }
    let Some(desc) = objects::read_descriptor(state, host, task_id, &entry) else {
        return miss("no_desc", String::new());
    };
    let Ok(f) = decode_function_descriptor(&desc) else {
        return miss("decode", format!("desc_len={}", desc.len()));
    };
    if f.blob_gva == 0 || f.blob_size < 4 {
        return miss(
            "bad_blob",
            format!("blob_gva={:#x} blob_size={}", f.blob_gva, f.blob_size),
        );
    }
    // Guest blob_size is authoritative — no product 1 MiB MTLB ceiling.
    let Some(len) = host_alloc_len(f.blob_size as u64) else {
        return miss(
            "host_len",
            format!("blob_gva={:#x} blob_size={}", f.blob_gva, f.blob_size),
        );
    };
    let mut mtlb = vec![0u8; len];
    if gva_mem::read_task_gva_fallback(
        host,
        &state.tasks,
        task_id,
        f.blob_gva,
        &mut mtlb,
        state.page_shift,
    )
    .is_err()
    {
        return miss(
            "gva_read",
            format!("blob_gva={:#x} blob_size={}", f.blob_gva, f.blob_size),
        );
    }
    Some(mtlb)
}

pub(crate) struct LoadedComputePipeline {
    pub kernel_func_ref: u32,
    /// Product-ready stage-input (None if absent, dropped caps, or incomplete).
    pub stage_input: Option<ComputeStageInputDescriptor>,
}

pub(crate) fn load_compute_pipeline<M: HostMemory + HostOps>(
    state: &DeviceState,
    host: &M,
    task_id: u32,
    pipeline_ref: u32,
) -> Option<LoadedComputePipeline> {
    // ref==0 is "no pipeline bound" (legitimate) — silent. Other None = a bound
    // pipeline that failed to materialize → caller's coarse MissingPipeline; log
    // the reason (audit).
    if pipeline_ref == 0 {
        return None;
    }
    let miss = |reason: &str, detail: String| -> Option<LoadedComputePipeline> {
        crate::observe::fail(format!(
            "compute_load_pipeline fail reason={reason} task={task_id} pipe_ref={pipeline_ref} {detail}"
        ));
        None
    };
    let Some(entry) = objects::lookup_list_entry(state, host, task_id, pipeline_ref) else {
        return miss("no_entry", String::new());
    };
    if entry.object_type != OBJECT_TYPE_TYPE7 {
        return miss("wrong_type", format!("ot={}", entry.object_type));
    }
    let Some(desc) = objects::read_descriptor(state, host, task_id, &entry) else {
        return miss("no_desc", String::new());
    };
    let Ok(decoded) = decode_type7_descriptor(&desc) else {
        return miss("decode", format!("desc_len={}", desc.len()));
    };
    match decoded {
        ResourceDescriptor::ComputePipeline(cp) if cp.kernel_func_ref != 0 => {
            let stage_input = cp.stage_input.and_then(|si| {
                // Dropped entries mean the wire exceeded product/backend caps — fail closed
                // by omitting stage-input rather than silently truncating.
                if si.dropped_attributes != 0 || si.dropped_layouts != 0 {
                    return None;
                }
                if si.attributes.is_empty() && si.layouts.is_empty() {
                    return None;
                }
                Some(si)
            });
            Some(LoadedComputePipeline {
                kernel_func_ref: cp.kernel_func_ref,
                stage_input,
            })
        }
        ResourceDescriptor::ComputePipeline(_) => miss("kernel_func_zero", String::new()),
        _ => miss("not_compute_pipeline", String::new()),
    }
}

/// Resolve a type-1 buffer GVA base + size for task-local reads.
fn buffer_gva_size<M: HostMemory + HostOps>(
    state: &DeviceState,
    host: &M,
    task_id: u32,
    buffer_ref: u32,
) -> Option<(u64, u64)> {
    if buffer_ref == 0 {
        return None;
    }
    let entry = objects::lookup_list_entry(state, host, task_id, buffer_ref)?;
    if entry.object_type != OBJECT_TYPE_BUFFER {
        return None;
    }
    let desc_bytes = objects::read_descriptor(state, host, task_id, &entry)?;
    let desc = crate::runtime::decode::resource::decode_buffer_descriptor(&desc_bytes).ok()?;
    // Product x86 page_shift=12; arm64e=14. Never use arm-only RESOURCE_PAGE_SHIFT
    // default on the live path (compute GuestIo Unmapped class serial-234118).
    desc.backing_gva_size(state.page_shift)
}

/// Read `len` bytes from a type-1 buffer at `offset` (product + session helpers).
pub(crate) fn read_buffer_window<M: HostMemory + HostOps>(
    state: &DeviceState,
    host: &M,
    task_id: u32,
    buffer_ref: u32,
    offset: u64,
    len: usize,
) -> Result<Vec<u8>, ComputeStatus> {
    let (base, size) = buffer_gva_size(state, host, task_id, buffer_ref)
        .ok_or(ComputeStatus::MissingBuffer("compute_buf_win_no_backing"))?;
    if offset
        .checked_add(len as u64)
        .map(|e| e > size)
        .unwrap_or(true)
    {
        return Err(ComputeStatus::MissingBuffer("compute_buf_win_oob"));
    }
    let gva = base
        .checked_add(offset)
        .ok_or(ComputeStatus::MissingBuffer("compute_buf_win_gva_overflow"))?;
    let mut bytes = vec![0u8; len];
    gva_mem::read_task_gva_fallback(
        host,
        &state.tasks,
        task_id,
        gva,
        &mut bytes,
        state.page_shift,
    )
    .map_err(|_| ComputeStatus::GuestIo("compute_buf_win_read"))?;
    Ok(bytes)
}

pub(crate) struct StagedBuffer {
    pub bind: ComputeBufferBind,
    pub gva: u64,
    pub bytes: Vec<u8>,
}

pub(crate) fn stage_buffer<M: HostMemory + HostOps>(
    state: &DeviceState,
    host: &M,
    task_id: u32,
    bind: &ComputeBufferBind,
) -> Result<StagedBuffer, ComputeStatus> {
    // Eight distinct checks answer with `MissingBuffer`; the status carries
    // which one, so the caller's line and this one name the same slug.
    let miss = |st: ComputeStatus, detail: String| -> Result<StagedBuffer, ComputeStatus> {
        crate::observe::fail(format!(
            "compute_stage_buf fail reason={} ref={} off={:#x} {detail}",
            st.reason(),
            bind.buffer_ref,
            bind.offset
        ));
        Err(st)
    };
    let Some(entry) = objects::lookup_list_entry(state, host, task_id, bind.buffer_ref) else {
        return miss(
            ComputeStatus::MissingBuffer("compute_stage_buf_no_entry"),
            String::new(),
        );
    };
    if entry.object_type != OBJECT_TYPE_BUFFER {
        return miss(
            ComputeStatus::MissingBuffer("compute_stage_buf_wrong_type"),
            format!("ot={}", entry.object_type),
        );
    }
    let Some(desc_bytes) = objects::read_descriptor(state, host, task_id, &entry) else {
        return miss(
            ComputeStatus::MissingBuffer("compute_stage_buf_no_desc"),
            String::new(),
        );
    };
    let Ok(desc) = crate::runtime::decode::resource::decode_buffer_descriptor(&desc_bytes) else {
        return miss(
            ComputeStatus::MissingBuffer("compute_stage_buf_decode"),
            format!("desc_len={}", desc_bytes.len()),
        );
    };
    // Device page_shift (x86=12): handle<<shift is the guest VA. Using the arm
    // default (14) mis-places buffers → walker Unmapped (live compute GuestIo).
    let Some((base_gva, size)) = desc.backing_gva_size(state.page_shift) else {
        return miss(
            ComputeStatus::MissingBuffer("compute_stage_buf_no_backing"),
            format!("handle={:#x}", desc.handle),
        );
    };
    if bind.offset >= size {
        return miss(
            ComputeStatus::MissingBuffer("compute_stage_buf_off_oob"),
            format!("size={size:#x}"),
        );
    }
    let avail = size - bind.offset;
    let Some(want) = host_alloc_len(avail).filter(|&n| n > 0) else {
        return miss(
            ComputeStatus::MissingBuffer("compute_stage_buf_want_bad"),
            format!("size={size:#x} avail={avail:#x}"),
        );
    };
    let Some(gva) = base_gva.checked_add(bind.offset) else {
        return miss(
            ComputeStatus::MissingBuffer("compute_stage_buf_gva_overflow"),
            format!("base={base_gva:#x} size={size:#x}"),
        );
    };
    let mut bytes = vec![0u8; want];
    if let Err(e) = gva_mem::read_task_gva_fallback(
        host,
        &state.tasks,
        task_id,
        gva,
        &mut bytes,
        state.page_shift,
    ) {
        // Full walk diagnosis on one line — max learn from a single product boot.
        let walk = gva_mem::diagnose_gva_walk(host, &state.tasks, task_id, gva, state.page_shift);
        // Also probe object base (no offset) in case only the offset page fails.
        let base_walk = if gva != base_gva {
            gva_mem::diagnose_gva_walk(host, &state.tasks, task_id, base_gva, state.page_shift)
        } else {
            String::new()
        };
        crate::observe::fail(format!(
            "compute_stage_buf_gva task={task_id} ref={} base={base_gva:#x} off={:#x} gva={gva:#x} want={want} size={size:#x} page_shift={} err={e:?} | {walk}{}",
            bind.buffer_ref,
            bind.offset,
            state.page_shift,
            if base_walk.is_empty() {
                String::new()
            } else {
                format!(" | base_walk {base_walk}")
            }
        ));
        return Err(ComputeStatus::GuestIo("compute_stage_buf_gva_read"));
    }
    Ok(StagedBuffer {
        bind: bind.clone(),
        gva,
        bytes,
    })
}

enum TextureWriteback {
    None,
    Linear {
        texture_ref: u32,
        gva: u64,
        pixel_format: u16,
        row_stride: u64,
        width: u32,
        height: u32,
        bpp: u32,
    },
    Type11 {
        mapping_id: u32,
        surface_offset: u64,
        surface_bpr: u32,
        span_end: u64,
        width: u32,
        height: u32,
        bpp: u32,
    },
}

pub(crate) struct StagedTexture {
    pub binding: u32,
    /// Raw Metal pixel format from the exact texture/view descriptor.
    pub pixel_format: u16,
    /// Product storage-selector ABI when this Metal format is storage-capable.
    /// Sample-only formats such as RGB9E5Float intentionally have no selector.
    pub storage_selector: Option<u32>,
    pub width: u32,
    pub height: u32,
    pub bytes: Vec<u8>,
    pub is_storage: bool,
    residency: Option<ComputeStorageResidencyCandidate>,
    /// Stage-time guest read skipped (resident generation verified); `bytes`
    /// is a zero placeholder the engine must never seed.
    seed_skipped: bool,
    /// Sampled input whose window the engine already holds GPU-resident (a
    /// prior dispatch's storage output at this generation): the guest read was
    /// skipped, `bytes` is a zero placeholder, and the engine must seed the
    /// sampled image by copy-on-sample from the resident (never the bytes).
    sample_resident: Option<(crate::model::ComputeStorageResidencyKey, u32)>,
    writeback: TextureWriteback,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct ComputeStorageResidencyCandidate {
    key: crate::model::ComputeStorageResidencyKey,
    seed_generation: u32,
}

/// Deferred-readback policy for a compute storage output. Deferring keeps the
/// dispatch result GPU-resident-only (guest pages stale until a flush choke
/// point), which is only a safe authority when the device grants
/// `deferred_gpu_only_content` — portability-subset (MoltenVK) devices must
/// write guest pages synchronously instead.
fn compute_defer_readback_allowed(
    deferred_gpu_only_content: bool,
    has_residency: bool,
    writeback_deferrable: bool,
) -> bool {
    deferred_gpu_only_content && has_residency && writeback_deferrable
}

fn storage_residency_opportunity(
    state: &DeviceState,
    textures: &[StagedTexture],
) -> (usize, usize, u64, u64) {
    let mut eligible = 0usize;
    let mut hits = 0usize;
    let mut eligible_bytes = 0u64;
    let mut hit_bytes = 0u64;
    for texture in textures.iter().filter(|texture| texture.is_storage) {
        let Some(candidate) = texture.residency else {
            continue;
        };
        if candidate.key.is_linear() {
            continue;
        }
        let bytes = u64::try_from(texture.bytes.len()).unwrap_or(u64::MAX);
        eligible += 1;
        eligible_bytes = eligible_bytes.saturating_add(bytes);
        if state.compute_storage_residency.get(&candidate.key) == Some(&candidate.seed_generation) {
            hits += 1;
            hit_bytes = hit_bytes.saturating_add(bytes);
        }
    }
    (eligible, hits, eligible_bytes, hit_bytes)
}

fn log_storage_residency_opportunity(
    pipe: u32,
    eligible: usize,
    hits: usize,
    eligible_bytes: u64,
    hit_bytes: u64,
) {
    if eligible == 0 {
        return;
    }
    crate::observe::off(format!(
        "compute_storage_residency reason=generation_match action=measure pipe={pipe} eligible={eligible} hits={hits} eligible_bytes={eligible_bytes} hit_bytes={hit_bytes}"
    ));
}

/// Bound on mirror entries per mapping: a ping-pong canvas needs 2, planar
/// layouts a few more; anything beyond is stale-key debris worth dropping.
const STORAGE_RESIDENCY_WINDOWS_PER_MAPPING: usize = 8;

fn note_storage_residency_writeback(state: &mut DeviceState, texture: &StagedTexture) {
    let Some(candidate) = texture.residency else {
        return;
    };
    // Linear windows keep their authority in the host_linear_textures entry
    // (resident_gen), never in the mapping-keyed mirror.
    if candidate.key.is_linear() {
        return;
    }
    if candidate.key.is_heap() {
        state.compute_storage_residency.insert(
            candidate.key,
            next_mapping_content_generation(candidate.seed_generation),
        );
        return;
    }
    // The engine registered the resident at exactly next(seed_generation)
    // (ComputeStorageResidency::output_generation). The mirror must store the
    // same currency — not the mapping-level content generation — so disjoint
    // sibling-window writebacks (ping-pong canvases) cannot desync the pair.
    let generation = next_mapping_content_generation(candidate.seed_generation);
    // Drop intersecting windows (normally already gone via the writeback's
    // exact-window invalidation — kept here as defense in depth); keep
    // disjoint siblings (ping-pong canvases) but bound the count.
    let mapping_id = candidate.key.mapping_id;
    state.invalidate_storage_residency_window(
        mapping_id,
        candidate.key.surface_offset,
        candidate.key.span_end,
    );
    let siblings: Vec<crate::model::ComputeStorageResidencyKey> = state
        .compute_storage_residency
        .keys()
        .filter(|key| key.mapping_id == mapping_id && **key != candidate.key)
        .cloned()
        .collect();
    for victim in siblings
        .iter()
        .take((siblings.len() + 1).saturating_sub(STORAGE_RESIDENCY_WINDOWS_PER_MAPPING))
    {
        state.compute_storage_residency.remove(victim);
    }
    state
        .compute_storage_residency
        .insert(candidate.key, generation);
}

fn next_mapping_content_generation(current: u32) -> u32 {
    let next = current.wrapping_add(1);
    if next == 0 {
        1
    } else {
        next
    }
}

/// Measure the full-screen compute-result boundary before guest writeback.
///
/// This is diagnosis only: it never changes dispatch, writeback, or present
/// behavior. Pair a nonzero `output_tex` destination with later
/// `empty_sample kind=display` lines to localize loss after shader execution.
/// `fused_stats` = `(rgb_nz, rgb_max)` already gathered during the engine's
/// readback copy (identical pixel semantics to `rgba_rgb_stats`).
fn log_compute_output_texture(pipe: u32, tex: &StagedTexture, fused_stats: Option<(usize, u8)>) {
    let Some(row_bytes) =
        pixel_format::bytes_per_pixel(tex.pixel_format).and_then(|bpp| tex.width.checked_mul(bpp))
    else {
        return;
    };
    // Reinterpreted views may be narrower in texels while covering the same
    // physical row (for example 320 RGBA32Uint texels == 1280 BGRA8 pixels).
    if row_bytes < 1280 * pixel_format::RGBA8_BPP || tex.height < 720 {
        return;
    }
    // Prefer the census the engine fused into the readback copy; a second
    // full scan of a display-sized image costs 2–3 ms on the stamp path.
    let (nz, max_rgb) = fused_stats.unwrap_or_else(|| {
        let (nz, max_rgb, _) = crate::observe::rgba_rgb_stats(&tex.bytes);
        (nz, max_rgb)
    });
    let destination = match &tex.writeback {
        TextureWriteback::None => "none".to_string(),
        TextureWriteback::Linear { gva, .. } => format!("linear gva={gva:#x}"),
        TextureWriteback::Type11 { mapping_id, .. } => format!("type11 mid={mapping_id}"),
    };
    crate::observe::off(format!(
        "compute_linux output_tex pipe={pipe} bind={} simg={} fmt={:#x} dst={destination} {}x{} rgb_nz={nz} max_rgb={max_rgb}",
        tex.binding,
        tex.storage_selector
            .map(|selector| selector.to_string())
            .unwrap_or_else(|| "none".into()),
        tex.pixel_format,
        tex.width,
        tex.height
    ));
}

/// Census marker for a storage image the engine wrote back GPU-direct:
/// content stats are not measured (nothing crossed the device→host
/// boundary). Same display-size gating as [`log_compute_output_texture`].
#[cfg(feature = "backend-vulkan")]
fn log_compute_output_texture_direct(pipe: u32, tex: &StagedTexture) {
    let Some(row_bytes) =
        pixel_format::bytes_per_pixel(tex.pixel_format).and_then(|bpp| tex.width.checked_mul(bpp))
    else {
        return;
    };
    if row_bytes < 1280 * pixel_format::RGBA8_BPP || tex.height < 720 {
        return;
    }
    let destination = match &tex.writeback {
        TextureWriteback::None => "none".to_string(),
        TextureWriteback::Linear { gva, .. } => format!("linear gva={gva:#x}"),
        TextureWriteback::Type11 { mapping_id, .. } => format!("type11 mid={mapping_id}"),
    };
    crate::observe::off(format!(
        "compute_linux output_tex pipe={pipe} bind={} simg={} fmt={:#x} dst={destination} {}x{} census=direct",
        tex.binding,
        tex.storage_selector
            .map(|selector| selector.to_string())
            .unwrap_or_else(|| "none".into()),
        tex.pixel_format,
        tex.width,
        tex.height
    ));
}

/// Measure storage-image seed traffic by structurally reflected content access.
///
/// `write_only` is intentionally still seeded: access alone does not prove a
/// dispatch overwrites every texel. The proxy makes that retained transfer
/// cost visible while preserving partial-write semantics.
fn log_storage_image_access(pipe: u32, binding: u32, access: &str, bytes: u64) {
    crate::observe::off(format!(
        "compute_linux storage_access pipe={pipe} bind={binding} access={access} seed=1 bytes={bytes}"
    ));
}

/// Load tight raw texels for a compute texture binding (type-2/3, type-5→surface, or type-11).
///
/// Type-5 (`RefTextureHandle`) is the live CI wallpaper path (`compute_stage_tex … ot=5`).
/// RE (type-5 wire + metal_draw sample path): surfaceID@0 is a type-4 object id (= mapping
/// mid). Product draw samples call [`objects::ensure_surface_for_present`] on that id and
/// stage from the **mapping registry**, never re-resolving the surface id through the
/// compute task's object list (that list uses a separate texture-ref namespace — live
/// ensure=1 then MissingTexture/GuestIo class when `resolve_type11_ref(task, sid)` hit a
/// different type-11 slot).
/// Local handoff dir for metal2vulkan failure artifacts. `$REIMS_VGPU_M2V_FAIL_DIR`
/// overrides; else `$REIMS_VGPU_REPO_ROOT/m2v-handoff/artifacts` (gitignored); else
/// `/tmp/reims-vgpu-m2v-compute-fails`.
fn m2v_handoff_dir_from_env(
    fail_dir: Option<std::ffi::OsString>,
    repo_root: Option<std::ffi::OsString>,
) -> std::path::PathBuf {
    fail_dir
        .map(std::path::PathBuf::from)
        .or_else(|| repo_root.map(|r| std::path::PathBuf::from(r).join("m2v-handoff/artifacts")))
        .unwrap_or_else(|| std::path::PathBuf::from("/tmp/reims-vgpu-m2v-compute-fails"))
}

pub(crate) fn m2v_handoff_dir() -> std::path::PathBuf {
    m2v_handoff_dir_from_env(
        std::env::var_os("REIMS_VGPU_M2V_FAIL_DIR"),
        std::env::var_os("REIMS_VGPU_REPO_ROOT"),
    )
}

/// Persist the exact inputs of a compute kernel that failed downstream, for
/// off-VM handoff to the metal2vulkan agent.
///
/// Writes, under [`m2v_handoff_dir`], stem `pipe<N>.<reason>`:
/// - `.mtlb` — raw MTLB container,
/// - `.air`  — extracted bitcode member (the exact bytes the m2v Kernel stage
///   consumes; disassemble with `llvm-dis`),
/// - `.spv`  — the translated SPIR-V when the failure is *post*-translation
///   (reflection / storage-format), so the agent sees what m2v emitted
///   (disassemble with `spirv-dis`); omitted when translation itself failed,
/// - `.txt`  — `reason`, dims, blob lengths, and the caller's `meta` line.
///
/// One dump per `(pipe, reason)` per boot: a translate-fail and a
/// reflection-fail for the same pipe both land, but a hot per-frame reject does
/// not spam.
fn dump_kernel_handoff(
    pipe: u32,
    reason: &str,
    mtlb: &[u8],
    air: &[u8],
    spirv: Option<&[u32]>,
    meta: &str,
) {
    use std::collections::HashSet;
    use std::sync::Mutex;
    static SEEN: Mutex<Option<HashSet<(u32, String)>>> = Mutex::new(None);
    {
        let mut g = SEEN.lock().unwrap_or_else(|e| e.into_inner());
        let set = g.get_or_insert_with(HashSet::new);
        if !set.insert((pipe, reason.to_string())) {
            return;
        }
    }
    let dir = m2v_handoff_dir();
    if std::fs::create_dir_all(&dir).is_err() {
        return;
    }
    let stem = format!("pipe{pipe}.{reason}");
    let _ = std::fs::write(dir.join(format!("{stem}.mtlb")), mtlb);
    let _ = std::fs::write(dir.join(format!("{stem}.air")), air);
    let spv_len = if let Some(words) = spirv {
        let mut bytes = Vec::with_capacity(words.len().saturating_mul(4));
        for w in words {
            bytes.extend_from_slice(&w.to_le_bytes());
        }
        let _ = std::fs::write(dir.join(format!("{stem}.spv")), &bytes);
        bytes.len()
    } else {
        0
    };
    let _ = std::fs::write(
        dir.join(format!("{stem}.txt")),
        format!(
            "pipe={pipe}\nreason={reason}\nmtlb_len={}\nair_len={}\nspv_len={spv_len}\n{meta}\n",
            mtlb.len(),
            air.len()
        ),
    );
    crate::observe::fail(format!(
        "compute_linux m2v_dump pipe={pipe} reason={reason} dir={} air_len={} spv_len={spv_len}",
        dir.display(),
        air.len()
    ));
}

pub(crate) fn stage_texture_raw<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    task_id: u32,
    texture_ref: u32,
    binding: u32,
    is_storage: bool,
) -> Result<StagedTexture, ComputeStatus> {
    // Type-5 RefTextureHandle → surface_id (live CI binds ot5).
    let mut stage_ref = texture_ref;
    let mut from_type5 = false;
    let mut from_type4_direct = false;
    let mut type5_record: Option<objects::Type5TextureView> = None;
    let mut view_level = 0;
    let mut view_pixel_format = None;
    let mut heap_texture = None;
    // A linear texture object (type-2/3) must resolve through its own
    // descriptor, never through the mapping registry: its numeric ref shares
    // the id space with type-4 surface mids, so the `mappings.contains(ref)`
    // fallback below would wrongly grab a same-numbered surface (live class:
    // `ref=N ot=2` dragged into the type-11 path and failing silently against
    // the biplanar wallpaper mid). Same collision the type-5 path documents.
    // Resolve the object-list entry once: `ref_is_linear` and the type5/type4
    // classification below both read it for the same ref, and the guest object
    // list is immutable for the life of the dispatch (the device never writes
    // those pages). `ListObjectEntry` is `Copy`, so one guest-DMA read+decode
    // serves both instead of two.
    let ref_entry = objects::lookup_list_entry(state, host, task_id, texture_ref);
    if let Some(entry) = ref_entry {
        if entry.object_type == OBJECT_TYPE_TEXTURE_VIEW {
            let Some(desc) = objects::read_descriptor(state, host, task_id, &entry) else {
                crate::observe::fail(format!(
                    "compute_stage_tex view_fail reason=no_desc ref={texture_ref} desc_len={}",
                    entry.descriptor_length
                ));
                return Err(ComputeStatus::MissingTexture(
                    "compute_stage_tex_view_no_desc",
                ));
            };
            let opcode = texture_type8_opcode(&desc).unwrap_or(0);
            if opcode == HEAP_TEXTURE_OPCODE {
                if desc.len() != HEAP_TEXTURE_LEN {
                    crate::observe::fail(format!(
                        "compute_stage_tex heap_fail reason=bad_len ref={texture_ref} len={} expected={HEAP_TEXTURE_LEN}",
                        desc.len(),
                    ));
                    return Err(ComputeStatus::MissingTexture(
                        "compute_stage_tex_heap_bad_len",
                    ));
                }
                let heap_ref = ld32(&desc[HEAP_TEXTURE_HEAP_REF..]);
                let use_offset = ld32(&desc[HEAP_TEXTURE_USE_OFFSET..]);
                let offset = crate::contract::endian::ld64(&desc[HEAP_TEXTURE_OFFSET..]);
                if heap_ref == 0 {
                    crate::observe::fail(format!(
                        "compute_stage_tex heap_fail reason=zero_heap ref={texture_ref}"
                    ));
                    return Err(ComputeStatus::MissingTexture(
                        "compute_stage_tex_heap_zero_ref",
                    ));
                }
                if use_offset > 1 {
                    crate::observe::fail(format!(
                        "compute_stage_tex heap_fail reason=bad_use_offset ref={texture_ref} heap={heap_ref} use_offset={use_offset} offset={offset:#x}"
                    ));
                    return Err(ComputeStatus::Unsupported("compute_heap_use_offset"));
                }
                let descriptor =
                    match crate::runtime::heap_query::decode_serialized_texture_descriptor(
                        &desc[HEAP_TEXTURE_DESCRIPTOR..HEAP_TEXTURE_USE_OFFSET],
                    ) {
                        Ok(descriptor) => descriptor,
                        Err(error) => {
                            crate::observe::Emit::decline("compute_stage_tex_heap", &error)
                                .field("ref", texture_ref)
                                .field("heap", heap_ref)
                                .field("use_offset", use_offset)
                                .field("offset", format!("{offset:#x}"))
                                .fail();
                            return Err(ComputeStatus::MissingTexture(
                                "compute_stage_tex_heap_desc_decode",
                            ));
                        }
                    };
                heap_texture = Some((heap_ref, use_offset != 0, offset, descriptor));
            }
            if heap_texture.is_some() {
                // Heap textures are complete resource objects, not texture
                // views. Their backing is a host GPU residency identity.
            } else if opcode == TEXTURE_VIEW_OPCODE_BUFFER_TEXTURE {
                crate::observe::fail(format!(
                    "compute_stage_tex view_fail reason=buffer_texture_unsupported ref={texture_ref} opcode={opcode} desc_len={}",
                    desc.len()
                ));
                return Err(ComputeStatus::Unsupported(
                    "compute_buffer_texture_unsupported",
                ));
            } else {
                let view = match crate::runtime::metal_draw::resolve_texture_view_reasoned(
                    state,
                    host,
                    task_id,
                    texture_ref,
                ) {
                    Ok(view) => view,
                    Err(reason) => {
                        crate::observe::Emit::decline("compute_stage_tex_view_resolve", &reason)
                            .field("ref", texture_ref)
                            .field("opcode", format!("{opcode:#x}"))
                            .fail_once(texture_ref as u64);
                        return Err(ComputeStatus::MissingTexture(
                            "compute_stage_tex_view_resolve",
                        ));
                    }
                };
                if view
                    .swizzle
                    .as_ref()
                    .is_some_and(|plan| !pixel_format::swizzle_is_identity(plan))
                {
                    crate::observe::fail(format!(
                        "compute_stage_tex view_fail reason=swizzle_unsupported ref={texture_ref} base={} opcode={opcode} storage={}",
                        view.base_texture_ref, is_storage as u8
                    ));
                    return Err(ComputeStatus::Unsupported(
                        "compute_view_swizzle_unsupported",
                    ));
                }
                stage_ref = view.base_texture_ref;
                view_level = view.level;
                view_pixel_format = view.pixel_format;
            }
        }
    }
    if let Some((heap_ref, use_offset, offset, descriptor)) = heap_texture {
        if descriptor.texture_type != 2
            || descriptor.depth != 1
            || descriptor.mipmap_level_count != 1
            || descriptor.sample_count != 1
            || descriptor.array_length != 1
        {
            crate::observe::fail(format!(
                "compute_stage_tex heap_fail reason=shape ref={texture_ref} heap={heap_ref} type={} dims={}x{}x{} mips={} samples={} array={} use_offset={} offset={offset:#x}",
                descriptor.texture_type,
                descriptor.width,
                descriptor.height,
                descriptor.depth,
                descriptor.mipmap_level_count,
                descriptor.sample_count,
                descriptor.array_length,
                use_offset as u8
            ));
            return Err(ComputeStatus::Unsupported("compute_heap_shape"));
        }
        let (width, height, format) =
            (descriptor.width, descriptor.height, descriptor.pixel_format);
        let Some(bpp) = pixel_format::bytes_per_pixel(format) else {
            crate::observe::fail(format!(
                "compute_stage_tex heap_fail reason=fmt_bytes ref={texture_ref} heap={heap_ref} fmt={format:#x} {width}x{height}"
            ));
            return Err(ComputeStatus::Unsupported("compute_heap_fmt_bytes"));
        };
        let storage_selector =
            pixel_format::storage_selector(format).map(|(selector, selector_bpp)| {
                debug_assert_eq!(selector_bpp, bpp);
                selector as u32
            });
        if is_storage && storage_selector.is_none() {
            crate::observe::fail(format!(
                "compute_stage_tex heap_fail reason=fmt_storage ref={texture_ref} heap={heap_ref} fmt={format:#x} {width}x{height}"
            ));
            return Err(ComputeStatus::Unsupported("compute_heap_fmt_storage"));
        }
        let Some(need) = (width as usize)
            .checked_mul(height as usize)
            .and_then(|texels| texels.checked_mul(bpp as usize))
        else {
            crate::observe::fail(format!(
                "compute_stage_tex heap_fail reason=host_len ref={texture_ref} heap={heap_ref} fmt={format:#x} {width}x{height} bpp={bpp}"
            ));
            return Err(ComputeStatus::Unsupported("compute_heap_host_len"));
        };
        let key = crate::model::ComputeStorageResidencyKey::heap(
            task_id,
            texture_ref,
            width,
            height,
            format,
        );
        let mut seed_generation = 0;
        let mut seed_skipped = false;
        let mut sample_resident = None;
        #[cfg(feature = "backend-vulkan")]
        {
            if let Some(&generation) = state.compute_storage_residency.get(&key) {
                if is_storage
                    && crate::backend::vulkan::engine::compute_resident_storage_generation(&key)
                        == Some(generation)
                {
                    seed_generation = generation;
                    seed_skipped = true;
                } else if !is_storage {
                    if let Some((engine_generation, engine_format)) =
                        crate::backend::vulkan::engine::compute_resident_sample_source(&key)
                    {
                        if engine_generation == generation
                            && mtl_to_engine_sampled(format)
                                .is_some_and(|f| f.vk_format() == engine_format.vk_format())
                        {
                            sample_resident = Some((key, generation));
                        }
                    }
                }
                if !seed_skipped && sample_resident.is_none() {
                    crate::observe::fail(format!(
                        "compute_stage_tex heap_fail reason=resident_lost ref={texture_ref} heap={heap_ref} fmt={format:#x} {width}x{height} gen={generation} use_offset={} offset={offset:#x}",
                        use_offset as u8
                    ));
                    return Err(ComputeStatus::MissingTexture(
                        "compute_stage_tex_heap_resident_lost",
                    ));
                }
            }
        }
        crate::observe::off(format!(
            "compute_stage_tex heap_ok ref={texture_ref} heap={heap_ref} fmt={format:#x} {width}x{height} storage={} seed_gen={seed_generation} resident_sample={} use_offset={} offset={offset:#x}",
            is_storage as u8,
            sample_resident.is_some() as u8,
            use_offset as u8
        ));
        return Ok(StagedTexture {
            binding,
            pixel_format: format,
            storage_selector,
            width,
            height,
            bytes: vec![0; need],
            is_storage,
            residency: is_storage.then_some(ComputeStorageResidencyCandidate {
                key,
                seed_generation,
            }),
            seed_skipped,
            sample_resident,
            writeback: TextureWriteback::None,
        });
    }
    let stage_entry = objects::lookup_list_entry(state, host, task_id, stage_ref);
    let ref_is_linear = stage_entry
        .map(|e| {
            e.object_type == OBJECT_TYPE_TEXTURE || e.object_type == OBJECT_TYPE_TEXTURE_VARIANT
        })
        .unwrap_or(false);
    if let Some(entry) = stage_entry {
        if entry.object_type == objects::OBJECT_TYPE_REF_TEXTURE {
            if let Some(desc) = objects::read_descriptor(state, host, task_id, &entry) {
                if desc.len() >= objects::TYPE5_MIN_LEN {
                    let sid = ld32(&desc[objects::TYPE5_SURFACE_ID..]);
                    if sid != 0 {
                        stage_ref = sid;
                        from_type5 = true;
                        type5_record = objects::decode_type5_texture_view(&desc);
                        let ok = objects::ensure_surface_for_present(state, host, sid);
                        // Per-bind type-5 descriptor RE census (args@+8 holds the
                        // serialized plane texture; product stage uses mapping geom
                        // only today). This is measurement, not a failure — it fired
                        // ~600×/boot on the always-on sink (same descriptor re-dumped
                        // per bind, no dedup), drowning genuine failures. Verbose-gated;
                        // build the head-hex only when REIMS_VGPU_DRAW_LOG is on. A genuine
                        // ensure failure surfaces downstream as `MissingTexture` (the
                        // mapping lookup below misses), so no always-on line is lost.
                        if crate::observe::draw_log_enabled() {
                            let field = if desc.len() >= objects::TYPE5_FIELD + 4 {
                                ld32(&desc[objects::TYPE5_FIELD..])
                            } else {
                                0
                            };
                            let args_n = desc.len().saturating_sub(objects::TYPE5_ARGS);
                            let mut args_hex = String::new();
                            if args_n > 0 {
                                let n = args_n.min(48);
                                args_hex.reserve(n * 2);
                                for b in &desc[objects::TYPE5_ARGS..objects::TYPE5_ARGS + n] {
                                    use std::fmt::Write as _;
                                    let _ = write!(args_hex, "{b:02x}");
                                }
                                if args_n > n {
                                    args_hex.push('…');
                                }
                            }
                            crate::observe::line(format!(
                                "compute_stage_tex type5 ref={texture_ref} sid={sid} ensure={} field={field:#x} desc_len={} args_n={args_n} args_hex={args_hex}",
                                ok as u8,
                                desc.len(),
                            ));
                        }
                    }
                }
            }
        } else if entry.object_type == objects::OBJECT_TYPE_SURFACE {
            // Direct type-4 surface bind (same id space as present mids).
            from_type4_direct = true;
            let _ = objects::ensure_surface_for_present(state, host, stage_ref);
        }
    }

    // Type-5 / direct type-4: surface id **is** the mapping mid. Never call
    // resolve_type11_ref(task, sid) — task object-list indices collide with texture refs.
    let mapping_id_opt = if from_type5 || from_type4_direct {
        if stage_ref != 0 && state.mappings.contains_key(&stage_ref) {
            Some(stage_ref)
        } else {
            None
        }
    } else if ref_is_linear {
        // Linear texture: never fall back to the mapping registry (id-space
        // collision with type-4 surface mids). Force the type-2/3 path.
        None
    } else {
        objects::resolve_type11_ref(state, host, task_id, stage_ref).or_else(|| {
            if stage_ref != 0 && state.mappings.contains_key(&stage_ref) {
                Some(stage_ref)
            } else {
                None
            }
        })
    };
    if mapping_id_opt.is_none() && from_type5 {
        crate::observe::fail(format!(
            "compute_stage_tex type5_no_map ref={texture_ref} sid={stage_ref}"
        ));
        return Err(ComputeStatus::MissingTexture(
            "compute_stage_tex_type5_no_map",
        ));
    }
    if let Some(mapping_id) = mapping_id_opt {
        let _ = mapper::ensure_resolved_for_scanout(state, host, mapping_id);
        // Geom/format: a type-5 record is the exact Metal texture view over
        // the IOSurface bytes. It is authoritative even for a stageable
        // single-plane mapping: the live BGRA8 desktop target is exposed as a
        // row-byte-equivalent, quarter-width RGBA32Uint view. Type-4 direct
        // refs use base mapping geometry. Type-11 refs may prefer the
        // IOSurface descriptor on this task's object list.
        if view_level != 0 {
            crate::observe::fail(format!(
                "compute_stage_tex view_fail reason=type11_mip ref={texture_ref} base={stage_ref} level={view_level} mapping={mapping_id}"
            ));
            return Err(ComputeStatus::Unsupported("compute_view_type11_mip"));
        }
        let (width, height, format) = if from_type5 || from_type4_direct {
            let m = state
                .mappings
                .get(&mapping_id)
                .ok_or(ComputeStatus::MissingTexture(
                    "compute_stage_tex_mapping_gone",
                ))?;
            let multiplanar = objects::mapping_is_multiplanar(m);
            let mapping_stageable =
                m.has_geom && m.width != 0 && m.height != 0 && m.format != 0 && !multiplanar;
            if let Some(rec) = type5_record {
                // `type11_sample_window` below matches actual plane records by
                // geometry+bpe and otherwise verifies a packed row-compatible
                // view over the same bytes. Per-bind measurement (view vs base
                // geom), not a failure — verbose-gated to keep the always-on sink
                // for genuine failures.
                crate::observe::line(format!(
                    "compute_stage_tex type5_view mapping={mapping_id} view={}x{} fmt={:#x} base={}x{} fmt={:#x} multiplanar={}",
                    rec.width,
                    rec.height,
                    rec.pixel_format,
                    m.width,
                    m.height,
                    m.format,
                    multiplanar as u8
                ));
                (rec.width, rec.height, rec.pixel_format)
            } else if !mapping_stageable {
                if !m.has_geom || m.width == 0 || m.height == 0 {
                    crate::observe::fail(format!(
                        "compute_stage_tex type11_fail reason=no_geom mapping={mapping_id} pages={} has_geom={}",
                        m.page_entries.len(),
                        m.has_geom as u8
                    ));
                    return Err(ComputeStatus::MissingTexture(
                        "compute_stage_tex_type11_no_geom",
                    ));
                } else if multiplanar {
                    // Multi-plane IOSurface without a plane record: fail closed,
                    // do not invent BGRA sample of the whole surface.
                    crate::observe::fail(format!(
                        "compute_stage_tex type11_fail reason=multiplane mapping={mapping_id} {}x{} fmt={:#x} pages={} (no type-5 plane record)",
                        m.width,
                        m.height,
                        m.format,
                        m.page_entries.len()
                    ));
                    return Err(ComputeStatus::Unsupported("stage_tex_multiplane_no_plane"));
                } else {
                    // Single-plane unknown format: fail closed (no BGRA invent).
                    crate::observe::fail(format!(
                        "compute_stage_tex type11_fail reason=fmt_unknown mapping={mapping_id} {}x{} pages={}",
                        m.width,
                        m.height,
                        m.page_entries.len()
                    ));
                    return Err(ComputeStatus::Unsupported("stage_tex_fmt_unknown"));
                }
            } else {
                (m.width, m.height, m.format)
            }
        } else if let Some(entry) = objects::lookup_list_entry(state, host, task_id, stage_ref) {
            if let Some(desc_bytes) = objects::read_descriptor(state, host, task_id, &entry) {
                if let Ok(ResourceDescriptor::IOSurfaceTexture {
                    width,
                    height,
                    pixel_format,
                    ..
                }) = crate::runtime::decode::resource::decode_iosurface_texture_descriptor(
                    &desc_bytes,
                ) {
                    let format = if pixel_format != 0 {
                        pixel_format
                    } else {
                        pixel_format::MTL_FORMAT_BGRA8_UNORM
                    };
                    (width, height, format)
                } else {
                    let m =
                        state
                            .mappings
                            .get(&mapping_id)
                            .ok_or(ComputeStatus::MissingTexture(
                                "compute_stage_tex_mapping_gone",
                            ))?;
                    if !m.has_geom || m.width == 0 || m.height == 0 {
                        return Err(ComputeStatus::MissingTexture(
                            "compute_stage_tex_mapping_no_geom",
                        ));
                    }
                    let format = if m.format != 0 {
                        m.format
                    } else {
                        pixel_format::MTL_FORMAT_BGRA8_UNORM
                    };
                    (m.width, m.height, format)
                }
            } else {
                let m = state
                    .mappings
                    .get(&mapping_id)
                    .ok_or(ComputeStatus::MissingTexture(
                        "compute_stage_tex_mapping_gone",
                    ))?;
                if !m.has_geom || m.width == 0 || m.height == 0 {
                    return Err(ComputeStatus::MissingTexture(
                        "compute_stage_tex_mapping_no_geom",
                    ));
                }
                let format = if m.format != 0 {
                    m.format
                } else {
                    pixel_format::MTL_FORMAT_BGRA8_UNORM
                };
                (m.width, m.height, format)
            }
        } else {
            let m = state
                .mappings
                .get(&mapping_id)
                .ok_or(ComputeStatus::MissingTexture(
                    "compute_stage_tex_mapping_gone",
                ))?;
            if !m.has_geom || m.width == 0 || m.height == 0 {
                return Err(ComputeStatus::MissingTexture(
                    "compute_stage_tex_mapping_no_geom",
                ));
            }
            let format = if m.format != 0 {
                m.format
            } else {
                pixel_format::MTL_FORMAT_BGRA8_UNORM
            };
            (m.width, m.height, format)
        };
        if width == 0 || height == 0 {
            return Err(ComputeStatus::MissingTexture("compute_stage_tex_zero_geom"));
        }
        // sRGB color-renderable surfaces stage as unorm storage (same bpp).
        let Some(view_format) =
            crate::runtime::metal_draw::effective_view_sample_format(format, view_pixel_format)
        else {
            crate::observe::fail(format!(
                "compute_stage_tex view_fail reason=format_incompatible ref={texture_ref} base={stage_ref} base_fmt={format:#x} view_fmt={view_pixel_format:?} mapping={mapping_id}"
            ));
            return Err(ComputeStatus::Unsupported("compute_view_format"));
        };
        let stage_fmt = match view_format {
            pixel_format::MTL_FORMAT_BGRA8_UNORM_SRGB => pixel_format::MTL_FORMAT_BGRA8_UNORM,
            pixel_format::MTL_FORMAT_RGBA8_UNORM_SRGB => pixel_format::MTL_FORMAT_RGBA8_UNORM,
            other => other,
        };
        let bpp = match pixel_format::bytes_per_pixel(stage_fmt) {
            Some(v) => v,
            None => {
                crate::observe::fail(format!(
                    "compute_stage_tex type11_fail reason=fmt_bytes mapping={mapping_id} {width}x{height} fmt={format:#x}"
                ));
                return Err(ComputeStatus::Unsupported("stage_tex_fmt_bytes"));
            }
        };
        let storage_selector =
            pixel_format::storage_selector(stage_fmt).map(|(selector, selector_bpp)| {
                debug_assert_eq!(selector_bpp, bpp);
                selector as u32
            });
        if is_storage && storage_selector.is_none() {
            crate::observe::fail(format!(
                "compute_stage_tex type11_fail reason=fmt_storage mapping={mapping_id} {width}x{height} fmt={format:#x}"
            ));
            return Err(ComputeStatus::Unsupported("stage_tex_fmt_storage"));
        }
        let m = state
            .mappings
            .get(&mapping_id)
            .ok_or(ComputeStatus::MissingTexture(
                "compute_stage_tex_mapping_gone",
            ))?;
        let map_generation = m.map_generation;
        let mut seed_generation = m.content_generation;
        let pages_n = m.page_entries.len();
        // Wire type-4 `length` (page-aligned getResidentSize), stashed as device_desc.alloc_size.
        // Independent of plane w/h and of MapMemory2 IOAccelMemory length — measure-only.
        let wire_len = crate::contract::iosurface_pages::decode_device_surface(&m.device_desc)
            .map(|s| s.alloc_size as u64)
            .unwrap_or(0);
        let (surface_offset, surface_bpr, span_end) = match mapping_write::type11_sample_window(
            m, width, height, stage_fmt,
        ) {
            Some(w) => w,
            None => {
                // Measure type4_len_vs_plane: which window path rejected (device bpr vs invent span).
                let ds = crate::contract::iosurface_pages::decode_device_surface(&m.device_desc);
                let (dw, dh, dbpr, dalloc) = ds
                    .as_ref()
                    .map(|s| (s.width, s.height, s.bytes_per_row, s.alloc_size))
                    .unwrap_or((0, 0, 0, 0));
                let invent_end =
                    crate::contract::iosurface_pages::sample_window(0, stage_fmt, width, height)
                        .map(|(_, _, e)| e)
                        .unwrap_or(0);
                crate::observe::fail(format!(
                    "compute_stage_tex type11_fail reason=window mapping={mapping_id} {width}x{height} fmt={stage_fmt:#x} pages={pages_n} wire_len={wire_len} desc={dw}x{dh} bpr={dbpr} alloc={dalloc} invent_end={invent_end}"
                ));
                return Err(ComputeStatus::MissingTexture(
                    "compute_stage_tex_type11_window",
                ));
            }
        };
        let tight = (width as u64)
            .checked_mul(bpp as u64)
            .ok_or(ComputeStatus::Unsupported("stage_tex_tight_bpr_overflow"))?
            as u32;
        if from_type5 && type5_record.is_some() {
            // Per-bind type-5 sample-window measurement, not a failure — verbose-gated
            // (was a per-bind always-on line). Genuine window failures above emit
            // `type11_fail reason=window` always-on.
            crate::observe::line(format!(
                "compute_stage_tex type5_view_window mapping={mapping_id} view={width}x{height} fmt={stage_fmt:#x} bpp={bpp} tight={tight} surface_off={surface_offset} surface_bpr={surface_bpr} span_end={span_end}"
            ));
        }
        let need_u64 = (tight as u64)
            .checked_mul(height as u64)
            .ok_or(ComputeStatus::Unsupported("stage_tex_need_overflow"))?;
        let Some(need) = host_alloc_len(need_u64) else {
            crate::observe::fail(format!(
                "compute_stage_tex type11_fail reason=host_len mapping={mapping_id} need={need_u64}"
            ));
            return Err(ComputeStatus::Unsupported("stage_tex_host_len"));
        };
        let page_bytes = (pages_n as u64).saturating_mul(1u64 << state.page_shift);
        if page_bytes < span_end {
            crate::observe::fail(format!(
                "compute_stage_tex type11_fail reason=span mapping={mapping_id} {width}x{height} pages={pages_n} page_bytes={page_bytes} span_end={span_end} bpr={surface_bpr} wire_len={wire_len}"
            ));
            return Err(ComputeStatus::GuestIo("compute_stage_tex_type11_span"));
        }
        let residency_key = crate::model::ComputeStorageResidencyKey {
            mapping_id,
            map_generation,
            surface_offset,
            surface_bpr,
            span_end,
            width,
            height,
            pixel_format: stage_fmt,
            texture_ref: 0,
        };
        // Chained-dispatch restage skip: when guest pages still hold exactly
        // our own last writeback for THIS WINDOW (mirror entry survives only
        // while no intersecting guest write lands — exact-window invalidation
        // in mapping_write/mapper) AND the engine still holds the resident
        // image at the mirror's generation, reading ~15 MB from guest pages
        // reproduces what the GPU already has. The mapping-level content
        // generation may have advanced via disjoint sibling windows
        // (ping-pong canvases), so the gate pairs mirror↔engine directly.
        // The zero placeholder is never seeded — the engine fails visibly
        // (`compute_resident_seed_lost`) if the resident vanishes by acquire
        // time and the caller restages.
        let mut seed_skipped = false;
        #[cfg(feature = "backend-vulkan")]
        if is_storage {
            if let Some(&mirror_generation) = state.compute_storage_residency.get(&residency_key) {
                if crate::backend::vulkan::engine::compute_resident_storage_generation(
                    &residency_key,
                ) == Some(mirror_generation)
                {
                    seed_skipped = true;
                    seed_generation = mirror_generation;
                    crate::observe::off(format!(
                        "compute_stage_resident_skip mapping={mapping_id} {width}x{height} fmt={stage_fmt:#x} gen={seed_generation} bytes={need}"
                    ));
                }
            }
        }
        // Copy-on-sample skip: a sampled input of a window whose current
        // content the engine already holds GPU-resident (a prior dispatch's
        // storage output — live class: the dispatch samples the very window it
        // storage-writes) never needs the guest read. The same mirror↔engine
        // generation gate as the storage skip applies, plus vk-format equality
        // between the resident image and the sampled view the engine will
        // create — the engine's resident-bind path guards it and would fail
        // the whole request on mismatch.
        let mut sample_resident = None;
        #[cfg(feature = "backend-vulkan")]
        if !is_storage {
            if let Some(&mirror_generation) = state.compute_storage_residency.get(&residency_key) {
                if let Some((resident_generation, resident_fmt)) =
                    crate::backend::vulkan::engine::compute_resident_sample_source(&residency_key)
                {
                    if resident_generation == mirror_generation
                        && mtl_to_engine_sampled(stage_fmt)
                            .is_some_and(|f| f.vk_format() == resident_fmt.vk_format())
                    {
                        sample_resident = Some((residency_key, mirror_generation));
                        crate::observe::off(format!(
                            "compute_stage_resident_sample mapping={mapping_id} {width}x{height} fmt={stage_fmt:#x} gen={mirror_generation} bytes={need}"
                        ));
                    }
                }
            }
            // Reinterpret sibling: a resident of the SAME byte window (the
            // 5-field mapping/offset/bpr/span prefix orders the BTreeMap, so
            // the range touches only sibling views of this window) whose rows
            // are byte-identical to this view (equal row bytes, equal height)
            // serves it through the engine's image→buffer→image hop —
            // `vkCmdCopyImage` cannot cross texel-block sizes. Live class:
            // the 1928-wide BGRA8 fade view of the resident 482-wide
            // Rgba32Uint blur window (equal 7712-byte rows).
            if sample_resident.is_none() {
                if let Some(dst_fmt) = mtl_to_engine_sampled(stage_fmt) {
                    let dst_row_bytes = width as u64 * dst_fmt.bytes_per_texel() as u64;
                    let lo = crate::model::ComputeStorageResidencyKey {
                        width: 0,
                        height: 0,
                        pixel_format: 0,
                        texture_ref: 0,
                        ..residency_key
                    };
                    let hi = crate::model::ComputeStorageResidencyKey {
                        width: u32::MAX,
                        height: u32::MAX,
                        pixel_format: u16::MAX,
                        texture_ref: u32::MAX,
                        ..residency_key
                    };
                    for (sib, &mirror_generation) in state.compute_storage_residency.range(lo..=hi)
                    {
                        if *sib == residency_key || sib.height != height {
                            continue;
                        }
                        let Some((resident_generation, resident_fmt)) =
                            crate::backend::vulkan::engine::compute_resident_sample_source(sib)
                        else {
                            continue;
                        };
                        if resident_generation != mirror_generation
                            || sib.width as u64 * resident_fmt.bytes_per_texel() as u64
                                != dst_row_bytes
                        {
                            continue;
                        }
                        sample_resident = Some((*sib, mirror_generation));
                        crate::observe::off(format!(
                            "compute_stage_resident_reinterpret mapping={mapping_id} src={}x{} sfmt={:#x} dst={width}x{height} fmt={stage_fmt:#x} gen={mirror_generation} bytes={need}",
                            sib.width, sib.height, sib.pixel_format
                        ));
                        break;
                    }
                }
            }
        }
        let mut bytes = vec![0u8; need];
        if !seed_skipped
            && sample_resident.is_none()
            && !mapping_write::read_rect_raw_at(
                state,
                host,
                mapping_id,
                surface_offset,
                surface_bpr,
                span_end,
                0,
                0,
                width,
                height,
                bpp,
                &mut bytes,
                tight,
            )
        {
            crate::observe::fail(format!(
                "compute_stage_tex type11_fail reason=read mapping={mapping_id} {width}x{height} off={surface_offset} bpr={surface_bpr} span_end={span_end} pages={pages_n}"
            ));
            return Err(ComputeStatus::GuestIo("compute_stage_tex_type11_read"));
        }
        let writeback = if is_storage {
            TextureWriteback::Type11 {
                mapping_id,
                surface_offset,
                surface_bpr,
                span_end,
                width,
                height,
                bpp,
            }
        } else {
            TextureWriteback::None
        };
        if from_type5 {
            // Per-bind type-5 stage SUCCESS census — not a failure; verbose-gated
            // (was always-on, ~300/boot). Genuine type-5 stage failures above emit
            // `type11_fail reason=<slug>` always-on.
            crate::observe::line(format!(
                "compute_stage_tex type5_ok ref={texture_ref} sid={mapping_id} {width}x{height} fmt={stage_fmt:#x} pages={pages_n}"
            ));
        }
        return Ok(StagedTexture {
            binding,
            pixel_format: stage_fmt,
            storage_selector,
            width,
            height,
            bytes,
            is_storage,
            residency: is_storage.then_some(ComputeStorageResidencyCandidate {
                key: residency_key,
                seed_generation,
            }),
            seed_skipped,
            sample_resident,
            writeback,
        });
    }

    // Type-2/3 linear. Fail-visible: name which gate rejected (live class:
    // silent ot=2 MissingTexture, journal 2026-07-14 compute census).
    // The reason travels *in* the status now, so this line and the caller's
    // both name the registered slug rather than a local shorthand only this
    // closure understood.
    let linear_fail = |st: ComputeStatus, detail: String| {
        crate::observe::fail(format!(
            "compute_stage_tex linear_fail reason={} ref={texture_ref} {detail}",
            st.reason()
        ));
        Err(st)
    };
    let Some(entry) = objects::lookup_list_entry(state, host, task_id, stage_ref) else {
        return linear_fail(
            ComputeStatus::MissingTexture("compute_linear_tex_no_entry"),
            String::new(),
        );
    };
    if entry.object_type != OBJECT_TYPE_TEXTURE && entry.object_type != OBJECT_TYPE_TEXTURE_VARIANT
    {
        return linear_fail(
            ComputeStatus::MissingTexture("compute_linear_tex_not_texture"),
            format!("ot={}", entry.object_type),
        );
    }
    let Some(desc_bytes) = objects::read_descriptor(state, host, task_id, &entry) else {
        return linear_fail(
            ComputeStatus::MissingTexture("compute_linear_tex_no_desc"),
            String::new(),
        );
    };
    let Ok(tex) = decode_texture_descriptor(&desc_bytes) else {
        return linear_fail(
            ComputeStatus::MissingTexture("compute_linear_tex_desc_decode"),
            format!("len={}", desc_bytes.len()),
        );
    };
    if !tex.has_pixel_format {
        return linear_fail(
            ComputeStatus::Unsupported("linear_tex_no_fmt"),
            String::new(),
        );
    }
    let Some(stage_format) = crate::runtime::metal_draw::effective_view_sample_format(
        tex.pixel_format,
        view_pixel_format,
    ) else {
        return linear_fail(
            ComputeStatus::Unsupported("linear_tex_view_format"),
            format!(
                "base={stage_ref} base_fmt={:#x} view_fmt={view_pixel_format:?}",
                tex.pixel_format
            ),
        );
    };
    let Some(bpp) = pixel_format::bytes_per_pixel(stage_format) else {
        return linear_fail(
            ComputeStatus::Unsupported("linear_tex_fmt_bytes"),
            format!("fmt={stage_format:#x}"),
        );
    };
    let storage_selector =
        pixel_format::storage_selector(stage_format).map(|(selector, selector_bpp)| {
            debug_assert_eq!(selector_bpp, bpp);
            selector as u32
        });
    if is_storage && storage_selector.is_none() {
        return linear_fail(
            ComputeStatus::Unsupported("linear_tex_fmt_storage"),
            format!("fmt={stage_format:#x}"),
        );
    }
    let Some((gva, layout)) = tex.level_gva(view_level, state.page_shift) else {
        return linear_fail(
            ComputeStatus::MissingTexture("compute_linear_tex_no_level"),
            format!(
                "base={stage_ref} level={view_level} handle={:#x} alloc={} levels={} data_off={} page_shift={}",
                tex.handle,
                tex.allocation_size,
                tex.levels.len(),
                tex.data_offset,
                state.page_shift
            ),
        );
    };
    let w = layout.width;
    let h = layout.height;
    if w == 0 || h == 0 || layout.row_stride == 0 {
        return linear_fail(
            ComputeStatus::MissingTexture("compute_linear_tex_zero_geom"),
            format!("{w}x{h} stride={}", layout.row_stride),
        );
    }
    let Some(tight) = (w as u64).checked_mul(bpp as u64).map(|v| v as usize) else {
        return linear_fail(
            ComputeStatus::Unsupported("linear_tex_tight_overflow"),
            format!("{w}x{h} bpp={bpp}"),
        );
    };
    if layout.row_stride < tight as u64 {
        return linear_fail(
            ComputeStatus::MissingTexture("compute_linear_tex_stride_lt_tight"),
            format!("stride={} tight={tight} {w}x{h}", layout.row_stride),
        );
    }
    let Some(need) = tight.checked_mul(h as usize) else {
        return linear_fail(
            ComputeStatus::Unsupported("linear_tex_need_overflow"),
            format!("{w}x{h} bpp={bpp}"),
        );
    };
    // Linear-window residency identity — mirrors the host_linear_textures
    // entry exactly. Absent when the stride overflows the key field (no live
    // class; such a window simply stays on the bytes path).
    let span = layout.row_stride.saturating_mul(h as u64);
    let linear_key = (layout.row_stride <= u32::MAX as u64).then(|| {
        crate::model::ComputeStorageResidencyKey::linear(
            task_id,
            stage_ref,
            gva,
            layout.row_stride as u32,
            span,
            w,
            h,
            stage_format,
        )
    });
    let mut seed_skipped = false;
    let mut seed_generation = 0u32;
    let mut sample_resident = None;
    let mut bytes = vec![0u8; need];
    let mut have_bytes = false;
    // Resident-authoritative window (deferred linear writeback): consume the
    // engine resident without bytes when possible; otherwise flush it into the
    // entry first — falling through to the raw guest read would silently serve
    // the pre-chain seed pages.
    #[cfg(feature = "backend-vulkan")]
    if let (Some(key), Some(resident_gen)) = (
        linear_key,
        crate::runtime::surface_cache::linear_texture_resident_gen(
            state,
            task_id,
            stage_ref,
            gva,
            stage_format,
            w,
            h,
            layout.row_stride,
        ),
    ) {
        let mut consumed = false;
        if is_storage {
            if crate::backend::vulkan::engine::compute_resident_storage_generation(&key)
                == Some(resident_gen)
            {
                seed_skipped = true;
                seed_generation = resident_gen;
                consumed = true;
                crate::observe::off(format!(
                    "compute_stage_linear_resident_seed task={task_id} ref={texture_ref} gva={gva:#x} fmt={:#x} dims={w}x{h} gen={resident_gen}",
                    tex.pixel_format
                ));
            }
        } else if let Some((engine_gen, engine_fmt)) =
            crate::backend::vulkan::engine::compute_resident_sample_source(&key)
        {
            if engine_gen == resident_gen
                && mtl_to_engine_sampled(stage_format)
                    .is_some_and(|f| f.vk_format() == engine_fmt.vk_format())
            {
                sample_resident = Some((key, resident_gen));
                consumed = true;
                crate::observe::off(format!(
                    "compute_stage_linear_resident_sample task={task_id} ref={texture_ref} gva={gva:#x} fmt={:#x} dims={w}x{h} gen={resident_gen}",
                    stage_format
                ));
            }
        }
        if !consumed {
            // A bytes consumer (format-mismatched view, non-vulkan reuse):
            // land the resident into the cache entry (and any owed guest
            // write) through the one flush path, then serve the bytes.
            if crate::runtime::storage_flush::flush_linear_one(state, host, &key, resident_gen) {
                if let Some(cached) = crate::runtime::surface_cache::get_linear_texture(
                    state,
                    task_id,
                    stage_ref,
                    gva,
                    stage_format,
                    w,
                    h,
                    layout.row_stride,
                ) {
                    bytes.copy_from_slice(cached);
                    have_bytes = true;
                    crate::observe::off(format!(
                        "compute_linear_flush task={task_id} ref={texture_ref} gva={gva:#x} fmt={:#x} dims={w}x{h} gen={resident_gen}",
                        stage_format
                    ));
                }
            }
            if !have_bytes {
                // Deferred content is unrecoverable — name the loss, clear
                // the marker, and fall back to the coherent stale seed.
                // (flush_linear_one already fail-logged the engine loss.)
                crate::observe::fail(format!(
                    "compute_stage_tex linear_resident_lost task={task_id} ref={texture_ref} gva={gva:#x} fmt={:#x} dims={w}x{h} gen={resident_gen}",
                    stage_format
                ));
                if let Some(e) = state.host_linear_textures.get_mut(&(task_id, stage_ref)) {
                    e.resident_gen = 0;
                }
            }
        }
    }
    if seed_skipped || sample_resident.is_some() || have_bytes {
        // Engine resident serves this window; no cache/guest read.
    } else if let Some(cached) = crate::runtime::surface_cache::get_linear_texture(
        state,
        task_id,
        stage_ref,
        gva,
        stage_format,
        w,
        h,
        layout.row_stride,
    ) {
        bytes.copy_from_slice(cached);
        crate::observe::off(format!(
            "compute_stage_tex linear_cache task={task_id} ref={texture_ref} gva={gva:#x} fmt={:#x} dims={w}x{h} row_stride={}",
            stage_format, layout.row_stride
        ));
    } else {
        // Deferred-writeback flush-on-access: the bulk/row reads below walk
        // raw task GVAs and bypass the mapping-keyed hooks — land any
        // resident-authoritative window aliasing the sampled span first.
        crate::runtime::storage_flush::flush_intersecting_task_gva(
            state,
            host,
            task_id,
            gva,
            layout.row_stride.saturating_mul(h as u64),
        );
        if read_linear_texture_bulk(
            state,
            host,
            task_id,
            gva,
            layout.row_stride,
            tight,
            h,
            &mut bytes,
        ) {
            // One cached-view walk for the whole span (render-path bulk analog).
        } else {
            let mut row = vec![0u8; tight];
            for y in 0..h {
                let row_gva = gva
                    .checked_add((y as u64).checked_mul(layout.row_stride).ok_or(
                        ComputeStatus::GuestIo("compute_stage_tex_linear_row_offset"),
                    )?)
                    .ok_or(ComputeStatus::GuestIo("compute_stage_tex_linear_row_gva"))?;
                if let Err(e) = gva_mem::read_task_gva_fallback(
                    host,
                    &state.tasks,
                    task_id,
                    row_gva,
                    &mut row,
                    state.page_shift,
                ) {
                    // First failing row only — full walk status for one-boot diagnosis.
                    if y == 0 {
                        let walk = gva_mem::diagnose_gva_walk(
                            host,
                            &state.tasks,
                            task_id,
                            row_gva,
                            state.page_shift,
                        );
                        crate::observe::fail(format!(
                            "compute_stage_tex_gva task={task_id} ref={texture_ref} gva={row_gva:#x} y=0 page_shift={} err={e:?} | {walk}",
                            state.page_shift
                        ));
                    }
                    return Err(ComputeStatus::GuestIo("compute_stage_tex_linear_row_read"));
                }
                let off = (y as usize) * tight;
                bytes[off..off + tight].copy_from_slice(&row);
            }
        }
    }
    let writeback = if is_storage {
        TextureWriteback::Linear {
            texture_ref: stage_ref,
            gva,
            pixel_format: stage_format,
            row_stride: layout.row_stride,
            width: w,
            height: h,
            bpp,
        }
    } else {
        TextureWriteback::None
    };
    // Deferred-writeback candidacy: a linear storage output of a format the
    // BGRA mirror ignores keeps the engine resident authoritative — the
    // readback, cache store, and next chained upload all disappear (the
    // fade-window blur pyramid class). If the GVA is
    // mapped at writeback time (the sync path would have written guest
    // pages), the deferred-writeback arm records a flush obligation with a
    // defer-time page index so aliased raw-GVA readers land it first.
    let mut residency = None;
    #[cfg(feature = "backend-vulkan")]
    if is_storage {
        if let Some(key) = linear_key {
            if !crate::runtime::surface_cache::linear_mirrorable(stage_format) {
                let seed = if seed_skipped {
                    seed_generation
                } else {
                    state
                        .host_linear_textures
                        .get(&(task_id, stage_ref))
                        .map(|e| e.host_gen)
                        .unwrap_or(0)
                };
                residency = Some(ComputeStorageResidencyCandidate {
                    key,
                    seed_generation: seed,
                });
            }
        }
    }
    let _ = (linear_key, span, seed_generation);
    Ok(StagedTexture {
        binding,
        pixel_format: stage_format,
        storage_selector,
        width: w,
        height: h,
        bytes,
        is_storage,
        residency,
        seed_skipped,
        sample_resident,
        writeback,
    })
}

/// Read a strided linear texture span through one cached GVA view (a single
/// page-table walk for the whole texture), de-striding rows into `bytes`
/// (tight rows). Returns `false` when the span cannot be packed — the caller
/// falls back to the per-row walk. Live transition cost of the per-row walk
/// was ~8–23 ms of `stage_us` per Core Image dispatch.
#[allow(
    clippy::too_many_arguments,
    reason = "the bulk path keeps the decoded texture window and row layout explicit"
)]
fn read_linear_texture_bulk<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    task_id: u32,
    gva: u64,
    row_stride: u64,
    tight: usize,
    height: u32,
    bytes: &mut [u8],
) -> bool {
    if height == 0 || tight == 0 || bytes.len() < (height as usize).saturating_mul(tight) {
        return false;
    }
    if row_stride == tight as u64 {
        return crate::runtime::gva_view::read_span(state, host, task_id, gva, bytes);
    }
    let Some(span_len) = (height as u64 - 1)
        .checked_mul(row_stride)
        .and_then(|v| v.checked_add(tight as u64))
    else {
        return false;
    };
    let Some((ptr, avail)) =
        crate::runtime::gva_view::host_ptr_for_span(state, host, task_id, gva, span_len)
    else {
        return false;
    };
    if (avail as u64) < span_len {
        return false;
    }
    for y in 0..height as usize {
        let src = (y as u64).saturating_mul(row_stride) as usize;
        let dst = y * tight;
        // SAFETY: host_ptr_for_span guarantees `span_len` readable bytes at
        // `ptr`; `src + tight <= span_len` for every row by construction.
        unsafe {
            std::ptr::copy_nonoverlapping(
                ptr.add(src),
                bytes[dst..dst + tight].as_mut_ptr(),
                tight,
            );
        }
    }
    true
}

/// Write tight rows of a linear storage texture through one fresh-walked
/// span mapping. Stride padding bytes are left untouched —
/// consumers address rows by `row_stride`, so padding is dead space and
/// writing it is never observable. Returns `false` when the span cannot be
/// packed or the write is outside the task's recorded map spans — the caller
/// falls back to the per-row walk (which fails visibly per contract).
#[allow(
    clippy::too_many_arguments,
    reason = "the bulk path keeps the decoded texture window and row layout explicit"
)]
fn write_linear_texture_bulk<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    task_id: u32,
    gva: u64,
    row_stride: u64,
    tight: usize,
    height: u32,
    bytes: &[u8],
) -> bool {
    if height == 0 || tight == 0 || bytes.len() < (height as usize).saturating_mul(tight) {
        return false;
    }
    let Some(span_len) = (height as u64 - 1)
        .checked_mul(row_stride)
        .and_then(|v| v.checked_add(tight as u64))
    else {
        return false;
    };
    if !state.gva_write_allowed(task_id, gva, span_len) {
        return false;
    }
    // Fresh PT walk at write time — never a cached view (stale-view class).
    let Some(span_map) =
        crate::runtime::gva_view::map_fresh_span(state, host, task_id, gva, span_len)
    else {
        return false;
    };
    let ptr = span_map.ptr;
    for y in 0..height as usize {
        let src = y * tight;
        let dst = (y as u64).saturating_mul(row_stride) as usize;
        // SAFETY: map_fresh_span guarantees `span_len` writable bytes at
        // `ptr`; `dst + tight <= span_len` for every row by construction.
        unsafe {
            std::ptr::copy_nonoverlapping(bytes[src..src + tight].as_ptr(), ptr.add(dst), tight);
        }
    }
    crate::runtime::gva_view::unmap_fresh_span(host, span_map);
    true
}

fn writeback_texture<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    task_id: u32,
    tex: &StagedTexture,
) -> Result<(), ComputeStatus> {
    match &tex.writeback {
        TextureWriteback::None => Ok(()),
        TextureWriteback::Linear {
            texture_ref,
            gva,
            pixel_format,
            row_stride,
            width,
            height,
            bpp,
        } => {
            let tight = (*width as usize) * (*bpp as usize);
            let required = tight.saturating_mul(*height as usize);
            if tight > *row_stride as usize || tex.bytes.len() < required {
                crate::observe::fail(format!(
                    "compute_writeback_tex fail reason=linear_layout bind={} gva={gva:#x} dims={}x{} bpp={} row_stride={} tight={} bytes={} required={required}",
                    tex.binding,
                    width,
                    height,
                    bpp,
                    row_stride,
                    tight,
                    tex.bytes.len()
                ));
                return Err(ComputeStatus::GuestIo("compute_wb_tex_linear_layout"));
            }
            if !crate::runtime::surface_cache::store_linear_texture(
                state,
                task_id,
                *texture_ref,
                *gva,
                *pixel_format,
                *width,
                *height,
                *row_stride,
                &tex.bytes,
            ) {
                crate::observe::fail(format!(
                    "compute_writeback_tex fail reason=linear_cache_store task={task_id} ref={texture_ref} bind={} gva={gva:#x} fmt={pixel_format:#x} dims={}x{} bpp={} row_stride={} bytes={}",
                    tex.binding,
                    width,
                    height,
                    bpp,
                    row_stride,
                    tex.bytes.len()
                ));
                return Err(ComputeStatus::GuestIo("compute_wb_tex_linear_cache_store"));
            }
            crate::runtime::surface_cache::mirror_linear_color_cache(
                state,
                *texture_ref,
                *gva,
                *pixel_format,
                *width,
                *height,
                &tex.bytes,
            );
            let Some(span) = row_stride.checked_mul(*height as u64) else {
                crate::observe::fail(format!(
                    "compute_writeback_tex fail reason=linear_span_overflow task={task_id} ref={texture_ref} bind={} gva={gva:#x} dims={}x{} row_stride={row_stride}",
                    tex.binding, width, height
                ));
                return Err(ComputeStatus::GuestIo(
                    "compute_wb_tex_linear_span_overflow",
                ));
            };
            if !state.gva_write_allowed(task_id, *gva, span) {
                crate::observe::fail(format!(
                    "compute_writeback_tex cache_only reason=linear_unmapped task={task_id} ref={texture_ref} bind={} gva={gva:#x} span={span:#x} fmt={pixel_format:#x} dims={}x{} bpp={} row_stride={row_stride}",
                    tex.binding, width, height, bpp
                ));
                return Ok(());
            }
            if write_linear_guest(
                state,
                host,
                task_id,
                *gva,
                *row_stride,
                tight,
                *height,
                &tex.bytes,
                &format!("bind={}", tex.binding),
            ) {
                Ok(())
            } else {
                Err(ComputeStatus::GuestIo("compute_wb_tex_linear_guest_write"))
            }
        }
        TextureWriteback::Type11 {
            mapping_id,
            surface_offset,
            surface_bpr,
            span_end,
            width,
            height,
            bpp,
        } => {
            let tight = width.saturating_mul(*bpp);
            if !mapping_write::write_full_rect_raw_at(
                state,
                host,
                *mapping_id,
                *surface_offset,
                *surface_bpr,
                *span_end,
                *width,
                *height,
                *bpp,
                &tex.bytes,
                tight,
            ) {
                crate::observe::fail(format!(
                    "compute_writeback_tex fail reason=type11_mapping_write task={task_id} bind={} mid={} surface_offset={surface_offset:#x} surface_bpr={} span_end={span_end:#x} dims={}x{} bpp={} bytes={} tight={tight}",
                    tex.binding,
                    mapping_id,
                    surface_bpr,
                    width,
                    height,
                    bpp,
                    tex.bytes.len()
                ));
                return Err(ComputeStatus::GuestIo("compute_wb_tex_type11_write"));
            }
            Ok(())
        }
    }
}

/// Write tight-row `bytes` into a strided linear guest window through fresh
/// task page-table walks (bulk view when packable, per-row fallback). Fail
/// lines carry `ctx` for the call site. Returns `false` on any failed write.
#[allow(clippy::too_many_arguments)]
pub(crate) fn write_linear_guest<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    task_id: u32,
    gva: u64,
    row_stride: u64,
    tight: usize,
    height: u32,
    bytes: &[u8],
    ctx: &str,
) -> bool {
    if write_linear_texture_bulk(state, host, task_id, gva, row_stride, tight, height, bytes) {
        return true;
    }
    let mut row = vec![0u8; row_stride as usize];
    for y in 0..height {
        let src_off = (y as usize) * tight;
        row[..tight].copy_from_slice(&bytes[src_off..src_off + tight]);
        // Pad rest of row with zeros already present.
        let Some(row_offset) = (y as u64).checked_mul(row_stride) else {
            crate::observe::fail(format!(
                "compute_writeback_tex fail reason=linear_row_offset_overflow {ctx} gva={gva:#x} y={y} row_stride={row_stride}"
            ));
            return false;
        };
        let Some(row_gva) = gva.checked_add(row_offset) else {
            crate::observe::fail(format!(
                "compute_writeback_tex fail reason=linear_gva_overflow {ctx} gva={gva:#x} y={y} row_offset={row_offset:#x}"
            ));
            return false;
        };
        if let Err(e) = gva_mem::write_task_gva_product(
            state,
            host,
            task_id,
            row_gva,
            &row[..row_stride as usize],
        ) {
            crate::observe::fail(format!(
                "compute_writeback_tex fail reason=linear_gva_write task={task_id} {ctx} gva={row_gva:#x} y={y} row_stride={row_stride} height={height} err={e:?}"
            ));
            return false;
        }
    }
    true
}

fn writeback_buffer<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    task_id: u32,
    pipe_ref: Option<u32>,
    context: &str,
    staged: &StagedBuffer,
) -> Result<(), ComputeStatus> {
    if let Err(e) = gva_mem::write_task_gva_product(state, host, task_id, staged.gva, &staged.bytes)
    {
        crate::observe::fail(format!(
            "compute_writeback_buf fail reason=task_gva_write task={task_id} pipe={} context={context} idx={} ref={} gva={:#x} len={} off={:#x} err={e:?}",
            pipe_ref
                .map(|value| value.to_string())
                .unwrap_or_else(|| "unknown".into()),
            staged.bind.index,
            staged.bind.buffer_ref,
            staged.gva,
            staged.bytes.len(),
            staged.bind.offset
        ));
        return Err(ComputeStatus::GuestIo("compute_wb_buf_task_gva_write"));
    }
    Ok(())
}

fn u32_dim(v: u64) -> Result<u32, ComputeStatus> {
    if v == 0 || v > u32::MAX as u64 {
        Err(ComputeStatus::BadGrid("compute_grid_dim_range"))
    } else {
        Ok(v as u32)
    }
}

/// Archive `REIMS_VGPU_COMPUTE_PLAN_DEFERRED_GRID_Y_SENTINEL` / `try_recover_sentinel_grid`.
///
/// Live Core Image wallpaper shape (journal 2026-07-06):
///   grid = [ceil(ow/tg), UINT64_MAX, 1], tg = [32, 0, 1]
/// Recover tg.y = tg.x (square tile) and both grid axes from the largest
/// write-capable bound texture. Without this every wallpaper VTMTS/CI dispatch
/// hits [`ComputeStatus::BadGrid`] and the desktop stays black.
const DEFERRED_GRID_Y_SENTINEL: u64 = u64::MAX;
const GRID_DIM_MAX: u64 = 0x1_0000;
const THREADGROUP_DIM_MAX: u64 = 1024;

fn ceil_div_u64(n: u64, d: u64) -> u64 {
    if d == 0 {
        return 0;
    }
    n.div_ceil(d)
}

/// Largest bound texture (type-11 or type-2/3) by pixel area — archive picks the
/// full-screen write target over small inputs.
fn largest_bound_texture_dims<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &M,
    task_id: u32,
    acc: &ComputeAccum,
) -> Option<(u32, u32)> {
    let mut best: Option<(u32, u32, u64)> = None;
    for t in &acc.textures {
        if t.texture_ref == 0 {
            continue;
        }
        // Prefer type-11 mapping geom.
        if let Some(mid) = objects::resolve_type11_ref(state, host, task_id, t.texture_ref) {
            let _ = mapper::ensure_resolved_for_scanout(state, host, mid);
            if let Some(m) = state.mappings.get(&mid) {
                if m.has_geom && m.width > 0 && m.height > 0 {
                    let area = (m.width as u64).saturating_mul(m.height as u64);
                    if best.map(|(_, _, a)| area > a).unwrap_or(true) {
                        best = Some((m.width, m.height, area));
                    }
                    continue;
                }
            }
        }
        // type-2/3 linear.
        let Some(entry) = objects::lookup_list_entry(state, host, task_id, t.texture_ref) else {
            continue;
        };
        if entry.object_type != OBJECT_TYPE_TEXTURE
            && entry.object_type != OBJECT_TYPE_TEXTURE_VARIANT
        {
            continue;
        }
        let Some(desc) = objects::read_descriptor(state, host, task_id, &entry) else {
            continue;
        };
        let Ok(tex) = decode_texture_descriptor(&desc) else {
            continue;
        };
        if !tex.has_width || !tex.has_height || tex.width == 0 || tex.height == 0 {
            continue;
        }
        let area = (tex.width as u64).saturating_mul(tex.height as u64);
        if best.map(|(_, _, a)| area > a).unwrap_or(true) {
            best = Some((tex.width, tex.height, area));
        }
    }
    best.map(|(w, h, _)| (w, h))
}

/// Recover CI sentinel grid/threadgroup (archive `try_recover_sentinel_grid`).
/// Returns `Some((gx,gy,gz,tx,ty,tz))` when the wire matches the sentinel shape
/// and a bound texture supplies coverage; otherwise `None`.
fn try_recover_sentinel_grid<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &M,
    task_id: u32,
    acc: &ComputeAccum,
    cmd: &ComputeCommand,
) -> Option<(u32, u32, u32, u32, u32, u32)> {
    if cmd.kind != Kind::DispatchThreadgroups {
        return None;
    }
    let g0 = cmd.grid.x;
    let g1 = cmd.grid.y;
    let g2 = cmd.grid.z;
    let t0 = cmd.threads_per_threadgroup.x;
    let t1 = cmd.threads_per_threadgroup.y;
    let t2 = cmd.threads_per_threadgroup.z;
    if !((1..=GRID_DIM_MAX).contains(&g0)
        && g1 == DEFERRED_GRID_Y_SENTINEL
        && g2 == 1
        && (1..=THREADGROUP_DIM_MAX).contains(&t0)
        && t1 == 0
        && t2 == 1)
    {
        return None;
    }
    let (tw, th) = largest_bound_texture_dims(state, host, task_id, acc)?;
    let ty = t0;
    let mut gx = ceil_div_u64(tw as u64, t0).max(1);
    let mut gy = ceil_div_u64(th as u64, ty).max(1);
    if gx > GRID_DIM_MAX {
        gx = GRID_DIM_MAX;
    }
    if gy > GRID_DIM_MAX {
        gy = GRID_DIM_MAX;
    }
    crate::observe::line(format!(
        "compute_sentinel_recover grid=[{gx},{gy},1] tg=[{t0},{ty},1] tex={tw}x{th} wire_g0={g0}"
    ));
    Some((gx as u32, gy as u32, 1, t0 as u32, ty as u32, 1))
}

/// Measure-only census for a compute dispatch (always-on fail log on Linux).
///
/// Live x86 wallpaper class (serial-213122): type-3 multi-bind samples stay
/// mapped zeros while MapMemory2/ReplacePhysical cycle; Core Image fills are
/// `SEGMENT_TYPE_COMPUTE` dispatches. On non-Apple hosts
/// [`execute_dispatch`] returns [`ComputeStatus::NoMetal`] and previously
/// left **no** fail-log line — CI wallpaper drops were invisible next to
/// m2v_empty_layer. Proxy: `compute_dispatch st=NoMetal …`.
fn log_compute_dispatch_census<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &M,
    task_id: u32,
    acc: &ComputeAccum,
    cmd: &ComputeCommand,
    st: ComputeStatus,
    nested: bool,
) {
    let kind = match cmd.kind {
        Kind::DispatchThreadgroups => "tg",
        Kind::DispatchThreads => "threads",
        Kind::DispatchThreadgroupsIndirect => "tg_ind",
        Kind::DispatchThreadsIndirect => "thr_ind",
        _ => "other",
    };
    let mut tex_parts: Vec<String> = Vec::new();
    for t in acc.textures.iter().take(8) {
        if t.texture_ref == 0 {
            continue;
        }
        // Compact geom for wallpaper vs tile discrimination (no content gate).
        let geom = if let Some(mid) =
            objects::resolve_type11_ref(state, host, task_id, t.texture_ref)
        {
            state
                .mappings
                .get(&mid)
                .filter(|m| m.has_geom)
                .map(|m| format!("{}x{}", m.width, m.height))
                .unwrap_or_else(|| "t11".into())
        } else if let Some(entry) = objects::lookup_list_entry(state, host, task_id, t.texture_ref)
        {
            if entry.object_type == OBJECT_TYPE_TEXTURE
                || entry.object_type == OBJECT_TYPE_TEXTURE_VARIANT
            {
                if let Some(desc) = objects::read_descriptor(state, host, task_id, &entry) {
                    if let Ok(tex) = decode_texture_descriptor(&desc) {
                        format!("{}x{}", tex.width, tex.height)
                    } else {
                        format!("ot{}", entry.object_type)
                    }
                } else {
                    format!("ot{}", entry.object_type)
                }
            } else {
                format!("ot{}", entry.object_type)
            }
        } else {
            "?".into()
        };
        tex_parts.push(format!("i{}:r{}:{}", t.index, t.texture_ref, geom));
    }
    let largest = largest_bound_texture_dims(state, host, task_id, acc)
        .map(|(w, h)| format!("{w}x{h}"))
        .unwrap_or_else(|| "none".into());
    let nest = if nested { " nested=1" } else { "" };
    // Always-on either way (a wallpaper CI miss must not hide in the env-gated
    // draw.log only), but a successful dispatch (`st=Ok`) is census — route it
    // to the `off()` sink so it stays in the log but leaves the curated
    // real-error view (`grep -v '^OFF '`) clean; a non-Ok dispatch is a genuine
    // failed guest compute command and stays `fail()`-visible with its reason.
    let line = format!(
        "compute_dispatch st={st:?} pipe={} kind={kind} ntex={} nbuf={} largest={largest} grid=[{},{},{}] tg=[{},{},{}] tex=[{}]{nest}",
        acc.pipeline_ref,
        acc.textures.len(),
        acc.buffers.len(),
        cmd.grid.x,
        cmd.grid.y,
        cmd.grid.z,
        cmd.threads_per_threadgroup.x,
        cmd.threads_per_threadgroup.y,
        cmd.threads_per_threadgroup.z,
        tex_parts.join(","),
    );
    if matches!(st, ComputeStatus::Ok) {
        crate::observe::off(line);
    } else {
        crate::observe::fail(line);
    }
}

/// Execute a direct or indirect dispatch against the current compute accum state.
pub fn execute_dispatch<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    task_id: u32,
    acc: &ComputeAccum,
    cmd: &ComputeCommand,
) -> ComputeStatus {
    #[cfg(all(feature = "backend-metal", target_os = "macos"))]
    {
        let st = execute_dispatch_metal(state, host, task_id, acc, cmd, None);
        log_compute_dispatch_census(state, host, task_id, acc, cmd, st, false);
        st
    }
    #[cfg(feature = "backend-vulkan")]
    {
        let st = execute_dispatch_linux(state, host, task_id, acc, cmd);
        log_compute_dispatch_census(state, host, task_id, acc, cmd, st, false);
        st
    }
    #[cfg(all(not(feature = "backend-vulkan"), feature = "backend-vulkan"))]
    {
        // Metal stubs / no vulkan feature: fail-visible census only.
        log_compute_dispatch_census(
            state,
            host,
            task_id,
            acc,
            cmd,
            ComputeStatus::NoMetal("compute_dispatch_no_backend"),
            false,
        );
        ComputeStatus::NoMetal("compute_dispatch_no_backend")
    }
}

/// Nested dispatch onto an open multi-record control-flow session encoder.
pub(crate) fn execute_dispatch_nested<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    task_id: u32,
    acc: &ComputeAccum,
    cmd: &ComputeCommand,
    session: &mut crate::runtime::compute_session::ComputeSession,
) -> ComputeStatus {
    #[cfg(all(feature = "backend-metal", target_os = "macos"))]
    {
        let st = execute_dispatch_metal(state, host, task_id, acc, cmd, Some(session));
        log_compute_dispatch_census(state, host, task_id, acc, cmd, st, true);
        st
    }
    #[cfg(feature = "backend-vulkan")]
    {
        // Nested/control-flow SPI not yet on Linux compute — fail-visible.
        let _ = session;
        log_compute_dispatch_census(
            state,
            host,
            task_id,
            acc,
            cmd,
            ComputeStatus::NoMetal("compute_nested_no_vulkan_path"),
            true,
        );
        ComputeStatus::NoMetal("compute_nested_no_vulkan_path")
    }
}

/// One nested dispatch's deferred writeback (GPU → host staging → GVA after session commit).
#[cfg(all(feature = "backend-metal", target_os = "macos"))]
pub(crate) struct NestedDispatchJob {
    staged_bufs: Vec<StagedBuffer>,
    /// Storage textures only (sampled need no writeback).
    storage_tex: Vec<StagedTexture>,
    mtl_buffers: Vec<metal::Buffer>,
    mtl_storage: Vec<metal::Texture>,
}

/// Build a deferred writeback job for ICB-filled kernel buffers (no storage textures).
#[cfg(all(feature = "backend-metal", target_os = "macos"))]
pub(crate) fn nested_job_from_icb_buffers(
    staged_bufs: Vec<StagedBuffer>,
    mtl_buffers: Vec<metal::Buffer>,
) -> NestedDispatchJob {
    nested_job_from_icb_resources(staged_bufs, mtl_buffers, Vec::new(), Vec::new())
}

/// Deferred writeback for parent-encoder ICB inheritance (buffers + storage textures).
#[cfg(all(feature = "backend-metal", target_os = "macos"))]
pub(crate) fn nested_job_from_icb_resources(
    staged_bufs: Vec<StagedBuffer>,
    mtl_buffers: Vec<metal::Buffer>,
    storage_tex: Vec<StagedTexture>,
    mtl_storage: Vec<metal::Texture>,
) -> NestedDispatchJob {
    NestedDispatchJob {
        staged_bufs,
        storage_tex,
        mtl_buffers,
        mtl_storage,
    }
}

#[cfg(all(feature = "backend-metal", target_os = "macos"))]
pub(crate) fn flush_nested_jobs<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    task_id: u32,
    jobs: &mut [NestedDispatchJob],
) -> ComputeStatus {
    use crate::backend::metal::abi::{ReimsVgpuBuffer, ReimsVgpuStorageImage};
    use crate::backend::metal::compute::compute_writeback_from_mtl;

    let mut err_buf = [0i8; 256];
    for job in jobs.iter_mut() {
        let mut reims_vgpu_bufs: Vec<ReimsVgpuBuffer> = job
            .staged_bufs
            .iter_mut()
            .map(|s| ReimsVgpuBuffer {
                binding: s.bind.index,
                data: s.bytes.as_mut_ptr(),
                len: s.bytes.len(),
                attribute_stride: s.bind.attribute_stride,
                has_attribute_stride: if s.bind.has_attribute_stride { 1 } else { 0 },
                reserved0: 0,
                backing_data: std::ptr::null_mut(),
                backing_len: 0,
                backing_offset: 0,
            })
            .collect();
        let mut storage: Vec<ReimsVgpuStorageImage> = job
            .storage_tex
            .iter_mut()
            .map(|t| ReimsVgpuStorageImage {
                binding: t.binding,
                format: t
                    .storage_selector
                    .expect("storage texture staged with a storage selector"),
                width: t.width,
                height: t.height,
                data: t.bytes.as_mut_ptr(),
                len: t.bytes.len(),
            })
            .collect();
        let st = compute_writeback_from_mtl(
            &mut reims_vgpu_bufs,
            &job.mtl_buffers,
            &mut storage,
            &job.mtl_storage,
            (err_buf.as_mut_ptr(), err_buf.len()),
        );
        if !st.is_ok() {
            return ComputeStatus::MetalFailed("compute_nested_writeback_metal");
        }
        for s in &job.staged_bufs {
            if let Err(e) = writeback_buffer(state, host, task_id, None, "nested_flush", s) {
                return e;
            }
        }
        for t in &job.storage_tex {
            if let Err(e) = writeback_texture(state, host, task_id, t) {
                return e;
            }
        }
    }
    ComputeStatus::Ok
}

type DispatchDims = (u32, u32, u32, u32, u32, u32, bool);

/// Resolve grid/threadgroup dims for direct or indirect dispatches.
fn resolve_dispatch_dims<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &M,
    task_id: u32,
    acc: &ComputeAccum,
    cmd: &ComputeCommand,
) -> Result<DispatchDims, ComputeStatus> {
    match cmd.kind {
        Kind::DispatchThreadgroups => {
            if let Some(dims) = try_recover_sentinel_grid(state, host, task_id, acc, cmd) {
                return Ok((dims.0, dims.1, dims.2, dims.3, dims.4, dims.5, false));
            }
            Ok((
                u32_dim(cmd.grid.x)?,
                u32_dim(cmd.grid.y)?,
                u32_dim(cmd.grid.z)?,
                u32_dim(cmd.threads_per_threadgroup.x)?,
                u32_dim(cmd.threads_per_threadgroup.y)?,
                u32_dim(cmd.threads_per_threadgroup.z)?,
                false,
            ))
        }
        Kind::DispatchThreads => Ok((
            u32_dim(cmd.grid.x)?,
            u32_dim(cmd.grid.y)?,
            u32_dim(cmd.grid.z)?,
            u32_dim(cmd.threads_per_threadgroup.x)?,
            u32_dim(cmd.threads_per_threadgroup.y)?,
            u32_dim(cmd.threads_per_threadgroup.z)?,
            true,
        )),
        Kind::DispatchThreadgroupsIndirect => {
            let raw = read_buffer_window(
                state,
                host,
                task_id,
                cmd.indirect_buffer_ref,
                cmd.indirect_buffer_offset,
                INDIRECT_THREADGROUPS_ARGS_LEN,
            )?;
            let gx = ld32(&raw[0..]);
            let gy = ld32(&raw[4..]);
            let gz = ld32(&raw[8..]);
            Ok((
                u32_dim(gx as u64)?,
                u32_dim(gy as u64)?,
                u32_dim(gz as u64)?,
                u32_dim(cmd.threads_per_threadgroup.x)?,
                u32_dim(cmd.threads_per_threadgroup.y)?,
                u32_dim(cmd.threads_per_threadgroup.z)?,
                false,
            ))
        }
        Kind::DispatchThreadsIndirect => {
            let raw = read_buffer_window(
                state,
                host,
                task_id,
                cmd.indirect_buffer_ref,
                cmd.indirect_buffer_offset,
                INDIRECT_THREADS_ARGS_LEN,
            )?;
            // MTLDispatchThreadsIndirectArguments: threadsPerGrid[3], threadsPerThreadgroup[3].
            Ok((
                u32_dim(ld32(&raw[0..]) as u64)?,
                u32_dim(ld32(&raw[4..]) as u64)?,
                u32_dim(ld32(&raw[8..]) as u64)?,
                u32_dim(ld32(&raw[12..]) as u64)?,
                u32_dim(ld32(&raw[16..]) as u64)?,
                u32_dim(ld32(&raw[20..]) as u64)?,
                true,
            ))
        }
        _ => Err(ComputeStatus::Unsupported("resolve_dims_unknown_kind")),
    }
}

/// Linux product compute path (doorbell / BQL).
///
/// Stages buffers/textures with device `page_shift`, translates the kernel AIR
/// via [`crate::runtime::m2v_cache::translate_cached_kernel_reflected`], dispatches on the
/// process-global [`crate::backend::vulkan::engine`] (shared GRAPHICS|COMPUTE
/// device), then writebacks GVA / type-11.
///
/// Nested/ICB/stage-in stay Unsupported (engine surface is storage buffers +
/// storage images only).
#[cfg(feature = "backend-vulkan")]
fn execute_dispatch_linux<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    task_id: u32,
    acc: &ComputeAccum,
    cmd: &ComputeCommand,
) -> ComputeStatus {
    use crate::backend::vulkan::engine::{
        self as vk_engine, ComputeBufferResource, ComputeRequest, ComputeSampledImageResource,
        ComputeStorageImageResource, DrawError,
    };

    const TEXTURE_BIND_BASE: u32 = 32;

    let total_started = std::time::Instant::now();
    let stage_started = std::time::Instant::now();

    if acc.pipeline_ref == 0 {
        return ComputeStatus::MissingPipeline("compute_vk_pipeline_ref_zero");
    }
    let Some(pipeline) = load_compute_pipeline(state, host, task_id, acc.pipeline_ref) else {
        return ComputeStatus::MissingPipeline("compute_vk_pipeline_load");
    };
    if pipeline.stage_input.is_some() || acc.imageblock.is_some() || acc.stage_in_region.is_some() {
        crate::observe::fail(format!(
            "compute_linux unsupported pipe={} stage_in={} imageblock={} (need SPI parity)",
            acc.pipeline_ref,
            pipeline.stage_input.is_some() as u8,
            acc.imageblock.is_some() as u8
        ));
        return ComputeStatus::Unsupported("linux_stage_in_imageblock");
    }
    // Dims first (cheap; proves sentinel recovery without m2v/vk).
    let (grid_x, grid_y, grid_z, tg_x, tg_y, tg_z, dispatch_threads) =
        match resolve_dispatch_dims(state, host, task_id, acc, cmd) {
            Ok(v) => v,
            Err(e) => {
                crate::observe::line(format!(
                "compute_resolve_dims fail {e:?} kind={:?} grid=[{},{},{}] tg=[{},{},{}] ntex={}",
                cmd.kind,
                cmd.grid.x,
                cmd.grid.y,
                cmd.grid.z,
                cmd.threads_per_threadgroup.x,
                cmd.threads_per_threadgroup.y,
                cmd.threads_per_threadgroup.z,
                acc.textures.len()
            ));
                return e;
            }
        };
    if tg_x == 0 || tg_y == 0 || tg_z == 0 || grid_x == 0 || grid_y == 0 || grid_z == 0 {
        return ComputeStatus::BadGrid("compute_vk_zero_dims");
    }

    // Stage buffers first (page_shift-correct). Texture staging follows kernel
    // translation because sampled-vs-storage access is a SPIR-V interface fact.
    // The translation cache keeps warm dispatches cheap; no Vulkan work occurs
    // until every declared resource has staged successfully.
    let mut staged_bufs: Vec<StagedBuffer> = Vec::new();
    for b in &acc.buffers {
        match stage_buffer(state, host, task_id, b) {
            Ok(s) => staged_bufs.push(s),
            Err(e) => {
                // `st={e:?}` alone was not greppable: the Debug spelling was
                // the only handle on which of stage_buffer's eight checks
                // refused. `reason=` names it.
                crate::observe::fail(format!(
                    "compute_linux stage_buf fail reason={} pipe={} idx={} ref={} off={:#x} class={}",
                    e.reason(),
                    acc.pipeline_ref,
                    b.index,
                    b.buffer_ref,
                    b.offset,
                    e.class()
                ));
                return e;
            }
        }
    }
    // MTLB → AIR → SPIR-V (LocalSize = threadgroup dims).
    let translate_started = std::time::Instant::now();
    let Some(mtlb) = load_mtlb(state, host, task_id, pipeline.kernel_func_ref) else {
        return ComputeStatus::MissingMtlb("compute_vk_mtlb_load");
    };
    // The function blob is an MTLB container; llvm-dis needs the wrapped AIR
    // bitcode member (same extract the render path does — passing the raw
    // container was the live `llvm-dis: file doesn't start with bitcode
    // header` MetalFailed class).
    let air = match crate::runtime::mtlb::extract_air(&mtlb) {
        Ok(a) => a,
        Err(e) => {
            crate::observe::Emit::decline("compute_linux_air_extract", &e)
                .field("pipe", acc.pipeline_ref)
                .fail_once(acc.pipeline_ref as u64);
            return ComputeStatus::MetalFailed("compute_vk_air_extract");
        }
    };
    let kernel_shader = match crate::runtime::m2v_cache::translate_cached_kernel_reflected(
        air,
        [tg_x, tg_y, tg_z],
        acc.pipeline_ref,
    ) {
        Ok(b) => b,
        Err(e) => {
            // Handoff: on a translator failure, dump the exact kernel inputs
            // (raw MTLB container + extracted AIR) once per pipe so the
            // metal2vulkan agent can reproduce off-VM. Apple-owned IR — path is
            // gitignored (REIMS_VGPU_M2V_FAIL_DIR or /tmp). One file per pipe; sidecar
            // records the tg dims + error.
            dump_kernel_handoff(
                acc.pipeline_ref,
                "translate",
                &mtlb,
                air,
                None,
                &format!("threadgroup=[{tg_x},{tg_y},{tg_z}]\nerror={e}"),
            );
            crate::observe::Emit::decline("compute_linux_m2v", &e)
                .field("pipe", acc.pipeline_ref)
                .fail_once(acc.pipeline_ref as u64);
            return ComputeStatus::MetalFailed("compute_vk_translate");
        }
    };
    let mut spirv = match spirv_words_le(&kernel_shader.spirv) {
        Ok(w) => w,
        Err(e) => {
            crate::observe::Emit::decline("compute_linux_spirv_parse", &e)
                .field("pipe", acc.pipeline_ref)
                .fail_once(acc.pipeline_ref as u64);
            return ComputeStatus::MetalFailed("compute_vk_spirv_parse");
        }
    };
    let translate_us = u64::try_from(translate_started.elapsed().as_micros()).unwrap_or(u64::MAX);

    let mut buffer_accesses = Vec::with_capacity(staged_bufs.len());
    let mut buffer_readonly_count = 0usize;
    let mut buffer_writable_count = 0usize;
    let mut buffer_unused_count = 0usize;
    for s in &staged_bufs {
        use crate::runtime::spirv_bind::BufferAccess;
        match crate::runtime::spirv_bind::buffer_access(&spirv, s.bind.index) {
            Some(BufferAccess::ReadOnly) => {
                buffer_readonly_count += 1;
                buffer_accesses.push((s.bind.index, false));
            }
            Some(BufferAccess::Writable) => {
                buffer_writable_count += 1;
                buffer_accesses.push((s.bind.index, true));
            }
            Some(BufferAccess::PointerEscape) => {
                crate::observe::fail(format!(
                    "compute_linux buffer_access fail reason=spirv_pointer_escape pipe={} idx={} ref={}",
                    acc.pipeline_ref, s.bind.index, s.bind.buffer_ref
                ));
                return ComputeStatus::Unsupported("buffer_spirv_pointer_escape");
            }
            Some(BufferAccess::AmbiguousBinding) => {
                crate::observe::fail(format!(
                    "compute_linux buffer_access fail reason=spirv_ambiguous_binding pipe={} idx={} ref={}",
                    acc.pipeline_ref, s.bind.index, s.bind.buffer_ref
                ));
                dump_kernel_handoff(
                    acc.pipeline_ref,
                    "buffer_ambiguous_binding",
                    &mtlb,
                    air,
                    Some(&spirv),
                    &format!(
                        "buffer_index={} buffer_ref={} grid=[{grid_x},{grid_y},{grid_z}] tg=[{tg_x},{tg_y},{tg_z}]",
                        s.bind.index, s.bind.buffer_ref
                    ),
                );
                return ComputeStatus::Unsupported("buffer_spirv_ambiguous_binding");
            }
            None => {
                buffer_unused_count += 1;
                crate::observe::line(format!(
                    "compute_linux buffer_unused pipe={} idx={} ref={}",
                    acc.pipeline_ref, s.bind.index, s.bind.buffer_ref
                ));
            }
        }
    }

    let mut staged_tex: Vec<StagedTexture> = Vec::new();
    let mut storage_writeonly_count = 0usize;
    for t in &acc.textures {
        use crate::runtime::spirv_bind::{ImageAccess, StorageImageAccess};
        let binding = TEXTURE_BIND_BASE + t.index;
        // Sampled-vs-storage comes solely from the translator's reflection — the
        // declared Metal access qualifier, exact at translate time. The always-on
        // `census_reflection_wellformed` guard proves the reflection is internally
        // consistent per translate.
        let image_access = crate::runtime::spirv_bind::image_access_from_reflection(
            &kernel_shader.reflection,
            binding,
        );
        let is_storage = match image_access {
            Some(ImageAccess::Sampled) => false,
            Some(ImageAccess::Storage) => true,
            None => {
                // Metal permits unused bound resources. If reflection lists no
                // texture shape at this binding, the shader does not sample/write
                // it — do not stage or invent access/writeback semantics for it.
                crate::observe::line(format!(
                    "compute_linux texture_unused pipe={} i={} ref={} bind={}",
                    acc.pipeline_ref, t.index, t.texture_ref, binding
                ));
                continue;
            }
        };
        let storage_access = if is_storage {
            match crate::runtime::spirv_bind::storage_image_access(&spirv, binding) {
                Some(StorageImageAccess::WriteOnly) => Some("write_only"),
                Some(StorageImageAccess::ReadOnly) => Some("read_only"),
                Some(StorageImageAccess::ReadWrite) => Some("read_write"),
                Some(StorageImageAccess::Unknown) => Some("unknown"),
                Some(StorageImageAccess::AmbiguousBinding) => {
                    crate::observe::fail(format!(
                        "compute_linux texture_access fail reason=spirv_storage_ambiguous_binding pipe={} i={} ref={} bind={binding}",
                        acc.pipeline_ref, t.index, t.texture_ref
                    ));
                    return ComputeStatus::Unsupported("texture_spirv_storage_ambiguous_binding");
                }
                None => {
                    crate::observe::fail(format!(
                        "compute_linux texture_access fail reason=spirv_storage_access_missing pipe={} i={} ref={} bind={binding}",
                        acc.pipeline_ref, t.index, t.texture_ref
                    ));
                    return ComputeStatus::Unsupported("texture_spirv_storage_access_missing");
                }
            }
        } else {
            None
        };
        match stage_texture_raw(state, host, task_id, t.texture_ref, binding, is_storage) {
            Ok(s) => {
                if let Some(storage_access) = storage_access {
                    if storage_access == "write_only" {
                        storage_writeonly_count += 1;
                    }
                    let bytes = (s.width as u64)
                        .saturating_mul(s.height as u64)
                        .saturating_mul(
                            pixel_format::bytes_per_pixel(s.pixel_format).unwrap_or(0) as u64
                        );
                    log_storage_image_access(acc.pipeline_ref, binding, storage_access, bytes);
                }
                staged_tex.push(s);
            }
            Err(e) => {
                let ot = objects::lookup_list_entry(state, host, task_id, t.texture_ref)
                    .map(|en| en.object_type)
                    .unwrap_or(0);
                crate::observe::fail(format!(
                    "compute_linux stage_tex fail reason={} pipe={} i={} ref={} ot={} bind={} access={} class={}",
                    e.reason(),
                    acc.pipeline_ref,
                    t.index,
                    t.texture_ref,
                    ot,
                    binding,
                    if is_storage { "storage" } else { "sampled" },
                    e.class()
                ));
                // Hand off the exact kernel so the m2v/format-table agent sees
                // which storage/sampled image format the shader declares. Every
                // status now names its check, so the handoff directory is keyed
                // by that slug instead of by the flat `stage_tex_status` every
                // non-`Unsupported` refusal used to share.
                dump_kernel_handoff(
                    acc.pipeline_ref,
                    e.reason(),
                    &mtlb,
                    air,
                    Some(&spirv),
                    &format!(
                        "tex_index={} tex_ref={} ot={ot} binding={binding} access={} status={e:?}",
                        t.index,
                        t.texture_ref,
                        if is_storage { "storage" } else { "sampled" }
                    ),
                );
                return e;
            }
        }
    }

    let mut max_nz = 0usize;
    let mut max_wh = (0u32, 0u32);
    let mut sampled_count = 0usize;
    let mut storage_count = 0usize;
    for t in &staged_tex {
        if t.is_storage {
            storage_count += 1;
        } else {
            sampled_count += 1;
        }
        let (nz, max_rgb, _) = crate::observe::rgba_rgb_stats(&t.bytes);
        if nz > max_nz {
            max_nz = nz;
            max_wh = (t.width, t.height);
        }
        if t.width >= 1280 && t.height >= 720 {
            crate::observe::off(format!(
                "compute_linux stage_tex pipe={} bind={} access={} {}x{} rgb_nz={} max_rgb={}",
                acc.pipeline_ref,
                t.binding,
                if t.is_storage { "storage" } else { "sampled" },
                t.width,
                t.height,
                nz,
                max_rgb
            ));
        }
    }
    let (residency_eligible, residency_hits, residency_eligible_bytes, residency_hit_bytes) =
        storage_residency_opportunity(state, &staged_tex);
    log_storage_residency_opportunity(
        acc.pipeline_ref,
        residency_eligible,
        residency_hits,
        residency_eligible_bytes,
        residency_hit_bytes,
    );
    let stage_us = u64::try_from(stage_started.elapsed().as_micros()).unwrap_or(u64::MAX);
    crate::observe::off(format!(
        "compute_linux stage_ok pipe={} nbuf={} bro={} brw={} bunused={} ntex={} sampled={} storage={} swo={} grid=[{grid_x},{grid_y},{grid_z}] tg=[{tg_x},{tg_y},{tg_z}] max_tex={}x{} max_nz={max_nz} stage_us={} encode=engine",
        acc.pipeline_ref,
        staged_bufs.len(),
        buffer_readonly_count,
        buffer_writable_count,
        buffer_unused_count,
        staged_tex.len(),
        sampled_count,
        storage_count,
        storage_writeonly_count,
        max_wh.0,
        max_wh.1,
        stage_us,
    ));
    let prep_started = std::time::Instant::now();

    // Workgroup counts: DispatchThreadgroups already is groups; DispatchThreads
    // is total threads → ceil-div by LocalSize.
    let (wg_x, wg_y, wg_z) = if dispatch_threads {
        (
            grid_x.div_ceil(tg_x).max(1),
            grid_y.div_ceil(tg_y).max(1),
            grid_z.div_ceil(tg_z).max(1),
        )
    } else {
        (grid_x, grid_y, grid_z)
    };

    let mut storage_buffers = Vec::with_capacity(buffer_accesses.len());
    for s in &mut staged_bufs {
        let Some((_, writable)) = buffer_accesses
            .iter()
            .find(|(binding, _)| *binding == s.bind.index)
        else {
            continue;
        };
        storage_buffers.push(ComputeBufferResource {
            binding: s.bind.index,
            bytes: std::mem::take(&mut s.bytes),
            writable: *writable,
        });
    }
    let mut sampled_images = Vec::with_capacity(sampled_count);
    let mut storage_images = Vec::with_capacity(storage_count);
    let mut storage_formats = Vec::with_capacity(storage_count);
    // Device support for format-less storage writes decides whether a guest
    // BGRA8Unorm storage surface can composite into a B8G8R8A8_UNORM view (no
    // R/B swap) or must degrade to the swapped Rgba8Unorm view.
    let write_without_format = vk_engine::supports_storage_image_write_without_format();
    for t in staged_tex.iter().filter(|texture| texture.is_storage) {
        let Some(selector) = t.storage_selector else {
            crate::observe::fail(format!(
                "compute_linux unsupported storage_format reason=no_storage_selector pipe={} bind={} fmt={:#x}",
                acc.pipeline_ref, t.binding, t.pixel_format
            ));
            return ComputeStatus::Unsupported("storage_no_selector_specialize");
        };
        let Some(guest_fmt) = simg_u32_to_engine_storage(selector) else {
            crate::observe::fail(format!(
                "compute_linux unsupported storage_format reason=selector_unknown pipe={} bind={} simg={selector} fmt={:#x}",
                acc.pipeline_ref, t.binding, t.pixel_format
            ));
            return ComputeStatus::Unsupported("storage_selector_unknown_specialize");
        };
        let Some(shader_decl) = crate::runtime::spirv_bind::image_format(&spirv, t.binding) else {
            crate::observe::fail(format!(
                "compute_linux storage_format fail reason=spirv_format_missing pipe={} bind={} guest={guest_fmt:?} simg={}",
                acc.pipeline_ref, t.binding, selector
            ));
            return ComputeStatus::Unsupported("storage_spirv_format_missing");
        };
        let specialized = match specialized_storage_image_format(
            guest_fmt,
            shader_decl,
            write_without_format,
        ) {
            Ok(format) => format,
            Err(reason) => {
                crate::observe::fail(format!(
                        "compute_linux storage_format fail reason={reason} pipe={} bind={} spirv={shader_decl:?} guest={guest_fmt:?} simg={} guest_bpp={} shader_bpp={}",
                        acc.pipeline_ref,
                        t.binding,
                        selector,
                        guest_fmt.bytes_per_texel(),
                        spirv_image_format_to_engine_storage(shader_decl)
                            .map(|format| format.bytes_per_texel())
                            .unwrap_or(0)
                    ));
                return ComputeStatus::Unsupported("storage_format_specialize_mismatch");
            }
        };
        storage_formats.push((t.binding, guest_fmt, shader_decl, specialized));
    }
    let specialization_requests: Vec<_> = storage_formats
        .iter()
        .map(|(binding, _, _, specialized)| (*binding, *specialized))
        .collect();
    if let Err(error) =
        crate::runtime::spirv_bind::specialize_image_formats(&mut spirv, &specialization_requests)
    {
        let error: crate::runtime::spirv_bind::ImageFormatSpecializeError = error;
        crate::observe::Emit::decline("compute_linux_storage_format", &error)
            .field("pipe", acc.pipeline_ref)
            .fail();
        return ComputeStatus::Unsupported("storage_format_specialize_error");
    }
    // A guest BGRA8Unorm storage surface retargets to an `Unknown`-format
    // storage image (viewed B8G8R8A8_UNORM) so the composite writes land in the
    // guest's channel order — that write is only legal if the module declares
    // `StorageImageWriteWithoutFormat`. Inject it once when any binding took the
    // Unknown path (idempotent; the translator declares only Shader/Float16/…).
    if storage_formats.iter().any(|(_, _, _, specialized)| {
        matches!(
            specialized,
            crate::runtime::spirv_bind::ImageFormat::Unknown
        )
    }) {
        crate::runtime::spirv_bind::ensure_storage_write_without_format_capability(&mut spirv);
    }
    // GPU-direct writeback support is a per-device constant; `None` disables
    // planning for this dispatch (extension absent or engine init failed).
    let direct_align = vk_engine::compute_host_writeback_alignment();
    // Compute-side analog of the render resident gates: a deferred storage
    // writeback leaves guest-visible bytes GPU-resident-only until a flush
    // choke point lands them, so it requires the device's
    // `deferred_gpu_only_content` capability (off on portability-subset /
    // MoltenVK, where guest pages stay authoritative and the writeback runs
    // synchronously in this call).
    let deferred_content_allowed = vk_engine::deferred_gpu_only_content_allowed();
    for t in &mut staged_tex {
        if t.is_storage {
            let Some(selector) = t.storage_selector else {
                crate::observe::fail(format!(
                    "compute_linux unsupported storage_format reason=no_storage_selector pipe={} bind={} fmt={:#x}",
                    acc.pipeline_ref, t.binding, t.pixel_format
                ));
                return ComputeStatus::Unsupported("storage_no_selector_writeback");
            };
            let Some(guest_fmt) = simg_u32_to_engine_storage(selector) else {
                crate::observe::fail(format!(
                    "compute_linux unsupported storage_format reason=selector_unknown pipe={} bind={} simg={selector} fmt={:#x}",
                    acc.pipeline_ref, t.binding, t.pixel_format
                ));
                return ComputeStatus::Unsupported("storage_selector_unknown_writeback");
            };
            let Some((_, _, shader_decl, specialized)) = storage_formats
                .iter()
                .find(|(binding, _, _, _)| *binding == t.binding)
            else {
                crate::observe::fail(format!(
                    "compute_linux storage_format fail reason=spirv_format_specialize_internal pipe={} bind={} simg={}",
                    acc.pipeline_ref, t.binding, selector
                ));
                return ComputeStatus::Unsupported("storage_format_specialize_internal");
            };
            // An `Unknown`-format storage image carries no SPIR-V texel format;
            // its engine format (and thus VkImageView) is the guest surface's
            // own format — here BGRA8Unorm → B8G8R8A8_UNORM — so the composite
            // write lands in guest channel order (the R/B-swap fix). Every other
            // format takes its engine format from the specialized SPIR-V format.
            let shader_fmt = if matches!(
                specialized,
                crate::runtime::spirv_bind::ImageFormat::Unknown
            ) {
                // Always-on proxy for the BGRA-storage-composite R/B class: this
                // line fires only on the corrected (without_format) path. Its
                // absence together with a `degraded_rb_swap` line below is the
                // regression signal that a swap is being emitted.
                crate::observe::off(format!(
                    "compute_linux bgra_storage_composite pipe={} bind={} mode=without_format guest={guest_fmt:?} view=B8G8R8A8_UNORM {}x{}",
                    acc.pipeline_ref, t.binding, t.width, t.height
                ));
                guest_fmt
            } else {
                let Some(fmt) = spirv_image_format_to_engine_storage(*specialized) else {
                    crate::observe::fail(format!(
                        "compute_linux storage_format fail reason=spirv_storage_format_unsupported pipe={} bind={} spirv={specialized:?} guest={guest_fmt:?} simg={}",
                        acc.pipeline_ref, t.binding, selector
                    ));
                    return ComputeStatus::Unsupported("storage_spirv_format_unsupported");
                };
                // Degraded path: a BGRA8Unorm guest fell back to a Rgba8Unorm
                // view because `shaderStorageImageWriteWithoutFormat` is absent —
                // the composite output is R/B-swapped. Fail-visible so the class
                // is never silent on an unsupported device.
                if matches!(
                    guest_fmt,
                    crate::backend::vulkan::engine::StorageImageFormat::Bgra8Unorm
                ) && matches!(
                    fmt,
                    crate::backend::vulkan::engine::StorageImageFormat::Rgba8Unorm
                ) {
                    crate::observe::fail(format!(
                        "compute_linux bgra_storage_composite pipe={} bind={} mode=degraded_rb_swap reason=no_storage_image_write_without_format {}x{}",
                        acc.pipeline_ref, t.binding, t.width, t.height
                    ));
                }
                fmt
            };
            if specialized != shader_decl {
                crate::observe::off(format!(
                    "compute_linux storage_format_specialize pipe={} bind={} spirv={shader_decl:?} specialized={specialized:?} engine={shader_fmt:?} guest={guest_fmt:?} simg={} guest_bpp={} shader_bpp={}",
                    acc.pipeline_ref,
                    t.binding,
                    selector,
                    guest_fmt.bytes_per_texel(),
                    spirv_image_format_to_engine_storage(*shader_decl)
                        .map(|format| format.bytes_per_texel())
                        .unwrap_or(0)
                ));
            }
            if shader_fmt != guest_fmt && t.width >= 1280 && t.height >= 720 {
                crate::observe::off(format!(
                    "compute_linux storage_format_view pipe={} bind={} spirv={specialized:?} engine={shader_fmt:?} guest={guest_fmt:?} simg={} bpp={}",
                    acc.pipeline_ref,
                    t.binding,
                    selector,
                    shader_fmt.bytes_per_texel()
                ));
            }
            // Deferred writeback: a resident type-11 output skips the engine
            // readback and the CPU guest writeback entirely — the pinned
            // resident is authoritative and every host access of the window
            // flushes first (storage_flush choke points). Linear windows only
            // carry `residency` when their defer gate passed at stage time
            // (cache-only + non-mirrorable), so residency alone qualifies
            // them. Direct writeback is moot when deferring.
            let defer_readback = compute_defer_readback_allowed(
                deferred_content_allowed,
                t.residency.is_some(),
                matches!(
                    t.writeback,
                    TextureWriteback::Type11 { .. } | TextureWriteback::Linear { .. }
                ),
            );
            let host_writeback = if defer_readback {
                None
            } else {
                plan_direct_writeback(state, host, t, direct_align)
            };
            storage_images.push(ComputeStorageImageResource {
                binding: t.binding,
                format: shader_fmt,
                width: t.width,
                height: t.height,
                layers: 1,
                one_dim: false,
                arrayed: false,
                volume: false,
                bytes: std::mem::take(&mut t.bytes),
                residency: t.residency.map(|candidate| {
                    crate::backend::vulkan::engine::ComputeStorageResidency {
                        identity: candidate.key,
                        seed_generation: candidate.seed_generation,
                        output_generation: next_mapping_content_generation(
                            candidate.seed_generation,
                        ),
                    }
                }),
                seed_skipped: t.seed_skipped,
                host_writeback,
                defer_readback,
            });
        } else {
            let Some(sampled_fmt) = mtl_to_engine_sampled(t.pixel_format) else {
                crate::observe::fail(format!(
                    "compute_linux sampled_format fail reason=mtl_format_unsupported pipe={} bind={} fmt={:#x}",
                    acc.pipeline_ref, t.binding, t.pixel_format
                ));
                return ComputeStatus::Unsupported("sampled_format_unsupported");
            };
            sampled_images.push(ComputeSampledImageResource {
                binding: t.binding,
                format: sampled_fmt,
                width: t.width,
                height: t.height,
                layers: 1,
                one_dim: false,
                arrayed: false,
                volume: false,
                bytes: std::mem::take(&mut t.bytes),
                resident_bind: t.sample_resident.map(|(identity, generation)| {
                    crate::backend::vulkan::engine::ComputeResidentSampleBind {
                        identity,
                        generation,
                    }
                }),
            });
        }
    }

    let mut samplers = Vec::new();
    for s in &acc.samplers {
        let binding = crate::runtime::spirv_bind::SAMPLER_BINDING_BASE + s.index;
        if !crate::runtime::spirv_bind::sampler_bindings(&spirv).contains(&binding) {
            continue;
        }
        let mut sampler = match crate::runtime::metal_draw::load_vulkan_sampler(
            state,
            host,
            task_id,
            s.sampler_ref,
            binding,
        ) {
            Ok(v) => v,
            Err(reason) => {
                crate::observe::Emit::decline("compute_linux_sampler", &reason)
                    .field("pipe", acc.pipeline_ref)
                    .fail_once((u64::from(s.sampler_ref) << 32) | u64::from(binding));
                return ComputeStatus::MissingSampler("compute_vk_sampler_load");
            }
        };
        if s.has_lod_clamp {
            sampler.lod_min = s.lod_min_bits;
            sampler.lod_max = s.lod_max_bits;
        }
        samplers.push(sampler);
    }
    for binding in crate::runtime::spirv_bind::sampler_bindings(&spirv) {
        if !samplers.iter().any(|sampler| sampler.binding == binding) {
            samplers
                .push(crate::backend::vulkan::engine::SamplerResource::normalized_default(binding));
        }
    }

    let mut req = ComputeRequest {
        spirv,
        entry: "main".into(),
        grid: [wg_x, wg_y, wg_z],
        storage_buffers,
        sampled_images,
        samplers,
        storage_images,
    };
    let prep_us = u64::try_from(prep_started.elapsed().as_micros()).unwrap_or(u64::MAX);
    let engine_before = vk_engine::counter_snapshot();
    let engine_started = std::time::Instant::now();
    let run_engine = |req: &ComputeRequest| {
        let engine_done = spawn_compute_engine_stall_watchdog(
            acc.pipeline_ref,
            req,
            std::time::Duration::from_millis(COMPUTE_ENGINE_STALL_PROXY_MS),
        );
        let out = vk_engine::execute_compute_request(req);
        engine_done.store(true, std::sync::atomic::Ordering::Release);
        out
    };
    let mut out_result = run_engine(&req);
    if let Err(e) = &out_result {
        let s = e.to_string();
        if s.contains("compute_resident_seed_lost") || s.contains("compute_resident_sample_lost") {
            // A resident vanished between the stage-time generation check and
            // engine acquire (same-request registry eviction). Re-read every
            // skipped seed and sampled resident window from guest pages and
            // retry once.
            if let Err(status) = restage_lost_residents(state, host, &mut req, acc.pipeline_ref) {
                return status;
            }
            out_result = run_engine(&req);
        }
    }
    let out = match out_result {
        Ok(o) => o,
        Err(e) => {
            let unsupported = matches!(&e, DrawError::Unsupported(_));
            crate::observe::Emit::decline("compute_linux_engine", &e)
                .field("pipe", acc.pipeline_ref)
                .fail_once(u64::from(acc.pipeline_ref));
            if unsupported {
                return ComputeStatus::Unsupported("engine_run_unsupported");
            }
            return ComputeStatus::MetalFailed("compute_vk_engine_run");
        }
    };
    let engine_us = u64::try_from(engine_started.elapsed().as_micros()).unwrap_or(u64::MAX);
    let engine_after = vk_engine::counter_snapshot();
    let engine_delta = engine_after.delta_since(&engine_before);
    let engine_phase_fields = compute_engine_phase_fields(engine_us, &engine_delta);

    if out.buffers.len() != buffer_writable_count
        || out.images.len() != storage_count
        || out.images_direct.len() != storage_count
        || out.images_deferred.len() != storage_count
    {
        crate::observe::fail(format!(
            "compute_linux readback count mismatch pipe={} buf={}/{} img={}/{} direct={}/{} deferred={}/{}",
            acc.pipeline_ref,
            out.buffers.len(),
            buffer_writable_count,
            out.images.len(),
            storage_count,
            out.images_direct.len(),
            storage_count,
            out.images_deferred.len(),
            storage_count
        ));
        return ComputeStatus::MetalFailed("compute_vk_readback_count");
    }
    let vk_engine::ComputeOutput {
        buffers: output_buffers,
        images: output_images,
        image_stats: output_image_stats,
        images_direct: output_images_direct,
        images_deferred: output_images_deferred,
    } = out;

    let writeback_started = std::time::Instant::now();
    for buffer in output_buffers {
        let Some(s) = staged_bufs
            .iter_mut()
            .find(|staged| staged.bind.index == buffer.binding)
        else {
            crate::observe::fail(format!(
                "compute_linux readback binding mismatch pipe={} bind={} bytes={}",
                acc.pipeline_ref,
                buffer.binding,
                buffer.bytes.len()
            ));
            return ComputeStatus::MetalFailed("compute_vk_readback_binding");
        };
        s.bytes = buffer.bytes;
        if let Err(e) = writeback_buffer(
            state,
            host,
            task_id,
            Some(acc.pipeline_ref),
            "vulkan_dispatch",
            s,
        ) {
            return e;
        }
    }
    for ((((t, bytes), fused_stats), direct), deferred) in staged_tex
        .iter_mut()
        .filter(|texture| texture.is_storage)
        .zip(output_images)
        .zip(output_image_stats)
        .zip(output_images_direct)
        .zip(output_images_deferred)
    {
        if deferred {
            // Deferred linear window: the pinned resident is the whole story —
            // today's sync path never wrote guest pages either (cache-only),
            // so the only bookkeeping is the cache entry's resident marker.
            if let (
                Some(candidate),
                TextureWriteback::Linear {
                    texture_ref,
                    gva,
                    pixel_format,
                    row_stride,
                    width,
                    height,
                    ..
                },
            ) = (t.residency, &t.writeback)
            {
                let generation = next_mapping_content_generation(candidate.seed_generation);
                if !crate::runtime::surface_cache::note_linear_texture_resident(
                    state,
                    task_id,
                    *texture_ref,
                    *gva,
                    *pixel_format,
                    *width,
                    *height,
                    *row_stride,
                    generation,
                ) {
                    crate::observe::fail(format!(
                        "compute_writeback_deferred fail reason=linear_note task={task_id} ref={texture_ref} gva={gva:#x} fmt={pixel_format:#x} dims={width}x{height} gen={generation}"
                    ));
                    return ComputeStatus::MetalFailed("compute_vk_deferred_linear_note");
                }
                // The sync path writes guest pages when the GVA is mapped —
                // record the flush obligation with a defer-time page index so
                // aliased raw-GVA readers land the content first. Any prior
                // obligation for this identity is superseded content. Pages
                // resolve fully at the defer edge (never at sample time —
                // the boot-19 guard-v1 regression).
                let key = candidate.key;
                state.disarm_linear_deferred_window(&key);
                let span = key.span_end;
                let guest_flush = state.gva_write_allowed(task_id, *gva, span);
                let mut indexed = 0usize;
                if guest_flush {
                    let mut pages = std::collections::HashSet::new();
                    crate::runtime::gva_mem::visit_task_gva_page_gpas(
                        host,
                        &state.tasks,
                        task_id,
                        *gva,
                        span,
                        state.page_shift,
                        1,
                        &mut |gpa_page| {
                            pages.insert(gpa_page);
                            true
                        },
                    );
                    indexed = pages.len();
                    state.arm_linear_deferred_window(key, generation, pages);
                }
                crate::observe::off(format!(
                    "compute_writeback_deferred kind=linear pipe={} bind={} task={task_id} ref={texture_ref} gva={gva:#x} {width}x{height} fmt={pixel_format:#x} gen={generation} guest_flush={} pages={indexed}",
                    acc.pipeline_ref,
                    t.binding,
                    guest_flush as u32
                ));
                continue;
            }
            // The pinned engine resident is authoritative; guest pages are now
            // stale until a flush choke point lands the content
            // (storage_flush::flush_intersecting). Keep the protocol
            // bookkeeping the CPU write would do, then register the window in
            // the deferred-flush map.
            let (Some(candidate), TextureWriteback::Type11 { mapping_id, .. }) =
                (t.residency, &t.writeback)
            else {
                crate::observe::fail(format!(
                    "compute_writeback_deferred fail reason=missing_identity pipe={} bind={}",
                    acc.pipeline_ref, t.binding
                ));
                return ComputeStatus::MetalFailed("compute_vk_deferred_identity");
            };
            let key = candidate.key;
            let generation = next_mapping_content_generation(candidate.seed_generation);
            // Superseded stale windows intersecting this one are dead content:
            // drop them (never flush over the newer output) and release their
            // pins — except our own identity, which the engine re-pinned.
            for (victim, victim_generation) in
                state.take_deferred_flush_windows(*mapping_id, key.surface_offset, key.span_end)
            {
                if victim != key {
                    crate::observe::off(format!(
                        "compute_writeback_deferred supersede mapping={mapping_id} victim={}x{} fmt={:#x} gen={victim_generation}",
                        victim.width, victim.height, victim.pixel_format
                    ));
                    crate::backend::vulkan::engine::unpin_resident_storage(&victim);
                }
            }
            state.compute_deferred_flush.insert(key, generation);
            state.index_deferred_alias_pages(*mapping_id);
            let _ = state.mark_mapping_written(*mapping_id);
            note_storage_residency_writeback(state, t);
            crate::observe::off(format!(
                "compute_writeback_deferred pipe={} bind={} mapping={mapping_id} {}x{} fmt={:#x} gen={generation}",
                acc.pipeline_ref, t.binding, key.width, key.height, key.pixel_format
            ));
            continue;
        }
        if direct {
            // The engine DMA'd the dispatch output straight into the guest
            // window; keep the protocol bookkeeping the CPU write would do
            // (exact-window residency invalidation + generation bump).
            let TextureWriteback::Type11 {
                mapping_id,
                surface_offset,
                span_end,
                ..
            } = &t.writeback
            else {
                crate::observe::fail(format!(
                    "compute_direct_writeback fail reason=non_type11_direct pipe={} bind={}",
                    acc.pipeline_ref, t.binding
                ));
                return ComputeStatus::MetalFailed("compute_vk_direct_non_type11");
            };
            state.invalidate_storage_residency_window(*mapping_id, *surface_offset, *span_end);
            let _ = state.mark_mapping_written(*mapping_id);
            note_storage_residency_writeback(state, t);
            log_compute_output_texture_direct(acc.pipeline_ref, t);
            continue;
        }
        t.bytes = bytes;
        let stats_started = std::time::Instant::now();
        log_compute_output_texture(
            acc.pipeline_ref,
            t,
            fused_stats.map(|s| (s.rgb_nz, s.rgb_max)),
        );
        let stats_us = stats_started.elapsed().as_micros() as u64;
        let write_started = std::time::Instant::now();
        if let Err(e) = writeback_texture(state, host, task_id, t) {
            return e;
        }
        let write_us = write_started.elapsed().as_micros() as u64;
        note_storage_residency_writeback(state, t);
        // Measure-only: split the writeback stall into the content-stats scan
        // vs the guest write.
        if stats_us + write_us > 1500 {
            crate::observe::off(format!(
                "compute_writeback_slow pipe={} bind={} {}x{} stats_us={stats_us} write_us={write_us}",
                acc.pipeline_ref, t.binding, t.width, t.height
            ));
        }
    }
    let writeback_us = u64::try_from(writeback_started.elapsed().as_micros()).unwrap_or(u64::MAX);
    let total_us = u64::try_from(total_started.elapsed().as_micros()).unwrap_or(u64::MAX);

    let snap = engine_after;
    crate::observe::off(format!(
        "compute_linux ok pipe={} wg=[{wg_x},{wg_y},{wg_z}] nbuf={} bro={} brw={} bunused={} ntex={} stage_us={stage_us} translate_us={translate_us} prep_us={prep_us} engine_us={engine_us} {engine_phase_fields} writeback_us={writeback_us} total_us={total_us} vk_engine_dispatches={} creates={} allocs={}",
        acc.pipeline_ref,
        staged_bufs.len(),
        buffer_readonly_count,
        buffer_writable_count,
        buffer_unused_count,
        staged_tex.len(),
        snap.dispatches,
        snap.creates,
        snap.allocs
    ));
    // Per-tranche attribution: compute dispatches are not render draws, so their
    // lock hold otherwise lands in the opaque `other_us` bucket.
    state.tranche.note_compute(total_us);
    ComputeStatus::Ok
}

#[cfg(feature = "backend-vulkan")]
fn compute_engine_phase_fields(
    engine_us: u64,
    delta: &crate::backend::vulkan::engine::CounterSnapshot,
) -> String {
    let resource_measured_us = delta
        .sampler_prepare_us
        .saturating_add(delta.storage_prepare_us)
        .saturating_add(delta.sampled_prepare_us)
        .saturating_add(delta.storage_image_prepare_us)
        .saturating_add(delta.descriptor_prepare_us);
    let resource_unattributed_us = delta.resource_us.saturating_sub(resource_measured_us);
    let measured_us = delta
        .lock_wait_us
        .saturating_add(delta.context_us)
        .saturating_add(delta.pool_init_us)
        .saturating_add(delta.cache_us)
        .saturating_add(delta.resource_us)
        .saturating_add(delta.pre_record_wait_us)
        .saturating_add(delta.record_us)
        .saturating_add(delta.submit_us)
        .saturating_add(delta.wait_us)
        .saturating_add(delta.retire_wait_us)
        .saturating_add(delta.readback_us)
        .saturating_add(delta.cleanup_us);
    let unattributed_us = engine_us.saturating_sub(measured_us);
    format!(
        "engine_lock_wait_us={} engine_context_us={} engine_pool_init_us={} engine_cache_us={} engine_shader_create_us={} engine_layout_create_us={} engine_pipeline_create_us={} engine_sampler_create_us={} engine_resource_us={} engine_resource_unattributed_us={} engine_sampler_prepare_us={} engine_storage_buffer_prepare_us={} engine_sampled_image_prepare_us={} engine_storage_image_prepare_us={} engine_descriptor_prepare_us={} engine_memory_alloc_us={} engine_pre_record_wait_us={} engine_record_us={} engine_submit_us={} engine_wait_us={} engine_retire_wait_us={} engine_post_wait_skips={} engine_ring_retire_blocks={} engine_readback_us={} engine_cleanup_us={} engine_unattributed_us={} engine_readbacks={} engine_readback_bytes={} engine_sampled_uploads={} engine_sampled_upload_bytes={} engine_sampled_resident_copies={} engine_sampled_resident_copy_bytes={} engine_sampled_reinterpret_copies={} engine_sampled_reinterpret_copy_bytes={} engine_storage_seed_uploads={} engine_storage_seed_upload_bytes={} engine_direct_writebacks={} engine_direct_writeback_bytes={} engine_direct_writeback_fallbacks={} engine_creates={} engine_allocs={}",
        delta.lock_wait_us,
        delta.context_us,
        delta.pool_init_us,
        delta.cache_us,
        delta.shader_create_us,
        delta.layout_create_us,
        delta.pipeline_create_us,
        delta.sampler_create_us,
        delta.resource_us,
        resource_unattributed_us,
        delta.sampler_prepare_us,
        delta.storage_prepare_us,
        delta.sampled_prepare_us,
        delta.storage_image_prepare_us,
        delta.descriptor_prepare_us,
        delta.memory_alloc_us,
        delta.pre_record_wait_us,
        delta.record_us,
        delta.submit_us,
        delta.wait_us,
        delta.retire_wait_us,
        delta.compute_post_wait_skips,
        delta.ring_retire_blocks,
        delta.readback_us,
        delta.cleanup_us,
        unattributed_us,
        delta.readbacks,
        delta.readback_bytes,
        delta.compute_sampled_uploads,
        delta.compute_sampled_upload_bytes,
        delta.compute_sampled_resident_copies,
        delta.compute_sampled_resident_copy_bytes,
        delta.compute_sampled_reinterpret_copies,
        delta.compute_sampled_reinterpret_copy_bytes,
        delta.compute_storage_seed_uploads,
        delta.compute_storage_seed_upload_bytes,
        delta.compute_direct_writebacks,
        delta.compute_direct_writeback_bytes,
        delta.compute_direct_writeback_fallbacks,
        delta.creates,
        delta.allocs
    )
}

#[cfg(feature = "backend-vulkan")]
const COMPUTE_ENGINE_STALL_PROXY_MS: u64 = 2_000;

/// Resolve a GPU-direct writeback window for a staged type-11 storage image:
/// the contig host view of the destination mapping plus the surface window,
/// for the engine to import (VK_EXT_external_memory_host) and copy into
/// instead of the host-visible readback buffer. `None` keeps the CPU
/// writeback path — correct, just slower. Texel-size/stride checks live
/// engine-side (it knows the view format); the host-pointer conditions are
/// checked here where the mapping is nameable.
#[cfg(feature = "backend-vulkan")]
fn plan_direct_writeback<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    tex: &StagedTexture,
    align: Option<u64>,
) -> Option<crate::backend::vulkan::engine::ComputeHostWriteback> {
    let align = align?;
    let TextureWriteback::Type11 {
        mapping_id,
        surface_offset,
        surface_bpr,
        span_end,
        ..
    } = &tex.writeback
    else {
        return None;
    };
    let (mapping_id, surface_offset, surface_bpr, span_end) =
        (*mapping_id, *surface_offset, *surface_bpr, *span_end);
    let Some((ptr, len)) = mapping_write::contig_ptr_for_span(state, host, mapping_id, span_end)
    else {
        // Fragmented mapping: the CPU multi-import writeback stays in charge.
        crate::observe::off(format!(
            "compute_direct_writeback skip reason=no_contig mid={mapping_id} span_end={span_end}"
        ));
        return None;
    };
    if !(ptr as u64).is_multiple_of(align) {
        crate::observe::off(format!(
            "compute_direct_writeback skip reason=ptr_align mid={mapping_id} ptr={ptr:#x} align={align}"
        ));
        return None;
    }
    Some(crate::backend::vulkan::engine::ComputeHostWriteback {
        ptr,
        len,
        buffer_offset: surface_offset,
        row_bytes: surface_bpr,
    })
}

/// Re-read every seed-skipped storage image from guest pages after the engine
/// reported `compute_resident_seed_lost` (the resident vanished between the
/// stage-time generation check and acquire — same-request registry eviction).
/// Guest pages still hold the last writeback for that generation, so the
/// re-read reconstructs exactly the seed the skip elided. Clears the skip
/// flags so the retry seeds normally. Fail-visible on every exit: seed loss is
/// rare, and a flood of restage lines means the stage-time gate and the engine
/// registry disagree.
#[cfg(feature = "backend-vulkan")]
fn restage_lost_residents<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    req: &mut crate::backend::vulkan::engine::ComputeRequest,
    pipeline_ref: u32,
) -> Result<(), ComputeStatus> {
    // bytes was allocated as exactly tight*height at stage time, so the
    // identity key recovers the read geometry for both resource kinds.
    let read_window = |state: &mut DeviceState,
                       host: &mut M,
                       key: &crate::model::ComputeStorageResidencyKey,
                       bytes: &mut Vec<u8>| {
        let rows = key.height.max(1) as u64;
        let tight = (bytes.len() as u64 / rows) as u32;
        let bpp = tight / key.width.max(1);
        bpp != 0
            && mapping_write::read_rect_raw_at(
                state,
                host,
                key.mapping_id,
                key.surface_offset,
                key.surface_bpr,
                key.span_end,
                0,
                0,
                key.width,
                key.height,
                bpp,
                bytes,
                tight,
            )
    };
    let mut restaged_storage = 0u32;
    for resource in req.storage_images.iter_mut().filter(|r| r.seed_skipped) {
        let Some(residency) = resource.residency else {
            crate::observe::fail(format!(
                "compute_restage fail reason=skip_without_identity pipe={pipeline_ref} bind={}",
                resource.binding
            ));
            return Err(ComputeStatus::MetalFailed(
                "compute_restage_skip_without_identity",
            ));
        };
        let key = residency.identity;
        // A linear window's deferred content lives only in the lost resident —
        // there is no mapping to re-read (guest pages hold the pre-chain
        // seed and may be unmapped). Terminal, named.
        if key.is_linear() {
            crate::observe::fail(format!(
                "compute_restage fail reason=linear_resident_lost pipe={pipeline_ref} bind={} task={} ref={} gva={:#x} {}x{} fmt={:#x}",
                resource.binding,
                key.map_generation,
                key.texture_ref,
                key.surface_offset,
                key.width,
                key.height,
                key.pixel_format
            ));
            return Err(ComputeStatus::MetalFailed("compute_restage_linear_lost"));
        }
        // Heap textures have no guest-memory backing. Their contents exist
        // only in the engine resident created for the heap allocation.
        if key.is_heap() {
            crate::observe::fail(format!(
                "compute_restage fail reason=heap_resident_lost pipe={pipeline_ref} bind={} task={} ref={} {}x{} fmt={:#x}",
                resource.binding,
                key.map_generation,
                key.texture_ref,
                key.width,
                key.height,
                key.pixel_format
            ));
            return Err(ComputeStatus::MetalFailed("compute_restage_heap_lost"));
        }
        // The resident is gone; if its window was writeback-deferred the guest
        // pages hold PRE-chain bytes — the deferred content is lost with it.
        // Name the loss and fall back to the coherent stale seed.
        if let Some(generation) = state.compute_deferred_flush.remove(&key) {
            crate::observe::fail(format!(
                "deferred_flush_lost mapping={} reason=restage gen={generation} {}x{} fmt={:#x}",
                key.mapping_id, key.width, key.height, key.pixel_format
            ));
        }
        if !read_window(state, host, &key, &mut resource.bytes) {
            crate::observe::fail(format!(
                "compute_restage fail reason=read pipe={pipeline_ref} bind={} mapping={} {}x{} off={} bpr={} span_end={}",
                resource.binding,
                key.mapping_id,
                key.width,
                key.height,
                key.surface_offset,
                key.surface_bpr,
                key.span_end
            ));
            return Err(ComputeStatus::GuestIo("compute_restage_read"));
        }
        resource.seed_skipped = false;
        restaged_storage += 1;
    }
    let mut restaged_sampled = 0u32;
    for resource in req.sampled_images.iter_mut() {
        let Some(bind) = resource.resident_bind else {
            continue;
        };
        let key = bind.identity;
        if key.is_linear() {
            crate::observe::fail(format!(
                "compute_restage fail reason=linear_resident_lost pipe={pipeline_ref} bind={} task={} ref={} gva={:#x} {}x{} fmt={:#x}",
                resource.binding,
                key.map_generation,
                key.texture_ref,
                key.surface_offset,
                key.width,
                key.height,
                key.pixel_format
            ));
            return Err(ComputeStatus::MetalFailed(
                "compute_restage_sampled_linear_lost",
            ));
        }
        if key.is_heap() {
            crate::observe::fail(format!(
                "compute_restage fail reason=heap_resident_lost pipe={pipeline_ref} bind={} task={} ref={} {}x{} fmt={:#x}",
                resource.binding,
                key.map_generation,
                key.texture_ref,
                key.width,
                key.height,
                key.pixel_format
            ));
            return Err(ComputeStatus::MetalFailed(
                "compute_restage_sampled_heap_lost",
            ));
        }
        if !read_window(state, host, &key, &mut resource.bytes) {
            crate::observe::fail(format!(
                "compute_restage fail reason=sampled_read pipe={pipeline_ref} bind={} mapping={} {}x{} off={} bpr={} span_end={}",
                resource.binding,
                key.mapping_id,
                key.width,
                key.height,
                key.surface_offset,
                key.surface_bpr,
                key.span_end
            ));
            return Err(ComputeStatus::GuestIo("compute_restage_sampled_read"));
        }
        resource.resident_bind = None;
        restaged_sampled += 1;
    }
    if restaged_storage == 0 && restaged_sampled == 0 {
        // The engine claimed a lost resident but no resource was skipped —
        // contract breach between stage and acquire; do not retry blind.
        crate::observe::fail(format!(
            "compute_restage fail reason=no_skipped_resource pipe={pipeline_ref}"
        ));
        return Err(ComputeStatus::MetalFailed(
            "compute_restage_no_skipped_resource",
        ));
    }
    crate::observe::fail(format!(
        "compute_resident_seed_restage pipe={pipeline_ref} n={restaged_storage} sampled={restaged_sampled}"
    ));
    Ok(())
}

/// Measurement-only watchdog for backend calls that cannot be bounded by a
/// Vulkan fence timeout (notably pipeline creation and some driver submits).
/// It never changes execution. A fired proxy preserves the private request
/// inputs under /tmp so the stall can be reproduced without another VM boot.
#[cfg(feature = "backend-vulkan")]
fn spawn_compute_engine_stall_watchdog(
    pipeline_ref: u32,
    req: &crate::backend::vulkan::engine::ComputeRequest,
    threshold: std::time::Duration,
) -> std::sync::Arc<std::sync::atomic::AtomicBool> {
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::Arc;

    let done = Arc::new(AtomicBool::new(false));
    let thread_done = Arc::clone(&done);
    let spirv = req.spirv.clone();
    let grid = req.grid;
    let buffers = req.storage_buffers.len();
    let images = req.storage_images.len();
    let image_geometry: Vec<_> = req
        .storage_images
        .iter()
        .map(|img| (img.binding, img.width, img.height, img.layers))
        .collect();
    std::thread::spawn(move || {
        std::thread::sleep(threshold);
        if thread_done.load(Ordering::Acquire) {
            return;
        }
        let elapsed_ms = threshold.as_millis();
        crate::observe::fail(format!(
            "compute_engine_stall reason=backend_call_unreturned pipe={pipeline_ref} elapsed_ms={elapsed_ms} grid={grid:?} nbuf={buffers} nimg={images} image_geom={image_geometry:?}"
        ));
        let base = format!("/tmp/reims-vgpu-compute-stall-pipe-{pipeline_ref}");
        let mut bytes = Vec::with_capacity(spirv.len().saturating_mul(4));
        for word in spirv {
            bytes.extend_from_slice(&word.to_le_bytes());
        }
        if let Err(e) = std::fs::write(format!("{base}.spv"), &bytes) {
            crate::observe::fail(format!(
                "compute_engine_stall reason=spv_dump_failed pipe={pipeline_ref} err={e}"
            ));
        }
        let meta = format!(
            "pipe={pipeline_ref}\nelapsed_ms={elapsed_ms}\ngrid={grid:?}\nnbuf={buffers}\nnimg={images}\nimage_geom={image_geometry:?}\n"
        );
        if let Err(e) = std::fs::write(format!("{base}.txt"), meta) {
            crate::observe::fail(format!(
                "compute_engine_stall reason=metadata_dump_failed pipe={pipeline_ref} err={e}"
            ));
        }
    });
    done
}

fn spirv_words_le(bytes: &[u8]) -> Result<Vec<u32>, ComputeSpirvDecline> {
    const HEADER_LEN: usize = 20;
    const WORD_ALIGNMENT: usize = 4;
    if bytes.len() < HEADER_LEN {
        return Err(ComputeSpirvDecline::HeaderTooShort {
            len: bytes.len(),
            minimum: HEADER_LEN,
        });
    }
    if !bytes.len().is_multiple_of(WORD_ALIGNMENT) {
        return Err(ComputeSpirvDecline::LengthMisaligned {
            len: bytes.len(),
            alignment: WORD_ALIGNMENT,
        });
    }
    Ok(bytes
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}

/// Thin `Option` adapters over the canonical tables in
/// [`crate::backend::vulkan::translate::pixel`].
///
/// These two used to *be* the tables — a second copy of the selector→engine and
/// Metal→engine mappings living in the compute path, where nothing checked them
/// against the pixel table they had to agree with. The call sites below are all
/// `if let Some(..)` / `let Some(..) else`, so the adapters keep that shape; the
/// decision itself now happens in exactly one place.
#[cfg(feature = "backend-vulkan")]
fn simg_u32_to_engine_storage(
    simg: u32,
) -> Option<crate::backend::vulkan::engine::StorageImageFormat> {
    crate::backend::vulkan::translate::pixel::storage_image_from_selector(simg).ok()
}

#[cfg(feature = "backend-vulkan")]
fn mtl_to_engine_sampled(
    format: u16,
) -> Option<crate::backend::vulkan::engine::StorageImageFormat> {
    crate::backend::vulkan::translate::pixel::storage_image(format).ok()
}

#[cfg(feature = "backend-vulkan")]
fn spirv_image_format_to_engine_storage(
    format: crate::runtime::spirv_bind::ImageFormat,
) -> Option<crate::backend::vulkan::engine::StorageImageFormat> {
    use crate::backend::vulkan::engine::StorageImageFormat as V;
    use crate::runtime::spirv_bind::ImageFormat as S;
    Some(match format {
        S::Rgba32Float => V::Rgba32Float,
        S::Rgba16Float => V::Rgba16Float,
        S::R16Float => V::R16Float,
        S::Rgba16Uint => V::Rgba16Uint,
        S::Rgba8Uint => V::Rgba8Uint,
        S::Rgba8Sint => V::Rgba8Sint,
        S::Rgba8Unorm => V::Rgba8Unorm,
        S::Rg16Float => V::Rg16Float,
        S::R8Unorm => V::R8Unorm,
        S::Rg8Unorm => V::Rg8Unorm,
        S::Rgba32Uint => V::Rgba32Uint,
        S::R32Float => V::R32Float,
        S::R32ui => V::R32Uint,
        // Format-less (`Unknown`) storage images carry no engine texel format —
        // their view format comes from the guest surface, resolved by the caller.
        S::Unknown | S::Unsupported(_) => return None,
    })
}

#[cfg(feature = "backend-vulkan")]
fn specialized_storage_image_format(
    guest: crate::backend::vulkan::engine::StorageImageFormat,
    shader: crate::runtime::spirv_bind::ImageFormat,
    write_without_format: bool,
) -> Result<crate::runtime::spirv_bind::ImageFormat, &'static str> {
    use crate::backend::vulkan::engine::StorageImageFormat as V;
    use crate::runtime::spirv_bind::ImageFormat as S;

    let Some(shader_engine) = spirv_image_format_to_engine_storage(shader) else {
        return Err("spirv_storage_format_unsupported");
    };
    // A guest BGRA8Unorm surface written by a normalized (float/unorm-class)
    // shader is a color store. SPIR-V has no `Bgra8` storage format, so a
    // concrete `Rgba8Unorm` view would store the shader's red at the guest's
    // blue byte — the resolution-independent R/B swap. Retarget to a format-less
    // `Unknown` storage image; the engine views it `B8G8R8A8_UNORM` (guest
    // channel order) and the GPU converts the written vec4 to BGRA natively, so
    // every downstream consumer (writeback, resident export, sampling) sees the
    // correct bytes with no per-frame swizzle. Requires
    // `StorageImageWriteWithoutFormat`; when absent we degrade to the swapped
    // `Rgba8Unorm` view and the caller logs the degraded class.
    //
    // A uint/sint shader over BGRA is instead a deliberate raw byte view (byte
    // order preserved, no conversion) and must keep its raw format — it falls
    // through to the raw-view / class-matched logic below, unchanged.
    if matches!(guest, V::Bgra8Unorm) {
        let normalized_color_store = matches!(
            shader,
            S::Rgba8Unorm
                | S::Rgba32Float
                | S::Rgba16Float
                | S::R16Float
                | S::R32Float
                | S::Rg16Float
                | S::R8Unorm
                | S::Rg8Unorm
        );
        if normalized_color_store {
            return Ok(if write_without_format {
                S::Unknown
            } else {
                S::Rgba8Unorm
            });
        }
    }
    if guest.bytes_per_texel() == shader_engine.bytes_per_texel() && !matches!(guest, V::R32Uint) {
        // Equal-size mismatches are intentional raw views (for example Metal
        // BGRA8Unorm bound to a uint texture and translated as Rgba8Uint).
        //
        // R32Uint is the ONE equal-bytes case that is NOT a valid raw view: the
        // translator declares a 4x8-bit `Rgba8ui` storage image for a generic
        // `texture2d<uint, write>`, but the guest surface is a single 32-bit
        // uint channel. Reinterpreting it as Rgba8ui would store only the low
        // byte of each written lane (correct only for values < 256). It falls
        // through to the class-matched specialization below, which re-targets
        // the SPIR-V storage image to `R32ui` (VK_FORMAT_R32_UINT) so a written
        // `uint4`'s `.x` lane is stored as the full u32.
        return Ok(shader);
    }

    let shader_class = match shader {
        S::Rgba32Float
        | S::Rgba16Float
        | S::R16Float
        | S::R32Float
        | S::Rgba8Unorm
        | S::Rg16Float
        | S::R8Unorm
        | S::Rg8Unorm => 0,
        S::Rgba32Uint | S::Rgba16Uint | S::Rgba8Uint | S::R32ui => 1,
        S::Rgba8Sint => 2,
        // A shader that itself declared `Unknown` (format-less) storage is not a
        // class we specialize by numeric class; the caller only mints `Unknown`
        // deliberately for the BGRA path, which returns above.
        S::Unknown | S::Unsupported(_) => return Err("spirv_storage_format_unsupported"),
    };
    let (guest_class, specialized) = match guest {
        // R32-single-channel: R32Uint is supported as a storage image by
        // re-targeting the SPIR-V to `R32ui` (its class must still match the
        // shader's numeric class below — a uint-write shader). The remaining
        // R32 sint/float and the packed Rgb9e5 stay sampled-only until a live
        // capture justifies enabling their storage path.
        V::R32Uint => (1, S::R32ui),
        V::R32Sint | V::R32Float | V::Rgb9e5Ufloat => {
            return Err("spirv_sampled_only_format_as_storage");
        }
        V::Rgba32Float => (0, S::Rgba32Float),
        V::Rgba16Float => (0, S::Rgba16Float),
        V::R16Float => (0, S::R16Float),
        // Bgra8Unorm normally returns above (Unknown/B8G8R8A8 view, or the
        // degraded Rgba8Unorm) before reaching here; this arm is only the
        // class/bytes fallthrough for Rgba8Unorm and a defensive default.
        V::Rgba8Unorm | V::Bgra8Unorm => (0, S::Rgba8Unorm),
        V::Rg16Float => (0, S::Rg16Float),
        V::R8Unorm => (0, S::R8Unorm),
        V::Rg8Unorm => (0, S::Rg8Unorm),
        V::Rgba32Uint => (1, S::Rgba32Uint),
        V::Rgba16Uint => (1, S::Rgba16Uint),
        V::Rgba8Uint => (1, S::Rgba8Uint),
        V::Rgba8Sint => (2, S::Rgba8Sint),
    };
    if shader_class != guest_class {
        return Err("spirv_guest_numeric_class_mismatch");
    }
    Ok(specialized)
}

#[cfg(all(feature = "backend-metal", target_os = "macos"))]
fn stage_input_to_apv(
    si: &ComputeStageInputDescriptor,
) -> crate::backend::metal::abi::ReimsVgpuComputeStageInputDescriptor {
    use crate::backend::metal::abi::{
        ReimsVgpuComputeStageInputAttribute, ReimsVgpuComputeStageInputDescriptor,
        ReimsVgpuComputeStageInputLayout, REIMS_VGPU_COMPUTE_STAGE_INPUT_MAX_ATTRIBUTES,
        REIMS_VGPU_COMPUTE_STAGE_INPUT_MAX_LAYOUTS,
    };
    let mut out = ReimsVgpuComputeStageInputDescriptor {
        word0: si.word0,
        header0: si.header0,
        header1: si.header1,
        attribute_count: si.attributes.len() as u32,
        layout_count: si.layouts.len() as u32,
        index_type: si.index_type,
        index_buffer_index: si.index_buffer_index,
        attributes: [ReimsVgpuComputeStageInputAttribute {
            raw_bits: 0,
            location: 0,
            format: 0,
            offset: 0,
            buffer_index: 0,
            reserved0: 0,
        }; REIMS_VGPU_COMPUTE_STAGE_INPUT_MAX_ATTRIBUTES],
        layouts: [ReimsVgpuComputeStageInputLayout {
            raw_bits: 0,
            buffer_index: 0,
            step_function: 0,
            step_rate: 0,
            stride: 0,
        }; REIMS_VGPU_COMPUTE_STAGE_INPUT_MAX_LAYOUTS],
    };
    for (i, a) in si
        .attributes
        .iter()
        .enumerate()
        .take(REIMS_VGPU_COMPUTE_STAGE_INPUT_MAX_ATTRIBUTES)
    {
        out.attributes[i] = ReimsVgpuComputeStageInputAttribute {
            raw_bits: a.raw_bits,
            location: a.location,
            format: a.format,
            offset: a.offset,
            buffer_index: a.buffer_index,
            reserved0: 0,
        };
    }
    for (i, l) in si
        .layouts
        .iter()
        .enumerate()
        .take(REIMS_VGPU_COMPUTE_STAGE_INPUT_MAX_LAYOUTS)
    {
        out.layouts[i] = ReimsVgpuComputeStageInputLayout {
            raw_bits: l.raw_bits,
            buffer_index: l.buffer_index,
            step_function: l.step_function,
            step_rate: l.step_rate,
            stride: l.stride,
        };
    }
    out
}

#[cfg(all(feature = "backend-metal", target_os = "macos"))]
fn execute_dispatch_metal<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    task_id: u32,
    acc: &ComputeAccum,
    cmd: &ComputeCommand,
    session: Option<&mut crate::runtime::compute_session::ComputeSession>,
) -> ComputeStatus {
    use crate::backend::metal::abi::{
        ReimsVgpuBuffer, ReimsVgpuComputeImageblockDimensions, ReimsVgpuComputeSampledImage,
        ReimsVgpuComputeStageInRegion, ReimsVgpuComputeStageInRegionIndirectArguments,
        ReimsVgpuComputeTextureUsage, ReimsVgpuSampler, ReimsVgpuStorageImage,
        ReimsVgpuThreadgroupMemory, REIMS_VGPU_BINDING_SAMPLER_BASE,
        REIMS_VGPU_BINDING_TEXTURE_BASE, REIMS_VGPU_COMPUTE_DISPATCH_KIND_THREADGROUPS,
        REIMS_VGPU_COMPUTE_DISPATCH_KIND_THREADS, REIMS_VGPU_COMPUTE_TEXTURE_ACCESS_READ,
        REIMS_VGPU_COMPUTE_TEXTURE_ACCESS_READ_WRITE, REIMS_VGPU_COMPUTE_TEXTURE_ACCESS_WRITE,
        REIMS_VGPU_MTL_DISPATCH_TYPE_CONCURRENT, REIMS_VGPU_MTL_DISPATCH_TYPE_SERIAL,
    };
    use crate::backend::metal::compute::{
        compute_core, compute_encode_on_encoder, reflect_compute_textures_mtlb,
    };
    if acc.pipeline_ref == 0 {
        return ComputeStatus::MissingPipeline("compute_mtl_pipeline_ref_zero");
    }
    let Some(pipeline) = load_compute_pipeline(state, host, task_id, acc.pipeline_ref) else {
        return ComputeStatus::MissingPipeline("compute_mtl_pipeline_load");
    };
    let Some(mtlb) = load_mtlb(state, host, task_id, pipeline.kernel_func_ref) else {
        return ComputeStatus::MissingMtlb("compute_mtl_mtlb_load");
    };

    let (grid_x, grid_y, grid_z, tg_x, tg_y, tg_z, dispatch_threads) =
        match resolve_dispatch_dims(state, host, task_id, acc, cmd) {
            Ok(v) => v,
            Err(e) => {
                crate::observe::line(format!(
                "compute_resolve_dims fail {e:?} kind={:?} grid=[{},{},{}] tg=[{},{},{}] ntex={}",
                cmd.kind,
                cmd.grid.x,
                cmd.grid.y,
                cmd.grid.z,
                cmd.threads_per_threadgroup.x,
                cmd.threads_per_threadgroup.y,
                cmd.threads_per_threadgroup.z,
                acc.textures.len()
            ));
                return e;
            }
        };

    let dispatch_kind = if dispatch_threads {
        REIMS_VGPU_COMPUTE_DISPATCH_KIND_THREADS
    } else {
        REIMS_VGPU_COMPUTE_DISPATCH_KIND_THREADGROUPS
    };
    let dispatch_type = if acc.dispatch_type == REIMS_VGPU_MTL_DISPATCH_TYPE_CONCURRENT {
        REIMS_VGPU_MTL_DISPATCH_TYPE_CONCURRENT
    } else {
        REIMS_VGPU_MTL_DISPATCH_TYPE_SERIAL
    };

    // Stage-input descriptor from pipeline (optional).
    let reims_vgpu_stage_input = pipeline.stage_input.as_ref().map(stage_input_to_apv);

    // Direct / indirect stage-in region.
    let direct_region = acc
        .stage_in_region
        .as_ref()
        .map(|r| ReimsVgpuComputeStageInRegion {
            origin_x: r.origin_x,
            origin_y: r.origin_y,
            origin_z: r.origin_z,
            size_x: r.size_x,
            size_y: r.size_y,
            size_z: r.size_z,
        });
    let mut indirect_region_args: Option<ReimsVgpuComputeStageInRegionIndirectArguments> = None;
    if let Some(ind) = &acc.stage_in_region_indirect {
        let raw = match read_buffer_window(
            state,
            host,
            task_id,
            ind.buffer_ref,
            ind.buffer_offset,
            STAGE_IN_INDIRECT_ARGS_LEN,
        ) {
            Ok(b) => b,
            Err(e) => return e,
        };
        indirect_region_args = Some(ReimsVgpuComputeStageInRegionIndirectArguments {
            origin_x: ld32(&raw[0..]),
            origin_y: ld32(&raw[4..]),
            origin_z: ld32(&raw[8..]),
            size_x: ld32(&raw[12..]),
            size_y: ld32(&raw[16..]),
            size_z: ld32(&raw[20..]),
        });
    }
    let imageblock = acc
        .imageblock
        .as_ref()
        .map(|d| ReimsVgpuComputeImageblockDimensions {
            width: d.width,
            height: d.height,
        });
    let tg_mem: Vec<ReimsVgpuThreadgroupMemory> = acc
        .threadgroup_memory
        .iter()
        .map(|t| ReimsVgpuThreadgroupMemory {
            index: t.index,
            length: t.length,
        })
        .collect();

    let mut staged_bufs: Vec<StagedBuffer> = Vec::new();
    for b in &acc.buffers {
        match stage_buffer(state, host, task_id, b) {
            Ok(s) => staged_bufs.push(s),
            Err(e) => return e,
        }
    }

    // Texture reflection: access decides storage vs sampled materialization.
    let mut usages = vec![
        ReimsVgpuComputeTextureUsage {
            binding: 0,
            access: 0,
        };
        32
    ];
    let mut usage_count = 0usize;
    let mut err_buf = [0i8; 256];
    if !acc.textures.is_empty() {
        let st = reflect_compute_textures_mtlb(
            &mtlb,
            usages.as_mut_ptr(),
            usages.len(),
            &mut usage_count,
            (err_buf.as_mut_ptr(), err_buf.len()),
        );
        if !st.is_ok() {
            return ComputeStatus::MetalBackend(st);
        }
        usages.truncate(usage_count);
    } else {
        usages.clear();
    }

    let access_for = |binding: u32| -> Option<u32> {
        usages
            .iter()
            .find(|u| u.binding == binding)
            .map(|u| u.access)
    };

    let mut staged_tex: Vec<StagedTexture> = Vec::new();
    for t in &acc.textures {
        let binding = REIMS_VGPU_BINDING_TEXTURE_BASE + t.index;
        let access = access_for(binding).unwrap_or(REIMS_VGPU_COMPUTE_TEXTURE_ACCESS_READ_WRITE);
        let is_storage = access != REIMS_VGPU_COMPUTE_TEXTURE_ACCESS_READ;
        let stage_call_started = std::time::Instant::now();
        match stage_texture_raw(state, host, task_id, t.texture_ref, binding, is_storage) {
            Ok(s) => {
                // Measure-only: localize per-texture stage cost (the
                // transition-window guest stall).
                let us = stage_call_started.elapsed().as_micros() as u64;
                if us > 1500 {
                    crate::observe::off(format!(
                        "compute_stage_slow pipe={} ref={} bind={binding} storage={} {}x{} fmt={:#x} us={us}",
                        acc.pipeline_ref,
                        t.texture_ref,
                        is_storage as u8,
                        s.width,
                        s.height,
                        s.pixel_format
                    ));
                }
                staged_tex.push(s)
            }
            Err(e) => return e,
        }
    }

    // Samplers.
    let mut reims_vgpu_samplers: Vec<ReimsVgpuSampler> = Vec::new();
    for s in &acc.samplers {
        let entry = match objects::lookup_list_entry(state, host, task_id, s.sampler_ref) {
            Some(e) => e,
            None => return ComputeStatus::MissingSampler("compute_mtl_sampler_no_entry"),
        };
        if entry.object_type != OBJECT_TYPE_TYPE7 {
            return ComputeStatus::MissingSampler("compute_mtl_sampler_wrong_type");
        }
        let desc = match objects::read_descriptor(state, host, task_id, &entry) {
            Some(d) => d,
            None => return ComputeStatus::MissingSampler("compute_mtl_sampler_no_desc"),
        };
        if desc.len() < 4 || ld32(&desc) != TYPE7_OBJECT_SAMPLER {
            return ComputeStatus::MissingSampler("compute_mtl_sampler_bad_tag");
        }
        let sd = match decode_sampler_descriptor(&desc) {
            Ok(v) => v,
            Err(_) => return ComputeStatus::MissingSampler("compute_mtl_sampler_decode"),
        };
        let mut lod_min = sd.lod_min_clamp.to_bits();
        let mut lod_max = sd.lod_max_clamp.to_bits();
        let mut has_lod = 1u32;
        if s.has_lod_clamp {
            lod_min = s.lod_min_bits;
            lod_max = s.lod_max_bits;
            has_lod = 1;
        }
        reims_vgpu_samplers.push(ReimsVgpuSampler {
            binding: REIMS_VGPU_BINDING_SAMPLER_BASE + s.index,
            unnormalized: if sd.normalized_coordinates { 0 } else { 1 },
            min_filter: sd.min_filter,
            mag_filter: sd.mag_filter,
            mip_filter: sd.mip_filter,
            s_address_mode: sd.s_address,
            t_address_mode: sd.t_address,
            r_address_mode: sd.r_address,
            border_color: sd.border_color,
            compare_function: sd.compare_function,
            lod_min_bits: lod_min,
            lod_max_bits: lod_max,
            max_anisotropy: sd.max_anisotropy.max(1),
            lod_average: if sd.lod_average { 1 } else { 0 },
            support_argument_buffers: if sd.support_argument_buffers { 1 } else { 0 },
            has_lod_clamp: has_lod,
            clamp_lod_min_bits: lod_min,
            clamp_lod_max_bits: lod_max,
        });
    }

    let mut reims_vgpu_bufs: Vec<ReimsVgpuBuffer> = staged_bufs
        .iter_mut()
        .map(|s| ReimsVgpuBuffer {
            binding: s.bind.index,
            data: s.bytes.as_mut_ptr(),
            len: s.bytes.len(),
            attribute_stride: s.bind.attribute_stride,
            has_attribute_stride: if s.bind.has_attribute_stride { 1 } else { 0 },
            reserved0: 0,
            backing_data: std::ptr::null_mut(),
            backing_len: 0,
            backing_offset: 0,
        })
        .collect();

    let mut storage: Vec<ReimsVgpuStorageImage> = Vec::new();
    let mut sampled: Vec<ReimsVgpuComputeSampledImage> = Vec::new();
    // Keep raw pointers valid: build storage/sampled from staged_tex after mut split.
    for t in &mut staged_tex {
        let Some(selector) = t.storage_selector else {
            crate::observe::fail(format!(
                "compute_metal texture_format fail reason=no_backend_selector pipe={} bind={} fmt={:#x}",
                acc.pipeline_ref, t.binding, t.pixel_format
            ));
            return ComputeStatus::Unsupported("metal_no_backend_selector");
        };
        if t.is_storage {
            storage.push(ReimsVgpuStorageImage {
                binding: t.binding,
                format: selector,
                width: t.width,
                height: t.height,
                data: t.bytes.as_mut_ptr(),
                len: t.bytes.len(),
            });
        } else {
            sampled.push(ReimsVgpuComputeSampledImage {
                binding: t.binding,
                format: selector,
                width: t.width,
                height: t.height,
                data: t.bytes.as_ptr(),
                len: t.bytes.len(),
                has_swizzle: 0,
                swizzle: [2, 3, 4, 5], // identity RGBA selectors
            });
        }
    }

    // Nested: encode onto open session encoder; writeback after segment commit.
    if let Some(sess) = session {
        let retain = match compute_encode_on_encoder(
            &sess.device,
            &sess.encoder,
            &mtlb,
            &mut reims_vgpu_bufs,
            &mut storage,
            &sampled,
            &reims_vgpu_samplers,
            &tg_mem,
            direct_region.as_ref(),
            indirect_region_args.as_ref(),
            imageblock.as_ref(),
            reims_vgpu_stage_input.as_ref(),
            dispatch_kind,
            grid_x,
            grid_y,
            grid_z,
            tg_x,
            tg_y,
            tg_z,
            (err_buf.as_mut_ptr(), err_buf.len()),
        ) {
            Ok(r) => r,
            Err(st) => return ComputeStatus::MetalBackend(st),
        };
        // Split storage textures out of staged_tex for deferred writeback alignment.
        let storage_tex: Vec<StagedTexture> =
            staged_tex.into_iter().filter(|t| t.is_storage).collect();
        if storage_tex.len() != retain.images.len() {
            return ComputeStatus::MetalFailed("compute_mtl_retain_image_count");
        }
        sess.retained.extend(retain.buffers.iter().cloned());
        sess.retained.extend(retain.indirect.iter().cloned());
        for t in &retain.sampled {
            let _ = t; // lifetime held by session via NestedDispatchJob storage only
        }
        sess.nested_jobs.push(NestedDispatchJob {
            staged_bufs,
            storage_tex,
            mtl_buffers: retain.buffers,
            mtl_storage: retain.images,
        });
        let _ = (
            dispatch_type,
            OBJECT_TYPE_IOSURFACE,
            REIMS_VGPU_COMPUTE_TEXTURE_ACCESS_WRITE,
        );
        return ComputeStatus::Ok;
    }

    let st = compute_core(
        &mtlb,
        &mut reims_vgpu_bufs,
        &mut storage,
        &sampled,
        &reims_vgpu_samplers,
        &tg_mem,
        direct_region.as_ref(),
        indirect_region_args.as_ref(),
        imageblock.as_ref(),
        reims_vgpu_stage_input.as_ref(),
        dispatch_kind,
        dispatch_type,
        grid_x,
        grid_y,
        grid_z,
        tg_x,
        tg_y,
        tg_z,
        (err_buf.as_mut_ptr(), err_buf.len()),
    );
    if !st.is_ok() {
        return ComputeStatus::MetalBackend(st);
    }

    for s in &staged_bufs {
        if let Err(e) = writeback_buffer(
            state,
            host,
            task_id,
            Some(acc.pipeline_ref),
            "metal_dispatch",
            s,
        ) {
            return e;
        }
    }
    for t in &staged_tex {
        if let Err(e) = writeback_texture(state, host, task_id, t) {
            return e;
        }
    }
    let _ = (
        OBJECT_TYPE_IOSURFACE,
        REIMS_VGPU_COMPUTE_TEXTURE_ACCESS_WRITE,
    );
    ComputeStatus::Ok
}

#[cfg(test)]
mod tests;

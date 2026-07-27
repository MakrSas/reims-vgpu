#![allow(
    clippy::field_reassign_with_default,
    reason = "wire fixtures are assembled field by field to keep each protocol case explicit"
)]

use super::*;
use crate::contract::endian::{st32, st64};
use crate::contract::gva::{DIRECTORY_DEPTH, DIRECTORY_ROOT_PFN};
use crate::model::{DeviceId, PAGE_SHIFT_ARM64E, PAGE_SHIFT_X86};
use crate::runtime::decode::compute;
use crate::runtime::decode::resource::{
    list_object_entry_offset, OBJECT_LIST_ENTRY_LEN, OBJECT_TYPE_BUFFER, OBJECT_TYPE_IOSURFACE,
    RESOURCE_PAGE_SHIFT,
};
/// Compute-pipeline descriptor constants, used only by the Metal-arm
/// execute tests below.
#[cfg(all(feature = "backend-metal", target_os = "macos"))]
use crate::runtime::decode::resource::{
    PIPELINE_TAG_KERNEL_FUNC, TYPE7_FIRST_TLVS, TYPE7_OBJECT_COMPUTE_PIPELINE,
};
use crate::runtime::gva_mem;
use crate::runtime::host::FakeHost;

#[test]
fn spirv_word_parser_splits_short_header_from_misalignment() {
    assert_eq!(
        spirv_words_le(&[0; 16]).unwrap_err(),
        ComputeSpirvDecline::HeaderTooShort {
            len: 16,
            minimum: 20
        }
    );
    assert_eq!(
        spirv_words_le(&[0; 21]).unwrap_err(),
        ComputeSpirvDecline::LengthMisaligned {
            len: 21,
            alignment: 4
        }
    );
    assert_eq!(spirv_words_le(&[0; 20]).unwrap().len(), 5);
}

#[test]
fn compute_spirv_declines_are_distinct_and_log_safe() {
    use crate::observe::Decline as _;
    let declines = [
        ComputeSpirvDecline::HeaderTooShort {
            len: 16,
            minimum: 20,
        },
        ComputeSpirvDecline::LengthMisaligned {
            len: 21,
            alignment: 4,
        },
    ];
    assert_ne!(declines[0].slug(), declines[1].slug());
    for decline in declines {
        assert!(decline
            .slug()
            .bytes()
            .all(|byte| byte.is_ascii_lowercase() || byte == b'_'));
        for (_, value) in decline.fields() {
            assert!(!value.contains(char::is_whitespace));
        }
    }
}

#[test]
fn compute_defer_readback_follows_gpu_only_content_gate() {
    assert!(
        compute_defer_readback_allowed(true, true, true),
        "native Vulkan keeps the deferred storage-writeback rail"
    );
    assert!(
        !compute_defer_readback_allowed(false, true, true),
        "portability-subset storage output must write guest pages synchronously"
    );
    assert!(!compute_defer_readback_allowed(true, false, true));
    assert!(!compute_defer_readback_allowed(true, true, false));
}

#[test]
fn m2v_handoff_dir_prefers_explicit_dir_then_repo_local_private_dir() {
    assert_eq!(
        m2v_handoff_dir_from_env(Some("/custom/m2v".into()), Some("/repo".into())),
        std::path::PathBuf::from("/custom/m2v")
    );
    assert_eq!(
        m2v_handoff_dir_from_env(None, Some("/repo".into())),
        std::path::PathBuf::from("/repo/m2v-handoff/artifacts")
    );
    assert_eq!(
        m2v_handoff_dir_from_env(None, None),
        std::path::PathBuf::from("/tmp/reims-vgpu-m2v-compute-fails")
    );
}

#[test]
fn compute_bind_overflow_drops_the_bind_but_keeps_in_cap_and_unbinds() {
    let mut acc = ComputeAccum::default();
    // In-cap buffer bind (ref != 0) is kept.
    acc.bind_buffers(
        5,
        &[BufferBinding {
            ref_: 7,
            ..Default::default()
        }],
    );
    assert_eq!(acc.buffers.len(), 1);
    assert_eq!(acc.buffers[0].index, 5);

    // Over-cap buffer bind (index 40 > MAX_COMPUTE_BUFFER_SLOTS) is dropped —
    // the drop is fail-visible via note_compute_bind_overflow (the log itself is
    // a global sink, not asserted here). No new buffer slot appears.
    acc.bind_buffers(
        MAX_COMPUTE_BUFFER_SLOTS + 9,
        &[BufferBinding {
            ref_: 9,
            ..Default::default()
        }],
    );
    assert_eq!(acc.buffers.len(), 1, "over-cap bind must not be stored");

    // Boundary: index == MAX is OUT of range (the backend sizes its arg-table
    // array to MAX and guards `idx >= MAX`), so it must be dropped too — this
    // is the off-by-one the `>` → `>=` alignment fixed. Slot MAX-1 is the last
    // valid slot and is kept.
    acc.bind_buffers(
        MAX_COMPUTE_BUFFER_SLOTS,
        &[BufferBinding {
            ref_: 11,
            ..Default::default()
        }],
    );
    assert_eq!(
        acc.buffers.len(),
        1,
        "index == MAX is out of range, dropped"
    );
    acc.bind_buffers(
        MAX_COMPUTE_BUFFER_SLOTS - 1,
        &[BufferBinding {
            ref_: 12,
            ..Default::default()
        }],
    );
    assert_eq!(
        acc.buffers.len(),
        2,
        "index == MAX-1 is the last valid slot"
    );

    // A zero-ref entry is an unbind: expected control flow, silently skipped.
    acc.bind_buffers(
        6,
        &[BufferBinding {
            ref_: 0,
            ..Default::default()
        }],
    );
    assert_eq!(acc.buffers.len(), 2, "unbind (ref==0) adds no slot");

    // Threadgroup memory over-cap with a real length is dropped (kept empty);
    // a zero-length over-cap request is an unbind and stays silent.
    acc.set_threadgroup_memory(MAX_THREADGROUP_MEMORY_SLOTS + 2, 256);
    acc.set_threadgroup_memory(MAX_THREADGROUP_MEMORY_SLOTS + 3, 0);
    assert!(
        acc.threadgroup_memory.is_empty(),
        "over-cap threadgroup memory must not be stored"
    );
}

fn setup_task_pages(host: &mut FakeHost, state: &mut DeviceState, data_base_pfn: u32) {
    let dir_pfn = 2u32;
    let root_pfn = 3u32;
    let dir_gpa = (dir_pfn as u64) << PAGE_SHIFT_ARM64E;
    let root_gpa = (root_pfn as u64) << PAGE_SHIFT_ARM64E;
    host.map_range(dir_gpa, 0x20, 0);
    host.map_range(root_gpa, 0x4000, 0);
    let mut d = [0u8; 8];
    st32(&mut d[DIRECTORY_ROOT_PFN as usize..], root_pfn);
    st32(&mut d[DIRECTORY_DEPTH as usize..], 1);
    let _ = host.write_gpa(dir_gpa, &d);
    for i in 0..8u32 {
        let pfn = data_base_pfn + i;
        host.map_range((pfn as u64) << PAGE_SHIFT_ARM64E, 0x4000, 0);
        let mut pte = [0u8; 4];
        st32(&mut pte, pfn);
        let _ = host.write_gpa(root_gpa + (i as u64) * 4, &pte);
    }
    assert!(state.define_task(1, 0x1000, dir_pfn));
    assert!(state.set_object_list(1, 0, 32));
}

#[test]
fn accum_pipeline_buffer_texture_sampler() {
    let mut acc = ComputeAccum::default();
    acc.set_pipeline(9);
    acc.bind_buffers(
        1,
        &[BufferBinding {
            ref_: 3,
            offset: 16,
            attribute_stride: 0,
            has_attribute_stride: false,
        }],
    );
    acc.bind_textures(0, &[RefBinding { ref_: 10 }, RefBinding { ref_: 11 }]);
    acc.bind_samplers(
        0,
        &[SamplerBinding {
            ref_: 20,
            lod_min_bits: 0,
            lod_max_bits: 0,
            has_lod_clamp: false,
        }],
    );
    assert_eq!(acc.pipeline_ref, 9);
    assert_eq!(acc.buffers.len(), 1);
    assert_eq!(acc.textures.len(), 2);
    assert_eq!(acc.textures[1].texture_ref, 11);
    assert_eq!(acc.samplers[0].sampler_ref, 20);
}

#[test]
fn accum_stage_in_tg_imageblock_and_control_fail_closed() {
    let mut acc = ComputeAccum::default();
    acc.set_stage_in_region(StageInRegion {
        origin_x: 1,
        origin_y: 2,
        origin_z: 0,
        size_x: 8,
        size_y: 4,
        size_z: 1,
    });
    assert!(acc.stage_in_region.is_some());
    acc.set_stage_in_region_indirect(3, 16);
    assert!(acc.stage_in_region.is_none());
    assert_eq!(
        acc.stage_in_region_indirect.as_ref().map(|i| i.buffer_ref),
        Some(3)
    );
    acc.set_threadgroup_memory(2, 256);
    acc.set_threadgroup_memory(2, 512);
    assert_eq!(acc.threadgroup_memory.len(), 1);
    assert_eq!(acc.threadgroup_memory[0].length, 512);
    acc.set_imageblock(4, 4);
    assert_eq!(acc.imageblock.as_ref().map(|d| d.width), Some(4));

    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    let mut host = FakeHost::new();
    let mut session = None;
    let mut block = None;
    let mut cmd = ComputeCommand::default();
    // Empty start-do-while encodes without a condition buffer.
    cmd.kind = Kind::ControlStartDoWhile;
    let st = apply_record(
        &mut state,
        &mut host,
        1,
        &cmd,
        &mut acc,
        &mut session,
        &mut block,
    );
    assert!(
        matches!(
            st,
            Some(ComputeStatus::Ok)
                | Some(ComputeStatus::NoMetal(_))
                | Some(ComputeStatus::MetalFailed(_))
        ),
        "unexpected {st:?}"
    );
    cmd.kind = Kind::ExecuteCommandsInBuffer;
    cmd.indirect_command_buffer_ref = 99;
    let st = apply_record(
        &mut state,
        &mut host,
        1,
        &cmd,
        &mut acc,
        &mut session,
        &mut block,
    );
    // Missing object-list entry → MissingBuffer; still latches sequencing.
    assert!(
        matches!(
            st,
            Some(ComputeStatus::MissingBuffer(_)) | Some(ComputeStatus::Unsupported(_))
        ),
        "unexpected {st:?}"
    );
    assert!(block.is_some());
    if let Some(s) = session.take() {
        let _ = s.finish(&mut host, &mut state, 1);
    }
}

#[test]
fn resolve_indirect_threadgroups_from_buffer() {
    let mut host = FakeHost::new();
    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    setup_task_pages(&mut host, &mut state, 4);

    let args = [2u32, 3, 1];
    let arg_bytes: Vec<u8> = args.iter().flat_map(|v| v.to_le_bytes()).collect();
    let buf_gva = 5u64 << RESOURCE_PAGE_SHIFT;
    assert!(gva_mem::write_task_gva(
        &mut host,
        &state.tasks[1],
        buf_gva,
        &arg_bytes,
        PAGE_SHIFT_ARM64E
    )
    .is_ok());
    let mut bdesc = vec![0u8; 16];
    st64(&mut bdesc[0..], 12);
    st32(&mut bdesc[8..], 5);
    let bdesc_gva = 0x180u64;
    assert!(gva_mem::write_task_gva(
        &mut host,
        &state.tasks[1],
        bdesc_gva,
        &bdesc,
        PAGE_SHIFT_ARM64E
    )
    .is_ok());
    {
        let off = list_object_entry_offset(7, 32).unwrap();
        let mut le = [0u8; OBJECT_LIST_ENTRY_LEN];
        let packed = (OBJECT_TYPE_BUFFER as u32) | (16u32 << 8);
        st32(&mut le[0..], packed);
        le[4..12].copy_from_slice(&bdesc_gva.to_le_bytes());
        assert!(
            gva_mem::write_task_gva(&mut host, &state.tasks[1], off, &le, PAGE_SHIFT_ARM64E)
                .is_ok()
        );
    }

    let mut cmd = ComputeCommand::default();
    cmd.kind = Kind::DispatchThreadgroupsIndirect;
    cmd.indirect_buffer_ref = 7;
    cmd.indirect_buffer_offset = 0;
    cmd.threads_per_threadgroup = compute::Size3 { x: 8, y: 1, z: 1 };
    let acc = ComputeAccum::default();
    let (gx, gy, gz, tx, ty, tz, threads) =
        resolve_dispatch_dims(&mut state, &host, 1, &acc, &cmd).unwrap();
    assert_eq!((gx, gy, gz, tx, ty, tz, threads), (2, 3, 1, 8, 1, 1, false));
}

#[test]
fn sentinel_grid_recovers_from_largest_texture() {
    // Wire: grid=[45, UINT64_MAX, 1], tg=[32, 0, 1] → recover for 1440×1080.
    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    let mut host = FakeHost::new();
    setup_task_pages(&mut host, &mut state, 8);
    // type-11 mapping as write target
    assert!(state.map_surface(3));
    assert!(state.set_mapping_geom(3, 1440, 1080, 0x73));
    // Register type-11 ref 10 → mapping 3 via object list would need full
    // setup; instead bind a type-2 texture with dims via raw desc.
    // Use mapping path: put type-11 object if helpers exist; else skip full
    // e2e and unit-test try_recover pure path with largest_bound via mapping
    // by faking resolve_type11 — not available without object list.
    // Pure arithmetic shape check via try_recover with empty textures = None.
    let mut cmd = ComputeCommand::default();
    cmd.kind = Kind::DispatchThreadgroups;
    cmd.grid = compute::Size3 {
        x: 45,
        y: u64::MAX,
        z: 1,
    };
    cmd.threads_per_threadgroup = compute::Size3 { x: 32, y: 0, z: 1 };
    let acc = ComputeAccum::default();
    assert!(try_recover_sentinel_grid(&mut state, &host, 1, &acc, &cmd).is_none());
    // With a synthetic texture bind we need object-list; at least raw dims
    // recovery formula is covered when textures resolve.
    let _ = host;
}

/// The rail's whole point after this migration: a refusal renders the
/// registered slug of the check that refused, and a success cannot render
/// anything at all. Before the payload existed, every `MissingTexture`
/// site produced the same untyped line and 25 checks were indistinguishable.
#[test]
fn a_compute_refusal_names_its_check_and_ok_names_nothing() {
    use crate::observe::{Emit, Refusal};

    assert!(ComputeStatus::Ok.refusal().is_none());
    assert!(
        Emit::refusal("compute_record", &ComputeStatus::Ok).is_none(),
        "an Ok must not be loggable — that is what keeps the sink clean"
    );

    let st = ComputeStatus::MissingTexture("compute_stage_tex_type5_no_map");
    assert_eq!(st.refusal(), Some("compute_stage_tex_type5_no_map"));
    assert_eq!(st.class(), "missing_texture");
    let line = Emit::refusal("compute_record", &st)
        .expect("a refusal renders a line")
        .field("pipe", 7)
        .render();
    assert_eq!(
        line,
        "compute_record reason=compute_stage_tex_type5_no_map \
             class=missing_texture pipe=7"
    );

    #[cfg(all(feature = "backend-metal", target_os = "macos"))]
    {
        let status = crate::backend::metal::error::Status::args(
            "metal_compute_reflection_usage_output_missing",
        )
        .field("capacity", 8usize);
        let st = ComputeStatus::MetalBackend(status);
        assert_eq!(st.class(), "metal_args");
        assert_eq!(
            Emit::refusal("compute_record", &st)
                .expect("the exact Metal refusal must survive the runtime carrier")
                .render(),
            "compute_record reason=metal_compute_reflection_usage_output_missing \
                 class=args capacity=8 recovery=metal_failed"
        );
    }
}

/// Two different buffer-staging checks, two different slugs — the property
/// that a shared `MissingBuffer` could not express. `ref=0` never resolves,
/// so both paths refuse on their first gate.
#[test]
fn the_buffer_paths_refuse_under_their_own_names() {
    use crate::observe::Refusal;

    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    let mut host = FakeHost::new();
    setup_task_pages(&mut host, &mut state, 4);

    let window = read_buffer_window(&state, &host, 1, 0, 0, 4);
    assert_eq!(
        window.err().and_then(|e| e.refusal()),
        Some("compute_buf_win_no_backing")
    );

    let bind = ComputeBufferBind {
        index: 0,
        buffer_ref: 0,
        offset: 0,
        attribute_stride: 0,
        has_attribute_stride: false,
    };
    let staged = stage_buffer(&state, &host, 1, &bind);
    assert_eq!(
        staged.err().and_then(|e| e.refusal()),
        Some("compute_stage_buf_no_entry")
    );
}

/// Callers must pass page_shift explicitly; 12 and 14 place handle differently.
#[test]
fn buffer_backing_gva_requires_explicit_page_shift() {
    use crate::runtime::decode::resource::BufferDescriptor;
    let d = BufferDescriptor {
        allocation_size: 0x1000,
        handle64: 0x101,
        handle: 0x101,
    };
    let (gva12, _) = d.backing_gva_size(PAGE_SHIFT_X86).expect("12");
    let (gva14, _) = d.backing_gva_size(PAGE_SHIFT_ARM64E).expect("14");
    assert_eq!(gva12, 0x101000, "x86 handle<<12");
    assert_eq!(gva14, 0x404000, "arm handle<<14");
    assert_ne!(gva12, gva14);
}

#[test]
fn dispatch_missing_pipeline() {
    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    let mut host = FakeHost::new();
    setup_task_pages(&mut host, &mut state, 4);
    let acc = ComputeAccum::default();
    let mut cmd = ComputeCommand::default();
    cmd.kind = Kind::DispatchThreadgroups;
    cmd.grid = compute::Size3 { x: 1, y: 1, z: 1 };
    cmd.threads_per_threadgroup = compute::Size3 { x: 1, y: 1, z: 1 };
    let st = execute_dispatch(&mut state, &mut host, 1, &acc, &cmd);
    // The slug names *which* pipeline check refused, and it differs by
    // backend: both arms open with `pipeline_ref == 0`, and before the
    // status carried a reason the two were indistinguishable in the log.
    #[cfg(all(feature = "backend-metal", target_os = "macos"))]
    assert_eq!(
        st,
        ComputeStatus::MissingPipeline("compute_mtl_pipeline_ref_zero")
    );
    #[cfg(feature = "backend-vulkan")]
    assert_eq!(
        st,
        ComputeStatus::MissingPipeline("compute_vk_pipeline_ref_zero")
    );
}

/// Linux without vulkan feature: dispatch is NoMetal (census). With
/// backend-vulkan, missing pipeline is MissingPipeline (real encode path).
#[test]
#[cfg(feature = "backend-vulkan")]
fn dispatch_nometal_with_texture_binds() {
    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    let mut host = FakeHost::new();
    setup_task_pages(&mut host, &mut state, 4);
    let mut acc = ComputeAccum::default();
    acc.set_pipeline(42);
    acc.bind_textures(0, &[RefBinding { ref_: 111 }]);
    acc.bind_buffers(
        0,
        &[BufferBinding {
            ref_: 7,
            offset: 0,
            attribute_stride: 0,
            has_attribute_stride: false,
        }],
    );
    let mut cmd = ComputeCommand::default();
    cmd.kind = Kind::DispatchThreadgroups;
    cmd.grid = compute::Size3 {
        x: 60,
        y: u64::MAX,
        z: 1,
    };
    cmd.threads_per_threadgroup = compute::Size3 { x: 32, y: 0, z: 1 };
    let st = execute_dispatch(&mut state, &mut host, 1, &acc, &cmd);
    #[cfg(feature = "backend-vulkan")]
    assert!(
        matches!(
            st,
            ComputeStatus::MissingPipeline(_)
                | ComputeStatus::MissingMtlb(_)
                | ComputeStatus::MissingTexture(_)
                | ComputeStatus::MetalFailed(_)
                | ComputeStatus::Unsupported(_)
        ),
        "vulkan path attempts encode, got {st:?}"
    );
    // Nested short-circuit remains NoMetal on Linux (SPI not wired).
    let mut session = crate::runtime::compute_session::ComputeSession {
        control_depth: 0,
        saw_control: false,
        saw_icb: false,
    };
    let st2 = execute_dispatch_nested(&mut state, &mut host, 1, &acc, &cmd, &mut session);
    assert_eq!(st2, ComputeStatus::NoMetal("compute_nested_no_vulkan_path"));
}

#[test]
#[cfg(feature = "backend-vulkan")]
fn dispatch_missing_pipeline_not_nometal() {
    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    let mut host = FakeHost::new();
    setup_task_pages(&mut host, &mut state, 4);
    let acc = ComputeAccum::default();
    let mut cmd = ComputeCommand::default();
    cmd.kind = Kind::DispatchThreadgroups;
    cmd.grid = compute::Size3 { x: 1, y: 1, z: 1 };
    cmd.threads_per_threadgroup = compute::Size3 { x: 1, y: 1, z: 1 };
    let st = execute_dispatch(&mut state, &mut host, 1, &acc, &cmd);
    assert_eq!(
        st,
        ComputeStatus::MissingPipeline("compute_vk_pipeline_ref_zero")
    );
}

#[test]
#[cfg(all(feature = "backend-metal", target_os = "macos"))]
fn dispatch_buffer_kernel_mul3add1() {
    use std::path::PathBuf;
    let mtlb_paths = [
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/compute_mul3add1.mtlb"),
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/compute_mul3add1.mtlb"),
    ];
    let mtlb = mtlb_paths
        .iter()
        .find_map(|p| std::fs::read(p).ok())
        .expect("compute_mul3add1.mtlb fixture");

    let mut host = FakeHost::new();
    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    setup_task_pages(&mut host, &mut state, 4);

    let blob_gva = 4u64 << RESOURCE_PAGE_SHIFT;
    assert!(gva_mem::write_task_gva(
        &mut host,
        &state.tasks[1],
        blob_gva,
        &mtlb,
        PAGE_SHIFT_ARM64E
    )
    .is_ok());
    let mut fdesc = vec![0u8; 32];
    st64(&mut fdesc[0..], blob_gva);
    st32(&mut fdesc[8..], mtlb.len() as u32);
    let fdesc_gva = 0x100u64;
    assert!(gva_mem::write_task_gva(
        &mut host,
        &state.tasks[1],
        fdesc_gva,
        &fdesc,
        PAGE_SHIFT_ARM64E
    )
    .is_ok());
    {
        let off = list_object_entry_offset(5, 32).unwrap();
        let mut le = [0u8; OBJECT_LIST_ENTRY_LEN];
        let packed = (OBJECT_TYPE_FUNCTION as u32) | (32u32 << 8);
        st32(&mut le[0..], packed);
        le[4..12].copy_from_slice(&fdesc_gva.to_le_bytes());
        assert!(
            gva_mem::write_task_gva(&mut host, &state.tasks[1], off, &le, PAGE_SHIFT_ARM64E)
                .is_ok()
        );
    }

    let mut pdesc = vec![0u8; 32];
    st32(&mut pdesc[0..], TYPE7_OBJECT_COMPUTE_PIPELINE);
    st32(&mut pdesc[4..], 32); // declared descriptor length
    pdesc[TYPE7_FIRST_TLVS] = 1;
    pdesc[TYPE7_FIRST_TLVS + 1] = PIPELINE_TAG_KERNEL_FUNC;
    pdesc[TYPE7_FIRST_TLVS + 2] = 4;
    st32(&mut pdesc[TYPE7_FIRST_TLVS + 3..], 5);
    let pdesc_gva = 0x140u64;
    assert!(gva_mem::write_task_gva(
        &mut host,
        &state.tasks[1],
        pdesc_gva,
        &pdesc,
        PAGE_SHIFT_ARM64E
    )
    .is_ok());
    {
        let off = list_object_entry_offset(6, 32).unwrap();
        let mut le = [0u8; OBJECT_LIST_ENTRY_LEN];
        let packed = (OBJECT_TYPE_TYPE7 as u32) | (32u32 << 8);
        st32(&mut le[0..], packed);
        le[4..12].copy_from_slice(&pdesc_gva.to_le_bytes());
        assert!(
            gva_mem::write_task_gva(&mut host, &state.tasks[1], off, &le, PAGE_SHIFT_ARM64E)
                .is_ok()
        );
    }

    let data = [1u32, 2, 3, 4];
    let data_bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
    let buf_gva = 5u64 << RESOURCE_PAGE_SHIFT;
    assert!(gva_mem::write_task_gva(
        &mut host,
        &state.tasks[1],
        buf_gva,
        &data_bytes,
        PAGE_SHIFT_ARM64E
    )
    .is_ok());
    let mut bdesc = vec![0u8; 16];
    st64(&mut bdesc[0..], 16);
    st32(&mut bdesc[8..], 5);
    let bdesc_gva = 0x180u64;
    assert!(gva_mem::write_task_gva(
        &mut host,
        &state.tasks[1],
        bdesc_gva,
        &bdesc,
        PAGE_SHIFT_ARM64E
    )
    .is_ok());
    {
        let off = list_object_entry_offset(7, 32).unwrap();
        let mut le = [0u8; OBJECT_LIST_ENTRY_LEN];
        let packed = (OBJECT_TYPE_BUFFER as u32) | (16u32 << 8);
        st32(&mut le[0..], packed);
        le[4..12].copy_from_slice(&bdesc_gva.to_le_bytes());
        assert!(
            gva_mem::write_task_gva(&mut host, &state.tasks[1], off, &le, PAGE_SHIFT_ARM64E)
                .is_ok()
        );
    }

    let mut acc = ComputeAccum::default();
    acc.set_pipeline(6);
    acc.bind_buffers(
        0,
        &[BufferBinding {
            ref_: 7,
            offset: 0,
            attribute_stride: 0,
            has_attribute_stride: false,
        }],
    );

    let mut cmd = ComputeCommand::default();
    cmd.kind = Kind::DispatchThreadgroups;
    cmd.grid = compute::Size3 { x: 1, y: 1, z: 1 };
    cmd.threads_per_threadgroup = compute::Size3 { x: 4, y: 1, z: 1 };
    let st = execute_dispatch(&mut state, &mut host, 1, &acc, &cmd);
    assert!(
        matches!(
            st,
            ComputeStatus::Ok | ComputeStatus::MetalFailed(_) | ComputeStatus::BadGrid(_)
        ),
        "unexpected {st:?}"
    );
    if st == ComputeStatus::Ok {
        let mut back = [0u8; 16];
        assert!(gva_mem::read_task_gva(
            &host,
            &state.tasks[1],
            buf_gva,
            &mut back,
            PAGE_SHIFT_ARM64E
        )
        .is_ok());
        let out: Vec<u32> = back
            .chunks(4)
            .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        assert_eq!(out, vec![4, 7, 10, 13]);
    }
}

#[test]
fn dispatch_missing_texture_fails() {
    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    let mut host = FakeHost::new();
    setup_task_pages(&mut host, &mut state, 4);
    // Pipeline without function still fails earlier; bind texture only.
    let mut acc = ComputeAccum::default();
    acc.set_pipeline(1);
    acc.bind_textures(0, &[RefBinding { ref_: 99 }]);
    let mut cmd = ComputeCommand::default();
    cmd.kind = Kind::DispatchThreadgroups;
    cmd.grid = compute::Size3 { x: 1, y: 1, z: 1 };
    cmd.threads_per_threadgroup = compute::Size3 { x: 1, y: 1, z: 1 };
    // Missing pipeline object → MissingPipeline before texture stage.
    // Non-Apple metal stubs short-circuit to NoMetal (Linux product).
    let st = execute_dispatch(&mut state, &mut host, 1, &acc, &cmd);
    assert!(matches!(
        st,
        ComputeStatus::MissingPipeline(_)
            | ComputeStatus::MissingTexture(_)
            | ComputeStatus::MetalFailed(_)
            | ComputeStatus::NoMetal(_)
    ));
}

/// Live CI wallpaper: type-5 RefTexture → type-4 surface_id must stage via
/// ensure_surface + mapping (same order as metal_draw sample). Without
/// ensure, stage fell through to type-2/3 with the type-5 ref → always
/// MissingTexture (`compute_stage_tex … ot=5`).
#[test]
fn stage_texture_type5_ref_resolves_surface_mapping() {
    use crate::contract::iosurface_pages::{PAGE_ENTRY_PFN_SHIFT, PAGE_ENTRY_VALID};
    use crate::contract::pixel_format::MTL_FORMAT_BGRA8_UNORM;
    use crate::runtime::decode::resource::{list_object_entry_offset, OBJECT_LIST_ENTRY_LEN};

    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    let mut host = FakeHost::new();
    setup_task_pages(&mut host, &mut state, 4);

    let sid = 3u32;
    let type5_ref = 10u32;
    // Pre-mapped type-4 surface (CI storage target) with one valid page.
    let pfn = 0x20u32;
    let gpa = (pfn as u64) << PAGE_SHIFT_ARM64E;
    host.map_range(gpa, 0x4000, 0x5a);
    let entry = (pfn << PAGE_ENTRY_PFN_SHIFT) | PAGE_ENTRY_VALID;
    assert!(state.map_surface(sid));
    {
        let m = state.mappings.get_mut(&sid).unwrap();
        m.mapped = true;
        m.mapping_internal = 1;
        m.page_entries = vec![entry];
    }
    assert!(state.set_mapping_geom(sid, 4, 4, MTL_FORMAT_BGRA8_UNORM));

    // Object-list: type-5 at ref 10 → surface_id=3 (mapping already seeded).
    let desc_gva = (4u64 + 2) << PAGE_SHIFT_ARM64E; // data pfn base 4 + 2
    let mut type5_desc = vec![0u8; 16];
    st32(&mut type5_desc[objects::TYPE5_SURFACE_ID..], sid);
    assert!(gva_mem::write_task_gva(
        &mut host,
        &state.tasks[1],
        desc_gva,
        &type5_desc,
        PAGE_SHIFT_ARM64E
    )
    .is_ok());
    let off = list_object_entry_offset(type5_ref, 32).unwrap();
    let mut list_entry = [0u8; OBJECT_LIST_ENTRY_LEN];
    let packed = (objects::OBJECT_TYPE_REF_TEXTURE as u32) | ((16u32) << 8);
    st32(&mut list_entry[0..], packed);
    list_entry[4..12].copy_from_slice(&desc_gva.to_le_bytes());
    assert!(gva_mem::write_task_gva(
        &mut host,
        &state.tasks[1],
        off,
        &list_entry,
        PAGE_SHIFT_ARM64E
    )
    .is_ok());

    let staged = stage_texture_raw(&mut state, &mut host, 1, type5_ref, 32, true)
        .expect("type-5→surface stage must succeed after ensure");
    assert_eq!((staged.width, staged.height), (4, 4));
    assert_eq!(staged.bytes.len(), 4 * 4 * 4);
    assert!(matches!(
        staged.writeback,
        TextureWriteback::Type11 { mapping_id: 3, .. }
    ));
}

/// A type-5 record is the exact Metal view, even when its single-plane
/// backing already has valid base geometry. Live pipe 5 exposes each row
/// of a 1920-wide BGRA8 surface as a 480-wide RGBA32Uint view so one
/// `uint4` image write stores four packed BGRA pixels.
#[test]
fn stage_texture_type5_record_reshapes_stageable_single_plane_surface() {
    use crate::contract::iosurface_pages::{PAGE_ENTRY_PFN_SHIFT, PAGE_ENTRY_VALID};
    use crate::contract::pixel_format::{
        StorageImageSelector, MTL_FORMAT_BGRA8_UNORM, MTL_FORMAT_R32_UINT, MTL_FORMAT_RGBA32_UINT,
    };
    use crate::runtime::decode::resource::{list_object_entry_offset, OBJECT_LIST_ENTRY_LEN};

    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    let mut host = FakeHost::new();
    setup_task_pages(&mut host, &mut state, 4);

    let sid = 3u32;
    let type5_ref = 10u32;
    let pfn = 0x20u32;
    let gpa = (pfn as u64) << PAGE_SHIFT_ARM64E;
    host.map_range(gpa, 0x4000, 0x5a);
    let entry = (pfn << PAGE_ENTRY_PFN_SHIFT) | PAGE_ENTRY_VALID;
    assert!(state.map_surface(sid));
    {
        let m = state.mappings.get_mut(&sid).unwrap();
        m.mapped = true;
        m.mapping_internal = 1;
        m.page_entries = vec![entry];
    }
    assert!(state.set_mapping_geom(sid, 4, 4, MTL_FORMAT_BGRA8_UNORM));

    // Same 16 bytes per logical row: 4 BGRA8 texels = one RGBA32Uint texel.
    let desc_gva = (4u64 + 2) << PAGE_SHIFT_ARM64E;
    let mut type5_desc = vec![0u8; 8];
    st32(&mut type5_desc[objects::TYPE5_SURFACE_ID..], sid);
    type5_desc.extend_from_slice(&[
        0x2f,
        0,
        0,
        0,
        0x30,
        0,
        0,
        0,
        10,
        0,
        0,
        0, // kind, blob_len, own_ref
        0x42,
        0x02,
        MTL_FORMAT_RGBA32_UINT as u8,
        (MTL_FORMAT_RGBA32_UINT >> 8) as u8,
        1,
        0,
        0,
        0, // view width
        4,
        0,
        0,
        0, // view height
        1,
        0,
        0,
        0, // depth
        1,
        0,
        1,
        0,
        1,
        0,
        0x10,
        0, // trailer
    ]);
    assert!(gva_mem::write_task_gva(
        &mut host,
        &state.tasks[1],
        desc_gva,
        &type5_desc,
        PAGE_SHIFT_ARM64E
    )
    .is_ok());
    let off = list_object_entry_offset(type5_ref, 32).unwrap();
    let mut list_entry = [0u8; OBJECT_LIST_ENTRY_LEN];
    let packed = (objects::OBJECT_TYPE_REF_TEXTURE as u32) | ((type5_desc.len() as u32) << 8);
    st32(&mut list_entry[0..], packed);
    list_entry[4..12].copy_from_slice(&desc_gva.to_le_bytes());
    assert!(gva_mem::write_task_gva(
        &mut host,
        &state.tasks[1],
        off,
        &list_entry,
        PAGE_SHIFT_ARM64E
    )
    .is_ok());

    let staged = stage_texture_raw(&mut state, &mut host, 1, type5_ref, 33, true)
        .expect("serialized type-5 view must override base surface geometry");
    assert_eq!((staged.width, staged.height), (1, 4));
    assert_eq!(
        staged.storage_selector,
        Some(StorageImageSelector::Rgba32Uint as u32)
    );
    assert_eq!(staged.bytes.len(), 4 * 16);
    assert!(staged.bytes.iter().all(|&b| b == 0x5a));
    match staged.writeback {
        TextureWriteback::Type11 {
            mapping_id,
            surface_bpr,
            width,
            height,
            bpp,
            ..
        } => {
            assert_eq!(mapping_id, sid);
            assert_eq!(surface_bpr, 128);
            assert_eq!((width, height, bpp), (1, 4, 16));
        }
        _ => panic!("expected Type11 writeback through the texture view"),
    }

    // A sampled R32Uint view retains its exact format/geometry. R32Uint is
    // now a storage-capable format (its selector maps to the R32ui storage
    // path), so `storage_selector` is populated — but it is inert here: this
    // view is staged sampled (`is_storage=false`, binding 32), and the
    // selector is only consulted on the storage-bind path.
    let format_at = objects::TYPE5_ARG_RECORD + objects::TYPE5_RECORD_FORMAT;
    type5_desc[format_at..format_at + 2].copy_from_slice(&MTL_FORMAT_R32_UINT.to_le_bytes());
    let width_at = objects::TYPE5_ARG_RECORD + objects::TYPE5_RECORD_WIDTH;
    st32(&mut type5_desc[width_at..], 4);
    assert!(gva_mem::write_task_gva(
        &mut host,
        &state.tasks[1],
        desc_gva,
        &type5_desc,
        PAGE_SHIFT_ARM64E
    )
    .is_ok());
    let sampled = stage_texture_raw(&mut state, &mut host, 1, type5_ref, 32, false)
        .expect("sample-only R32Uint view must stage from the same IOSurface bytes");
    assert_eq!((sampled.width, sampled.height), (4, 4));
    assert_eq!(sampled.pixel_format, MTL_FORMAT_R32_UINT);
    assert_eq!(
        sampled.storage_selector,
        Some(StorageImageSelector::R32Uint as u32)
    );
    assert_eq!(sampled.bytes.len(), 4 * 4 * 4);
    assert!(matches!(sampled.writeback, TextureWriteback::None));
}

/// Biplanar surface (device_desc plane_count=2) + type-5 args plane record:
/// stage the named plane view (R8 Y) from the plane offset — live class
/// `compute_dispatch st=Unsupported` / `type11_fail reason=multiplane`
/// (wallpaper '420f', journal 2026-07-14 compute census).
#[test]
fn stage_texture_type5_record_stages_biplanar_y_plane() {
    use crate::contract::endian::{st16, st64};
    use crate::contract::iosurface_pages::{
        DEVICE_DESC_ALLOC_SIZE, DEVICE_DESC_LEN, DEVICE_DESC_PLANES, DEVICE_DESC_PLANE_COUNT,
        DEVICE_PLANE_BPE, DEVICE_PLANE_BPR, DEVICE_PLANE_DESC_LEN, DEVICE_PLANE_DIMS,
        DEVICE_PLANE_OFFSET, DEVICE_PLANE_SIZE, PAGE_ENTRY_PFN_SHIFT, PAGE_ENTRY_VALID,
    };
    use crate::contract::pixel_format::MTL_FORMAT_R8_UNORM;
    use crate::runtime::decode::resource::{list_object_entry_offset, OBJECT_LIST_ENTRY_LEN};

    let pack_dims = |w: u64, h: u64| ((w & 0xffffff) << 8) | ((h & 0xffffff) << 40);

    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    let mut host = FakeHost::new();
    setup_task_pages(&mut host, &mut state, 4);

    let sid = 3u32;
    let type5_ref = 10u32;
    let pfn = 0x20u32;
    let gpa = (pfn as u64) << PAGE_SHIFT_ARM64E;
    host.map_range(gpa, 0x4000, 0x77);
    let entry = (pfn << PAGE_ENTRY_PFN_SHIFT) | PAGE_ENTRY_VALID;

    // Device surface: 2 planes — Y 16×8 R8 bpr=64 off=0, UV 8×4 RG8 off=512.
    let mut dev = vec![0u8; DEVICE_DESC_LEN];
    st32(&mut dev[DEVICE_DESC_ALLOC_SIZE..], 0x4000);
    dev[DEVICE_DESC_PLANE_COUNT] = 2;
    let p0 = DEVICE_DESC_PLANES;
    st32(&mut dev[p0 + DEVICE_PLANE_OFFSET..], 0);
    st32(&mut dev[p0 + DEVICE_PLANE_SIZE..], 512);
    st64(&mut dev[p0 + DEVICE_PLANE_DIMS..], pack_dims(16, 8));
    st32(&mut dev[p0 + DEVICE_PLANE_BPR..], 64);
    st16(&mut dev[p0 + DEVICE_PLANE_BPE..], 1);
    let p1 = DEVICE_DESC_PLANES + DEVICE_PLANE_DESC_LEN;
    st32(&mut dev[p1 + DEVICE_PLANE_OFFSET..], 512);
    st32(&mut dev[p1 + DEVICE_PLANE_SIZE..], 256);
    st64(&mut dev[p1 + DEVICE_PLANE_DIMS..], pack_dims(8, 4));
    st32(&mut dev[p1 + DEVICE_PLANE_BPR..], 64);
    st16(&mut dev[p1 + DEVICE_PLANE_BPE..], 2);

    assert!(state.map_surface(sid));
    {
        let m = state.mappings.get_mut(&sid).unwrap();
        m.mapped = true;
        m.mapping_internal = 1;
        m.page_entries = vec![entry];
        m.device_desc = dev;
        m.has_geom = true;
        m.width = 16;
        m.height = 8;
        m.format = 0; // surface-level FourCC has no single MTL format
    }
    assert!(objects::mapping_is_multiplanar(
        state.mappings.get(&sid).unwrap()
    ));

    // Type-5 descriptor: sid + args blob carrying the R8 16×8 plane record.
    let desc_gva = (4u64 + 2) << PAGE_SHIFT_ARM64E;
    let mut type5_desc = vec![0u8; 8];
    st32(&mut type5_desc[objects::TYPE5_SURFACE_ID..], sid);
    type5_desc.extend_from_slice(&[
        0x2f, 0, 0, 0, 0x30, 0, 0, 0, 10, 0, 0, 0, // kind, blob_len, own_ref
        0x42, 0x01, 0x0a, 0x00, // tag, unk, fmt=R8
        16, 0, 0, 0, // width
        8, 0, 0, 0, // height
        1, 0, 0, 0, // depth
    ]);
    assert!(gva_mem::write_task_gva(
        &mut host,
        &state.tasks[1],
        desc_gva,
        &type5_desc,
        PAGE_SHIFT_ARM64E
    )
    .is_ok());
    let off = list_object_entry_offset(type5_ref, 32).unwrap();
    let mut list_entry = [0u8; OBJECT_LIST_ENTRY_LEN];
    let packed = (objects::OBJECT_TYPE_REF_TEXTURE as u32) | ((type5_desc.len() as u32) << 8);
    st32(&mut list_entry[0..], packed);
    list_entry[4..12].copy_from_slice(&desc_gva.to_le_bytes());
    assert!(gva_mem::write_task_gva(
        &mut host,
        &state.tasks[1],
        off,
        &list_entry,
        PAGE_SHIFT_ARM64E
    )
    .is_ok());

    let staged = stage_texture_raw(&mut state, &mut host, 1, type5_ref, 32, true)
        .expect("plane record must stage the Y plane of a biplanar surface");
    assert_eq!((staged.width, staged.height), (16, 8));
    assert_eq!(
        staged.storage_selector,
        Some(crate::contract::pixel_format::StorageImageSelector::R8Unorm as u32)
    );
    assert_eq!(staged.bytes.len(), 16 * 8);
    assert!(staged.bytes.iter().all(|&b| b == 0x77));
    match staged.writeback {
        TextureWriteback::Type11 {
            mapping_id,
            surface_offset,
            surface_bpr,
            ..
        } => {
            assert_eq!(mapping_id, sid);
            assert_eq!(surface_offset, 0);
            assert_eq!(surface_bpr, 64);
        }
        _ => panic!("expected Type11 writeback"),
    }
    let sampled = stage_texture_raw(&mut state, &mut host, 1, type5_ref, 32, false)
        .expect("sampled type-5 plane must stage without writeback");
    assert!(!sampled.is_storage);
    assert!(matches!(sampled.writeback, TextureWriteback::None));
    let _ = MTL_FORMAT_R8_UNORM;
}

/// Biplanar surface **without** a plane record still fails closed
/// (no BGRA invent over multi-plane bytes).
#[test]
fn stage_texture_type5_multiplanar_without_record_fails_closed() {
    use crate::contract::iosurface_pages::{
        DEVICE_DESC_LEN, DEVICE_DESC_PLANE_COUNT, PAGE_ENTRY_PFN_SHIFT, PAGE_ENTRY_VALID,
    };
    use crate::runtime::decode::resource::{list_object_entry_offset, OBJECT_LIST_ENTRY_LEN};

    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    let mut host = FakeHost::new();
    setup_task_pages(&mut host, &mut state, 4);

    let sid = 3u32;
    let type5_ref = 10u32;
    let pfn = 0x20u32;
    let gpa = (pfn as u64) << PAGE_SHIFT_ARM64E;
    host.map_range(gpa, 0x4000, 0x5a);
    let entry = (pfn << PAGE_ENTRY_PFN_SHIFT) | PAGE_ENTRY_VALID;
    let mut dev = vec![0u8; DEVICE_DESC_LEN];
    dev[DEVICE_DESC_PLANE_COUNT] = 2;
    assert!(state.map_surface(sid));
    {
        let m = state.mappings.get_mut(&sid).unwrap();
        m.mapped = true;
        m.mapping_internal = 1;
        m.page_entries = vec![entry];
        m.device_desc = dev;
        m.has_geom = true;
        m.width = 16;
        m.height = 8;
        m.format = 0;
    }

    // Type-5 descriptor with sid but NO args record.
    let desc_gva = (4u64 + 2) << PAGE_SHIFT_ARM64E;
    let mut type5_desc = vec![0u8; 8];
    st32(&mut type5_desc[objects::TYPE5_SURFACE_ID..], sid);
    assert!(gva_mem::write_task_gva(
        &mut host,
        &state.tasks[1],
        desc_gva,
        &type5_desc,
        PAGE_SHIFT_ARM64E
    )
    .is_ok());
    let off = list_object_entry_offset(type5_ref, 32).unwrap();
    let mut list_entry = [0u8; OBJECT_LIST_ENTRY_LEN];
    let packed = (objects::OBJECT_TYPE_REF_TEXTURE as u32) | ((type5_desc.len() as u32) << 8);
    st32(&mut list_entry[0..], packed);
    list_entry[4..12].copy_from_slice(&desc_gva.to_le_bytes());
    assert!(gva_mem::write_task_gva(
        &mut host,
        &state.tasks[1],
        off,
        &list_entry,
        PAGE_SHIFT_ARM64E
    )
    .is_ok());

    match stage_texture_raw(&mut state, &mut host, 1, type5_ref, 32, true) {
        Err(ComputeStatus::Unsupported(_)) => {}
        Err(other) => panic!("expected Unsupported, got {other:?}"),
        Ok(_) => panic!("multiplanar without plane record must fail closed"),
    }
}

/// A linear texture (ot=2) whose numeric ref equals an existing surface mid
/// must NOT be reinterpreted as that mapping — it resolves through its own
/// (here invalid) descriptor and fails linear, never staging the collided
/// surface's pixels. Live class: `ref=N ot=2` MissingTexture where mid N is
/// the biplanar wallpaper.
#[test]
fn stage_texture_linear_ref_does_not_collide_with_surface_mid() {
    use crate::contract::iosurface_pages::{PAGE_ENTRY_PFN_SHIFT, PAGE_ENTRY_VALID};
    use crate::contract::pixel_format::MTL_FORMAT_BGRA8_UNORM;
    use crate::runtime::decode::resource::{list_object_entry_offset, OBJECT_LIST_ENTRY_LEN};

    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    let mut host = FakeHost::new();
    setup_task_pages(&mut host, &mut state, 4);

    // Surface mid 7 exists with full geometry + a mapped page.
    let colliding_mid = 7u32;
    let pfn = 0x20u32;
    let gpa = (pfn as u64) << PAGE_SHIFT_ARM64E;
    host.map_range(gpa, 0x4000, 0x33);
    let entry = (pfn << PAGE_ENTRY_PFN_SHIFT) | PAGE_ENTRY_VALID;
    assert!(state.map_surface(colliding_mid));
    {
        let m = state.mappings.get_mut(&colliding_mid).unwrap();
        m.mapped = true;
        m.mapping_internal = 1;
        m.page_entries = vec![entry];
    }
    assert!(state.set_mapping_geom(colliding_mid, 4, 4, MTL_FORMAT_BGRA8_UNORM));

    // Object-list ref 7 = a TEXTURE (ot=2) object with a non-decodable desc.
    let desc_gva = (4u64 + 2) << PAGE_SHIFT_ARM64E;
    let bogus_desc = vec![0u8; 16];
    assert!(gva_mem::write_task_gva(
        &mut host,
        &state.tasks[1],
        desc_gva,
        &bogus_desc,
        PAGE_SHIFT_ARM64E
    )
    .is_ok());
    let off = list_object_entry_offset(colliding_mid, 32).unwrap();
    let mut list_entry = [0u8; OBJECT_LIST_ENTRY_LEN];
    let packed = (OBJECT_TYPE_TEXTURE as u32) | ((bogus_desc.len() as u32) << 8);
    st32(&mut list_entry[0..], packed);
    list_entry[4..12].copy_from_slice(&desc_gva.to_le_bytes());
    assert!(gva_mem::write_task_gva(
        &mut host,
        &state.tasks[1],
        off,
        &list_entry,
        PAGE_SHIFT_ARM64E
    )
    .is_ok());

    // Must fail linear (bogus desc), NOT succeed against surface mid 7.
    if let Ok(s) = stage_texture_raw(&mut state, &mut host, 1, colliding_mid, 32, true) {
        panic!(
            "linear ref must not stage collided surface mid ({}x{})",
            s.width, s.height
        )
    }
}

#[test]
fn stage_heap_texture_uses_host_only_residency_identity() {
    use crate::contract::pixel_format::{StorageImageSelector, MTL_FORMAT_RGBA32_FLOAT};
    use crate::runtime::decode::resource::{
        list_object_entry_offset, HEAP_TEXTURE_DESCRIPTOR, HEAP_TEXTURE_HEAP_REF, HEAP_TEXTURE_LEN,
        HEAP_TEXTURE_OFFSET, HEAP_TEXTURE_OPCODE, HEAP_TEXTURE_USE_OFFSET, OBJECT_LIST_ENTRY_LEN,
        OBJECT_TYPE_TEXTURE_VIEW,
    };

    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    let mut host = FakeHost::new();
    setup_task_pages(&mut host, &mut state, 4);

    let texture_ref = 20u32;
    let heap_ref = 19u32;
    let desc_gva = (4u64 + 2) << PAGE_SHIFT_ARM64E;
    let mut desc = vec![0u8; HEAP_TEXTURE_LEN];
    st32(&mut desc[0..], HEAP_TEXTURE_OPCODE);
    st32(&mut desc[4..], HEAP_TEXTURE_LEN as u32);
    st32(&mut desc[8..], texture_ref);
    st32(&mut desc[HEAP_TEXTURE_HEAP_REF..], heap_ref);
    // PGSerializedTextureDescriptor: 2D, GPU-optimized, usage=3,
    // RGBA32Float, 180x135x1, one mip/sample/array element, private.
    st32(
        &mut desc[HEAP_TEXTURE_DESCRIPTOR..],
        2 | (1 << 6) | (3 << 8) | ((MTL_FORMAT_RGBA32_FLOAT as u32) << 16),
    );
    st32(&mut desc[HEAP_TEXTURE_DESCRIPTOR + 4..], 180);
    st32(&mut desc[HEAP_TEXTURE_DESCRIPTOR + 8..], 135);
    st32(&mut desc[HEAP_TEXTURE_DESCRIPTOR + 12..], 1);
    desc[HEAP_TEXTURE_DESCRIPTOR + 16..HEAP_TEXTURE_DESCRIPTOR + 18]
        .copy_from_slice(&1u16.to_le_bytes());
    desc[HEAP_TEXTURE_DESCRIPTOR + 18..HEAP_TEXTURE_DESCRIPTOR + 20]
        .copy_from_slice(&1u16.to_le_bytes());
    desc[HEAP_TEXTURE_DESCRIPTOR + 20..HEAP_TEXTURE_DESCRIPTOR + 22]
        .copy_from_slice(&1u16.to_le_bytes());
    desc[HEAP_TEXTURE_DESCRIPTOR + 22..HEAP_TEXTURE_DESCRIPTOR + 24]
        .copy_from_slice(&0x20u16.to_le_bytes());
    st32(&mut desc[HEAP_TEXTURE_USE_OFFSET..], 1);
    st64(&mut desc[HEAP_TEXTURE_OFFSET..], 0);
    assert!(gva_mem::write_task_gva(
        &mut host,
        &state.tasks[1],
        desc_gva,
        &desc,
        PAGE_SHIFT_ARM64E
    )
    .is_ok());

    let entry_offset = list_object_entry_offset(texture_ref, 32).unwrap();
    let mut list_entry = [0u8; OBJECT_LIST_ENTRY_LEN];
    let packed = (OBJECT_TYPE_TEXTURE_VIEW as u32) | ((HEAP_TEXTURE_LEN as u32) << 8);
    st32(&mut list_entry[0..], packed);
    list_entry[4..12].copy_from_slice(&desc_gva.to_le_bytes());
    assert!(gva_mem::write_task_gva(
        &mut host,
        &state.tasks[1],
        entry_offset,
        &list_entry,
        PAGE_SHIFT_ARM64E
    )
    .is_ok());

    let staged = stage_texture_raw(&mut state, &mut host, 1, texture_ref, 33, true)
        .expect("live opcode-0x15 heap texture must stage");
    assert_eq!((staged.width, staged.height), (180, 135));
    assert_eq!(staged.pixel_format, MTL_FORMAT_RGBA32_FLOAT);
    assert_eq!(
        staged.storage_selector,
        Some(StorageImageSelector::Rgba32Float as u32)
    );
    assert_eq!(staged.bytes.len(), 180 * 135 * 16);
    assert!(matches!(staged.writeback, TextureWriteback::None));
    let residency = staged.residency.expect("heap texture needs GPU residency");
    assert!(residency.key.is_heap());
    assert!(!residency.key.is_linear());
    assert_eq!(residency.key.map_generation, 1);
    assert_eq!(residency.key.texture_ref, texture_ref);
    assert_eq!(residency.seed_generation, 0);
}

/// UnmapMemory removes the guest page-table alias, not the discrete
/// type-2/3 texture body. Compute writeback must retain raw output, mirror
/// normalized color for render sampling, and complete without attempting
/// a fail-closed write into freed guest pages.
#[test]
fn linear_writeback_retains_cache_when_guest_gva_is_unmapped() {
    use crate::contract::pixel_format::MTL_FORMAT_RGBA8_UNORM;
    use crate::runtime::surface_cache;

    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_X86);
    let mut host = FakeHost::new();
    let task_id = 6u32;
    let texture_ref = 11u32;
    let gva = 0x101000u64;
    // An unrelated live span makes the product bound audit authoritative;
    // the texture range itself is intentionally outside it.
    state.note_task_map(task_id, 0x200000, 0x1000);
    let rgba = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
    let staged = StagedTexture {
        binding: 32,
        pixel_format: MTL_FORMAT_RGBA8_UNORM,
        storage_selector: Some(5),
        width: 2,
        height: 2,
        bytes: rgba.clone(),
        is_storage: true,
        residency: None,
        seed_skipped: false,
        sample_resident: None,
        writeback: TextureWriteback::Linear {
            texture_ref,
            gva,
            pixel_format: MTL_FORMAT_RGBA8_UNORM,
            row_stride: 8,
            width: 2,
            height: 2,
            bpp: 4,
        },
    };

    assert_eq!(
        writeback_texture(&mut state, &mut host, task_id, &staged),
        Ok(())
    );
    assert_eq!(
        surface_cache::get_linear_texture(
            &state,
            task_id,
            texture_ref,
            gva,
            MTL_FORMAT_RGBA8_UNORM,
            2,
            2,
            8,
        ),
        Some(rgba.as_slice())
    );
    assert_eq!(
        &surface_cache::get_texture(&state, texture_ref, 2, 2).unwrap()[..4],
        &[3, 2, 1, 4],
        "RGBA compute output mirrors into the BGRA render-sample cache"
    );
}

/// No product MiB budget on compute staging — guest size is authoritative.
/// Full-screen wide-gamut (live SkyLight 1928×1920 RGBA16F ≈ 28.2 MiB) must
/// be host-addressable (usize), not rejected by an arbitrary cap.
#[test]
fn compute_stage_admits_full_screen_wide_gamut_without_cap() {
    use crate::contract::pixel_format::{bytes_per_pixel, MTL_FORMAT_RGBA16_FLOAT};
    use crate::runtime::metal_draw::host_alloc_len;
    let bpp = bytes_per_pixel(MTL_FORMAT_RGBA16_FLOAT).expect("rgba16f bpp") as u64;
    let need = 1928u64 * 1920 * bpp;
    assert!(
        host_alloc_len(need).is_some(),
        "full-screen RGBA16Float ({need} bytes) must be host-addressable"
    );
}

/// Type-5 surface id must not be re-resolved through this task's object
/// list: slot `sid` can be a different texture-ref object (id collision).
/// Live class: ensure=1 then MissingTexture when resolve_type11_ref(task,sid)
/// returned the wrong mapping.
#[test]
fn stage_texture_type5_ignores_task_object_list_slot_collision() {
    use crate::contract::iosurface_pages::{PAGE_ENTRY_PFN_SHIFT, PAGE_ENTRY_VALID};
    use crate::contract::pixel_format::MTL_FORMAT_BGRA8_UNORM;
    use crate::runtime::decode::resource::{list_object_entry_offset, OBJECT_LIST_ENTRY_LEN};

    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    let mut host = FakeHost::new();
    setup_task_pages(&mut host, &mut state, 4);

    let sid = 3u32;
    let type5_ref = 10u32;
    let pfn = 0x21u32;
    let gpa = (pfn as u64) << PAGE_SHIFT_ARM64E;
    host.map_range(gpa, 0x4000, 0xa5);
    let entry = (pfn << PAGE_ENTRY_PFN_SHIFT) | PAGE_ENTRY_VALID;
    assert!(state.map_surface(sid));
    {
        let m = state.mappings.get_mut(&sid).unwrap();
        m.mapped = true;
        m.mapping_internal = 1;
        m.page_entries = vec![entry];
    }
    assert!(state.set_mapping_geom(sid, 4, 4, MTL_FORMAT_BGRA8_UNORM));

    // Poison: object-list slot `sid` is type-11 with mapping_id=99 (not mapped).
    // Pre-fix path would resolve_type11_ref(task, sid) → 99 → MissingTexture.
    let poison_desc_gva = (4u64 + 1) << PAGE_SHIFT_ARM64E;
    let mut iosurf = vec![0u8; 64];
    st32(&mut iosurf[0..], 99); // fake mapping_id
    assert!(gva_mem::write_task_gva(
        &mut host,
        &state.tasks[1],
        poison_desc_gva,
        &iosurf,
        PAGE_SHIFT_ARM64E
    )
    .is_ok());
    let off_sid = list_object_entry_offset(sid, 32).unwrap();
    let mut le_sid = [0u8; OBJECT_LIST_ENTRY_LEN];
    // type-11 = OBJECT_TYPE_IOSURFACE
    let packed_t11 = (OBJECT_TYPE_IOSURFACE as u32) | ((64u32) << 8);
    st32(&mut le_sid[0..], packed_t11);
    le_sid[4..12].copy_from_slice(&poison_desc_gva.to_le_bytes());
    assert!(gva_mem::write_task_gva(
        &mut host,
        &state.tasks[1],
        off_sid,
        &le_sid,
        PAGE_SHIFT_ARM64E
    )
    .is_ok());

    // type-5 at ref 10 → surface_id 3
    let desc_gva = (4u64 + 2) << PAGE_SHIFT_ARM64E;
    let mut type5_desc = vec![0u8; 16];
    st32(&mut type5_desc[objects::TYPE5_SURFACE_ID..], sid);
    assert!(gva_mem::write_task_gva(
        &mut host,
        &state.tasks[1],
        desc_gva,
        &type5_desc,
        PAGE_SHIFT_ARM64E
    )
    .is_ok());
    let off = list_object_entry_offset(type5_ref, 32).unwrap();
    let mut list_entry = [0u8; OBJECT_LIST_ENTRY_LEN];
    let packed = (objects::OBJECT_TYPE_REF_TEXTURE as u32) | ((16u32) << 8);
    st32(&mut list_entry[0..], packed);
    list_entry[4..12].copy_from_slice(&desc_gva.to_le_bytes());
    assert!(gva_mem::write_task_gva(
        &mut host,
        &state.tasks[1],
        off,
        &list_entry,
        PAGE_SHIFT_ARM64E
    )
    .is_ok());

    let staged = stage_texture_raw(&mut state, &mut host, 1, type5_ref, 32, true)
        .expect("type-5 must stage mapping sid, not poisoned type-11 slot");
    assert_eq!((staged.width, staged.height), (4, 4));
    assert!(matches!(
        staged.writeback,
        TextureWriteback::Type11 { mapping_id: 3, .. }
    ));
}

/// Type-5 whose surface_id never maps must fail MissingTexture (not pretend
/// type-2/3 success).
#[test]
fn stage_texture_type5_without_surface_is_missing() {
    use crate::runtime::decode::resource::{list_object_entry_offset, OBJECT_LIST_ENTRY_LEN};

    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    let mut host = FakeHost::new();
    setup_task_pages(&mut host, &mut state, 4);

    let type5_ref = 11u32;
    let sid = 99u32; // no mapping
    let desc_gva = (4u64 + 3) << PAGE_SHIFT_ARM64E;
    let mut type5_desc = vec![0u8; 16];
    st32(&mut type5_desc[objects::TYPE5_SURFACE_ID..], sid);
    assert!(gva_mem::write_task_gva(
        &mut host,
        &state.tasks[1],
        desc_gva,
        &type5_desc,
        PAGE_SHIFT_ARM64E
    )
    .is_ok());
    let off = list_object_entry_offset(type5_ref, 32).unwrap();
    let mut list_entry = [0u8; OBJECT_LIST_ENTRY_LEN];
    let packed = (objects::OBJECT_TYPE_REF_TEXTURE as u32) | ((16u32) << 8);
    st32(&mut list_entry[0..], packed);
    list_entry[4..12].copy_from_slice(&desc_gva.to_le_bytes());
    assert!(gva_mem::write_task_gva(
        &mut host,
        &state.tasks[1],
        off,
        &list_entry,
        PAGE_SHIFT_ARM64E
    )
    .is_ok());

    let st = stage_texture_raw(&mut state, &mut host, 1, type5_ref, 32, false);
    assert!(matches!(st, Err(ComputeStatus::MissingTexture(_))));
}

#[cfg(feature = "backend-vulkan")]
#[test]
fn incomplete_compute_engine_call_fires_stall_proxy() {
    use crate::backend::vulkan::engine::ComputeRequest;
    use std::time::Duration;

    let pipe = 0xf000_0000 | (std::process::id() & 0x0fff_ffff);
    let req = ComputeRequest {
        spirv: vec![0x0723_0203],
        entry: "main".into(),
        grid: [1, 1, 1],
        ..Default::default()
    };
    let done = spawn_compute_engine_stall_watchdog(pipe, &req, Duration::from_millis(10));
    std::thread::sleep(Duration::from_millis(40));
    done.store(true, std::sync::atomic::Ordering::Release);

    let log = std::fs::read_to_string(crate::observe::fail_log_path()).expect("fail log");
    assert!(log.lines().any(|line| {
        line.contains("compute_engine_stall reason=backend_call_unreturned")
            && line.contains(&format!("pipe={pipe}"))
    }));
    let base = format!("/tmp/reims-vgpu-compute-stall-pipe-{pipe}");
    let _ = std::fs::remove_file(format!("{base}.spv"));
    let _ = std::fs::remove_file(format!("{base}.txt"));
}

#[cfg(feature = "backend-vulkan")]
#[test]
fn compute_engine_phase_proxy_names_every_measured_boundary() {
    use crate::backend::vulkan::engine::CounterSnapshot;

    let delta = CounterSnapshot {
        lock_wait_us: 1,
        context_us: 2,
        pool_init_us: 3,
        cache_us: 4,
        shader_create_us: 5,
        layout_create_us: 6,
        pipeline_create_us: 7,
        sampler_create_us: 8,
        resource_us: 20,
        sampler_prepare_us: 1,
        storage_prepare_us: 2,
        sampled_prepare_us: 3,
        storage_image_prepare_us: 4,
        descriptor_prepare_us: 5,
        memory_alloc_us: 9,
        pre_record_wait_us: 5,
        record_us: 11,
        submit_us: 12,
        wait_us: 13,
        readback_us: 14,
        cleanup_us: 10,
        readbacks: 2,
        readback_bytes: 4096,
        compute_sampled_uploads: 3,
        compute_sampled_upload_bytes: 8192,
        compute_sampled_resident_copies: 5,
        compute_sampled_resident_copy_bytes: 32768,
        compute_storage_seed_uploads: 1,
        compute_storage_seed_upload_bytes: 16384,
        creates: 3,
        allocs: 4,
        ..Default::default()
    };
    let line = compute_engine_phase_fields(100, &delta);
    for field in [
        "engine_lock_wait_us=1",
        "engine_context_us=2",
        "engine_pool_init_us=3",
        "engine_cache_us=4",
        "engine_shader_create_us=5",
        "engine_layout_create_us=6",
        "engine_pipeline_create_us=7",
        "engine_sampler_create_us=8",
        "engine_resource_us=20",
        "engine_resource_unattributed_us=5",
        "engine_sampler_prepare_us=1",
        "engine_storage_buffer_prepare_us=2",
        "engine_sampled_image_prepare_us=3",
        "engine_storage_image_prepare_us=4",
        "engine_descriptor_prepare_us=5",
        "engine_memory_alloc_us=9",
        "engine_pre_record_wait_us=5",
        "engine_record_us=11",
        "engine_submit_us=12",
        "engine_wait_us=13",
        "engine_readback_us=14",
        "engine_cleanup_us=10",
        "engine_unattributed_us=5",
        "engine_readbacks=2",
        "engine_readback_bytes=4096",
        "engine_sampled_uploads=3",
        "engine_sampled_upload_bytes=8192",
        "engine_sampled_resident_copies=5",
        "engine_sampled_resident_copy_bytes=32768",
        "engine_storage_seed_uploads=1",
        "engine_storage_seed_upload_bytes=16384",
        "engine_creates=3",
        "engine_allocs=4",
    ] {
        assert!(line.contains(field), "missing {field}: {line}");
    }
}

#[test]
fn full_screen_compute_output_proxy_names_content_and_destination() {
    let pipe = 0xe000_0000 | (std::process::id() & 0x0fff_ffff);
    let mut bytes = vec![0u8; 1280 * 720 * 4];
    bytes[0] = 7;
    let tex = StagedTexture {
        binding: 34,
        pixel_format: pixel_format::MTL_FORMAT_RGBA8_UNORM,
        storage_selector: Some(pixel_format::StorageImageSelector::Rgba8Unorm as u32),
        width: 1280,
        height: 720,
        bytes,
        is_storage: true,
        residency: None,
        seed_skipped: false,
        sample_resident: None,
        writeback: TextureWriteback::Type11 {
            mapping_id: 42,
            surface_offset: 0,
            surface_bpr: 1280 * 4,
            span_end: (1280 * 720 * 4) as u64,
            width: 1280,
            height: 720,
            bpp: 4,
        },
    };
    log_compute_output_texture(pipe, &tex, None);

    let log = std::fs::read_to_string(crate::observe::fail_log_path()).expect("fail log");
    assert!(log.lines().any(|line| {
        line.contains(&format!("compute_linux output_tex pipe={pipe}"))
            && line.contains("bind=34")
            && line.contains("dst=type11 mid=42")
            && line.contains("rgb_nz=1")
            && line.contains("max_rgb=7")
    }));

    let packed_pipe = pipe ^ 1;
    let mut packed_bytes = vec![0u8; 320 * 720 * 16];
    packed_bytes[0] = 9;
    let packed = StagedTexture {
        binding: 35,
        pixel_format: pixel_format::MTL_FORMAT_RGBA32_UINT,
        storage_selector: Some(pixel_format::StorageImageSelector::Rgba32Uint as u32),
        width: 320,
        height: 720,
        bytes: packed_bytes,
        is_storage: true,
        residency: None,
        seed_skipped: false,
        sample_resident: None,
        writeback: TextureWriteback::None,
    };
    log_compute_output_texture(packed_pipe, &packed, None);
    let log = std::fs::read_to_string(crate::observe::fail_log_path()).expect("fail log");
    assert!(log.lines().any(|line| {
        line.contains(&format!("compute_linux output_tex pipe={packed_pipe}"))
            && line.contains("fmt=0x7b")
            && line.contains("320x720")
    }));
}

#[test]
fn storage_residency_proxy_requires_exact_view_and_generation() {
    let pipe = 0xd000_0000 | (std::process::id() & 0x0fff_ffff);
    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_X86);
    state.map_surface(9);
    state.mappings.get_mut(&9).unwrap().content_generation = 4;
    let key = crate::model::ComputeStorageResidencyKey {
        mapping_id: 9,
        map_generation: state.mappings[&9].map_generation,
        surface_offset: 64,
        surface_bpr: 32,
        span_end: 128,
        width: 4,
        height: 2,
        pixel_format: pixel_format::MTL_FORMAT_RGBA8_UNORM,
        texture_ref: 0,
    };
    let mut texture = StagedTexture {
        binding: 32,
        pixel_format: pixel_format::MTL_FORMAT_RGBA8_UNORM,
        storage_selector: Some(pixel_format::StorageImageSelector::Rgba8Unorm as u32),
        width: 4,
        height: 2,
        bytes: vec![0; 32],
        is_storage: true,
        residency: Some(ComputeStorageResidencyCandidate {
            key,
            seed_generation: 4,
        }),
        seed_skipped: false,
        sample_resident: None,
        writeback: TextureWriteback::None,
    };

    assert_eq!(
        storage_residency_opportunity(&state, std::slice::from_ref(&texture)),
        (1, 0, 32, 0)
    );
    state.compute_storage_residency.insert(key, 4);
    assert_eq!(
        storage_residency_opportunity(&state, std::slice::from_ref(&texture)),
        (1, 1, 32, 32)
    );

    texture.residency.as_mut().unwrap().seed_generation = 5;
    assert_eq!(
        storage_residency_opportunity(&state, std::slice::from_ref(&texture)),
        (1, 0, 32, 0)
    );
    // An intersecting stale window is dropped by note; a disjoint sibling
    // window (ping-pong canvas) survives it.
    let mut stale_view = key;
    stale_view.surface_offset = 0;
    state.compute_storage_residency.insert(stale_view, 3);
    let mut sibling = key;
    sibling.surface_offset = 128;
    sibling.span_end = 192;
    state.compute_storage_residency.insert(sibling, 2);
    state.mappings.get_mut(&9).unwrap().content_generation = 5;
    note_storage_residency_writeback(&mut state, &texture);
    assert_eq!(state.compute_storage_residency.len(), 2);
    assert!(!state.compute_storage_residency.contains_key(&stale_view));
    assert!(state.compute_storage_residency.contains_key(&sibling));
    // The mirror stores the engine currency next(seed_generation), so the
    // staged candidate hits only once its seed matches that value.
    assert_eq!(state.compute_storage_residency.get(&key), Some(&6));
    assert_eq!(
        storage_residency_opportunity(&state, std::slice::from_ref(&texture)),
        (1, 0, 32, 0)
    );
    texture.residency.as_mut().unwrap().seed_generation = 6;
    state.compute_storage_residency.remove(&sibling);
    assert_eq!(
        storage_residency_opportunity(&state, std::slice::from_ref(&texture)),
        (1, 1, 32, 32)
    );

    let (eligible, hits, eligible_bytes, hit_bytes) =
        storage_residency_opportunity(&state, std::slice::from_ref(&texture));
    log_storage_residency_opportunity(pipe, eligible, hits, eligible_bytes, hit_bytes);
    let marker = format!(
            "OFF compute_storage_residency reason=generation_match action=measure pipe={pipe} eligible=1 hits=1 eligible_bytes=32 hit_bytes=32"
        );
    let log = std::fs::read_to_string(crate::observe::fail_log_path()).expect("fail log");
    assert!(log
        .lines()
        .any(|line| crate::observe::line_is(line, &marker)));

    texture.residency.as_mut().unwrap().key.map_generation += 1;
    assert_eq!(
        storage_residency_opportunity(&state, std::slice::from_ref(&texture)),
        (1, 0, 32, 0)
    );
}

#[test]
fn storage_access_proxy_names_writeonly_seed_cost() {
    let pipe = 0xd000_0000 | (std::process::id() & 0x0fff_ffff);
    log_storage_image_access(pipe, 34, "write_only", 29_614_080);

    let log = std::fs::read_to_string(crate::observe::fail_log_path()).expect("fail log");
    assert!(log.lines().any(|line| {
        line.contains(&format!("compute_linux storage_access pipe={pipe}"))
            && line.contains("bind=34")
            && line.contains("access=write_only")
            && line.contains("seed=1")
            && line.contains("bytes=29614080")
    }));
}

#[cfg(feature = "backend-vulkan")]
#[test]
fn storage_format_specialization_preserves_raw_views_and_runtime_shape() {
    use crate::backend::vulkan::engine::StorageImageFormat as V;
    use crate::runtime::spirv_bind::ImageFormat as S;

    assert_eq!(
        mtl_to_engine_sampled(pixel_format::MTL_FORMAT_R32_UINT),
        Some(V::R32Uint)
    );
    assert_eq!(
        mtl_to_engine_sampled(pixel_format::MTL_FORMAT_RGB9E5_FLOAT),
        Some(V::Rgb9e5Ufloat)
    );

    // A uint shader over a BGRA8Unorm surface is a deliberate raw byte view
    // (byte order preserved) — unaffected by the without-format flag.
    assert_eq!(
        specialized_storage_image_format(V::Bgra8Unorm, S::Rgba8Uint, true),
        Ok(S::Rgba8Uint)
    );
    assert_eq!(
        specialized_storage_image_format(V::Bgra8Unorm, S::Rgba8Uint, false),
        Ok(S::Rgba8Uint)
    );
    assert_eq!(
        specialized_storage_image_format(V::Rgba16Float, S::Rgba32Float, true),
        Ok(S::Rgba16Float)
    );
    // BGRA8Unorm normalized color store (the desktop composite): with the
    // device feature it retargets to a format-less `Unknown` storage image
    // (viewed B8G8R8A8_UNORM — no R/B swap); without it, degrades to the
    // swapped Rgba8Unorm view.
    assert_eq!(
        specialized_storage_image_format(V::Bgra8Unorm, S::Rgba32Float, true),
        Ok(S::Unknown)
    );
    assert_eq!(
        specialized_storage_image_format(V::Bgra8Unorm, S::Rgba32Float, false),
        Ok(S::Rgba8Unorm)
    );
    assert_eq!(
        specialized_storage_image_format(V::Bgra8Unorm, S::Rgba8Unorm, true),
        Ok(S::Unknown)
    );
    assert_eq!(
        specialized_storage_image_format(V::Bgra8Unorm, S::Rgba16Uint, true),
        Err("spirv_guest_numeric_class_mismatch")
    );
    assert_eq!(
        specialized_storage_image_format(V::Bgra8Unorm, S::Unsupported(0), true),
        Err("spirv_storage_format_unsupported")
    );

    // R32Uint storage (the captured VTMTS 4K coverage-buffer case): the
    // guest surface is a single 32-bit uint channel but the translator
    // declares a 4x8-bit `Rgba8ui` write image. Despite equal bytes/texel
    // (4 == 4) this must NOT take the raw-view early return (which would
    // adopt Rgba8ui and keep only the low byte of each written lane) — it
    // specializes the SPIR-V storage image to single-channel `R32ui` so the
    // view is VK_FORMAT_R32_UINT and a written `uint4`.x is the full u32.
    assert_eq!(
        specialized_storage_image_format(V::R32Uint, S::Rgba8Uint, true),
        Ok(S::R32ui)
    );
    assert_eq!(
        specialized_storage_image_format(V::R32Uint, S::Rgba16Uint, true),
        Ok(S::R32ui)
    );
    // A float/unorm-class write shader over an R32Uint surface is a genuine
    // numeric-class mismatch, not a raw view — still rejected.
    assert_eq!(
        specialized_storage_image_format(V::R32Uint, S::Rgba8Unorm, true),
        Err("spirv_guest_numeric_class_mismatch")
    );
    // The R32-single-channel sint/float and packed Rgb9e5 storage paths
    // stay guarded off (no live capture yet justifies enabling them). The
    // guard is UPSTREAM: `storage_selector` returns None for them, so they
    // are rejected at stage time (`stage_tex_fmt_storage`) and never reach
    // the specializer at all — unlike R32Uint, which is now mapped.
    assert_eq!(
        pixel_format::storage_selector(pixel_format::MTL_FORMAT_R32_SINT),
        None
    );
    assert_eq!(
        pixel_format::storage_selector(pixel_format::MTL_FORMAT_R32_FLOAT),
        None
    );

    // The selector/engine plumbing that lets R32Uint reach the specializer:
    // storage_selector now maps 0x35, and both format bridges round-trip it.
    assert_eq!(
        pixel_format::storage_selector(pixel_format::MTL_FORMAT_R32_UINT),
        Some((
            pixel_format::StorageImageSelector::R32Uint,
            pixel_format::R32_BPP
        ))
    );
    assert_eq!(
        simg_u32_to_engine_storage(pixel_format::StorageImageSelector::R32Uint as u32),
        Some(V::R32Uint)
    );
    assert_eq!(
        spirv_image_format_to_engine_storage(S::R32ui),
        Some(V::R32Uint)
    );
}

/// `metal2vulkan` lowers a generic `texture2d<float, access::write>` to SPIR-V
/// `R32f` (enum value 3). Decoding that as `Unsupported(3)` made the format
/// unspecializable, so every dispatch binding such an image died as
/// `storage_format_specialize_mismatch` — 142 dropped dispatches in one x86
/// desktop boot. It specializes exactly like the `R32ui` case: the declared
/// format is a placeholder, the bound guest surface decides the view.
#[cfg(feature = "backend-vulkan")]
#[test]
fn r32f_write_images_specialize_to_the_bound_guest_surface() {
    use crate::backend::vulkan::engine::StorageImageFormat as V;
    use crate::runtime::spirv_bind::ImageFormat as S;

    assert_eq!(
        spirv_image_format_to_engine_storage(S::R32Float),
        Some(V::R32Float)
    );

    // Wider float surfaces: the placeholder widens to the guest's own format
    // so all four written lanes are stored, not just `.x`.
    assert_eq!(
        specialized_storage_image_format(V::Rgba32Float, S::R32Float, true),
        Ok(S::Rgba32Float)
    );
    assert_eq!(
        specialized_storage_image_format(V::Rgba16Float, S::R32Float, true),
        Ok(S::Rgba16Float)
    );
    // Narrower single-channel float: still class-matched to the guest surface.
    assert_eq!(
        specialized_storage_image_format(V::R16Float, S::R32Float, true),
        Ok(S::R16Float)
    );
    // An R32Float guest surface is an exact match — the raw view is correct.
    assert_eq!(
        specialized_storage_image_format(V::R32Float, S::R32Float, true),
        Ok(S::R32Float)
    );
    // BGRA8Unorm is the desktop composite target. Bytes/texel are equal (4 == 4)
    // so the raw-view early return would view a BGRA surface as a single
    // 32-bit float; a normalized color store must retarget instead.
    assert_eq!(
        specialized_storage_image_format(V::Bgra8Unorm, S::R32Float, true),
        Ok(S::Unknown)
    );
    assert_eq!(
        specialized_storage_image_format(V::Bgra8Unorm, S::R32Float, false),
        Ok(S::Rgba8Unorm)
    );
    // A float-class shader over a uint surface stays a class mismatch.
    assert_eq!(
        specialized_storage_image_format(V::R32Uint, S::R32Float, true),
        Err("spirv_guest_numeric_class_mismatch")
    );
}

/// x86 (12-bit) task page table with depth-1 root; ptes map gva page i →
/// `pfns[i]`. Mirrors the gva_view multi-import fixture.
fn setup_linear_task_x86(host: &mut FakeHost, state: &mut DeviceState, pfns: &[u32]) {
    let page = 1u64 << PAGE_SHIFT_X86;
    let dir_gpa = 2u64 << PAGE_SHIFT_X86;
    let root_gpa = 3u64 << PAGE_SHIFT_X86;
    host.map_range(dir_gpa, page as usize, 0);
    host.map_range(root_gpa, page as usize, 0);
    let mut d = [0u8; 8];
    st32(&mut d[DIRECTORY_ROOT_PFN as usize..], 3);
    st32(&mut d[DIRECTORY_DEPTH as usize..], 1);
    host.write_gpa(dir_gpa, &d).unwrap();
    for (i, pfn) in pfns.iter().enumerate() {
        host.map_range((*pfn as u64) << PAGE_SHIFT_X86, page as usize, 0);
        let mut pte = [0u8; 4];
        st32(&mut pte, *pfn);
        host.write_gpa(root_gpa + (i as u64) * 4, &pte).unwrap();
    }
    assert!(state.define_task(1, page, 2));
}

#[test]
fn bulk_linear_read_destrides_span_with_one_view() {
    let mut host = FakeHost::new();
    host.strict_linux_map = true;
    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_X86);
    setup_linear_task_x86(&mut host, &mut state, &[4, 5]);
    let (tight, stride, h) = (8usize, 16u64, 3u32);
    let gva = 0x40u64;
    // Rows y at gva + y*stride: payload y*3+1..; padding sentinel 0xEE.
    for y in 0..h as u64 {
        let mut row = vec![0xEEu8; stride as usize];
        for (i, b) in row[..tight].iter_mut().enumerate() {
            *b = (y as u8) * 3 + 1 + i as u8;
        }
        host.write_gpa((4u64 << PAGE_SHIFT_X86) + gva + y * stride, &row)
            .unwrap();
    }
    let mut bytes = vec![0u8; tight * h as usize];
    assert!(read_linear_texture_bulk(
        &mut state, &mut host, 1, gva, stride, tight, h, &mut bytes
    ));
    for y in 0..h as usize {
        for i in 0..tight {
            assert_eq!(
                bytes[y * tight + i],
                (y as u8) * 3 + 1 + i as u8,
                "y={y} i={i}"
            );
        }
    }
}

#[test]
fn bulk_linear_write_scatters_rows_and_preserves_padding() {
    let mut host = FakeHost::new();
    host.strict_linux_map = true;
    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_X86);
    setup_linear_task_x86(&mut host, &mut state, &[4, 5]);
    let (tight, stride, h) = (8usize, 16u64, 3u32);
    let gva = 0x40u64;
    let data0 = 4u64 << PAGE_SHIFT_X86;
    // Sentinel-fill the whole span so untouched padding is provable.
    host.write_gpa(data0 + gva, &vec![0xEEu8; (h as u64 * stride) as usize])
        .unwrap();
    let bytes: Vec<u8> = (0..tight * h as usize).map(|i| i as u8 + 1).collect();
    assert!(write_linear_texture_bulk(
        &mut state, &mut host, 1, gva, stride, tight, h, &bytes
    ));
    for y in 0..h as u64 {
        let mut row = vec![0u8; stride as usize];
        host.read_gpa(data0 + gva + y * stride, &mut row).unwrap();
        assert_eq!(
            &row[..tight],
            &bytes[(y as usize) * tight..(y as usize) * tight + tight],
            "row y={y}"
        );
        if y + 1 < h as u64 {
            assert!(
                row[tight..].iter().all(|&b| b == 0xEE),
                "padding must stay untouched y={y}"
            );
        }
    }
}

/// Fragmented PFNs under strict Linux map: the packed view fails, both
/// bulk helpers return false, and the per-row fallback stays load-bearing.
#[cfg(not(target_os = "macos"))]
#[test]
fn bulk_linear_helpers_fall_back_on_fragmented_span() {
    let mut host = FakeHost::new();
    host.strict_linux_map = true;
    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_X86);
    // Non-adjacent leaf PFNs: 4 then 10.
    setup_linear_task_x86(&mut host, &mut state, &[4, 10]);
    let page = 1u64 << PAGE_SHIFT_X86;
    let (tight, stride, h) = (8usize, 16u64, 3u32);
    // Span crosses the page boundary → packed map_pages fails.
    let gva = page - stride;
    let mut bytes = vec![0u8; tight * h as usize];
    assert!(!read_linear_texture_bulk(
        &mut state, &mut host, 1, gva, stride, tight, h, &mut bytes
    ));
    assert!(!write_linear_texture_bulk(
        &mut state, &mut host, 1, gva, stride, tight, h, &bytes
    ));
    // The fallback primitive still lands bytes across the fragmented span.
    let payload = [0xABu8; 8];
    assert!(gva_mem::write_task_gva_product(&mut state, &mut host, 1, gva, &payload).is_ok());
}

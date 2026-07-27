//! IOSurface mapper capture + page-table / geometry resolve.
//!
//! Capture runs on the iosfc producer MMIO path (guest x19/x21/x22 still hold
//! the directed handoff from `do_host_mapping_gated`). Resolve builds
//! `MappingEntry.page_entries` and geometry from MappingInternal + device
//! descriptor via guest KVA reads ([`HostOps::read_kva`]).

use crate::contract::iosurface_pages::{
    self, build_table_plan, decode_device_surface, decode_mapper_request_entry, guest_kernel_va,
    mapper_request_published_entry_offset, read_internal_desc_ptr, read_mapper_identity,
    read_mapper_internal, sample_window, sample_window_prefer_device, validate_mapper_internal,
    MapperInternalFields, PagesMemory, DEVICE_DESC_LEN, MAPPER_CAPTURE_REG_MAPPER_DEVICE,
    MAPPER_CAPTURE_REG_MAPPING_INTERNAL, MAPPER_CAPTURE_REG_REQUEST_TYPE, MAPPER_REQUEST_ENTRY_LEN,
    MAPPER_REQUEST_MAP, MAPPER_REQUEST_UNMAP,
};
use crate::model::{DeviceState, MapperCapture, MAX_MAPPINGS};
use crate::runtime::decode::resource::OBJECT_LIST_ENTRY_LEN;
use crate::runtime::host::{HostMemory, HostOps, MemError};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum MapperDecline {
    CaptureMapperXregRead(MemError),
    CaptureRequestTypeXregRead(MemError),
    CaptureInternalXregRead(MemError),
    CaptureRequestTypeMismatch,
    CaptureInternalZero,
    CaptureInternalKvaInvalid,
    CaptureMapperKvaInvalid,
    DeviceDescriptorRead(MemError),
}

impl crate::observe::Decline for MapperDecline {
    fn slug(&self) -> &'static str {
        match self {
            Self::CaptureMapperXregRead(_) => "mapper_capture_mapper_xreg_read",
            Self::CaptureRequestTypeXregRead(_) => "mapper_capture_request_type_xreg_read",
            Self::CaptureInternalXregRead(_) => "mapper_capture_internal_xreg_read",
            Self::CaptureRequestTypeMismatch => "mapper_capture_request_type_mismatch",
            Self::CaptureInternalZero => "mapper_capture_internal_zero",
            Self::CaptureInternalKvaInvalid => "mapper_capture_internal_kva_invalid",
            Self::CaptureMapperKvaInvalid => "mapper_capture_mapper_kva_invalid",
            Self::DeviceDescriptorRead(_) => "mapper_device_descriptor_read",
        }
    }

    fn fields(&self) -> Vec<(&'static str, String)> {
        match self {
            Self::CaptureMapperXregRead(error)
            | Self::CaptureRequestTypeXregRead(error)
            | Self::CaptureInternalXregRead(error)
            | Self::DeviceDescriptorRead(error) => vec![(
                "host_reason",
                crate::observe::Decline::slug(error).to_string(),
            )],
            _ => Vec::new(),
        }
    }
}

fn refusal_reason(status: &iosurface_pages::Status) -> &'static str {
    crate::observe::Refusal::refusal(status)
        .expect("an IOSurface contract error must carry a refusal reason")
}

/// Fail-visible, **de-duplicated per `(mapping_id, reason)`**, for the
/// `resolve_mapping_backing` blind spot: a mapped surface whose page-table /
/// geometry resolve fails leaves the mapping silently un-resolved, and every
/// downstream present/Store/sample paints or writes back **black** for it with
/// no log naming why. `resolve_mapping_backing` runs on the per-present `force`
/// path (drain.rs), so a bare `observe::fail` at a failing site would flood.
/// This latch logs each `(mapping_id, reason)` **once** and is cleared for a
/// mapping the moment it resolves ([`clear_resolve_fail`]), so a genuinely
/// broken mapping logs one line, a flapping one re-logs per transition, and a
/// healthy boot fires nothing. Runs on the drain worker (off the QEMU main
/// core). Speculative/not-ready returns (unmapped, dims-not-yet-landed) are
/// **not** routed here — only genuine anomalies for an already-mapped surface.
fn resolve_fail_latch() -> &'static std::sync::Mutex<std::collections::HashSet<(u32, &'static str)>>
{
    use std::collections::HashSet;
    use std::sync::{Mutex, OnceLock};
    static SEEN: OnceLock<Mutex<HashSet<(u32, &'static str)>>> = OnceLock::new();
    SEEN.get_or_init(|| Mutex::new(HashSet::new()))
}

fn note_resolve_fail(mapping_id: u32, reason: &'static str, detail: String) {
    let mut guard = resolve_fail_latch()
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    if guard.insert((mapping_id, reason)) {
        crate::observe::fail(detail);
    }
}

fn note_resolve_keep_cached(mapping_id: u32, reason: &'static str, detail: String) {
    let mut guard = resolve_fail_latch()
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    if guard.insert((mapping_id, reason)) {
        crate::observe::off(detail);
    }
}

/// Fail-visible capture miss, sharing the same per-`(mapping_id, reason)` latch
/// as [`note_resolve_fail`]. `capture_at_producer` runs on the publishing vCPU
/// (iosfc producer MMIO write), not the drain worker, but it fires **once per
/// mapper-ring publish** — a rare setup/map event, never per-frame — so a
/// latched genuine-only line here costs nothing on the hot path. A capture miss
/// for an already-decoded MAP/UNMAP request means the mapping's `MappingInternal`
/// never attaches, and every downstream present/Store for it paints **black**
/// with no reason. Speculative returns (producer==0, ring not ready, a non
/// MAP/UNMAP request type) are **not** routed here. Sharing the latch means a
/// mapping that later resolves cleanly re-arms its capture reasons too
/// ([`clear_resolve_fail`] clears all reasons for the id).
fn note_capture_fail(mapping_id: u32, reason: &'static str, detail: String) {
    note_resolve_fail(mapping_id, reason, detail);
}

/// Re-arm every reason latch for a mapping that just resolved, so a later
/// genuine failure on the same mapping is logged again (catches flapping).
fn clear_resolve_fail(mapping_id: u32) {
    let mut guard = resolve_fail_latch()
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    guard.retain(|(mid, _)| *mid != mapping_id);
}

fn capture_xreg_failed(
    mapping_id: u32,
    producer: u32,
    decline: MapperDecline,
) -> Option<MapperCapture> {
    note_capture_fail(
        mapping_id,
        crate::observe::Decline::slug(&decline),
        crate::observe::Emit::decline("mapper_capture_fail", &decline)
            .field("mapping", mapping_id)
            .field("producer", producer)
            .render(),
    );
    None
}

/// Adapter: mapper internals are KVA; page content GPAs use HostMemory.
struct MapperMem<'a, H: HostMemory + HostOps> {
    host: &'a H,
    last_error: std::cell::Cell<Option<MemError>>,
}

impl<'a, H: HostMemory + HostOps> MapperMem<'a, H> {
    fn new(host: &'a H) -> Self {
        Self {
            host,
            last_error: std::cell::Cell::new(None),
        }
    }

    fn last_error(&self) -> Option<MemError> {
        self.last_error.get()
    }
}

impl<H: HostMemory + HostOps> PagesMemory for MapperMem<'_, H> {
    fn read(&self, address: u64, dst: &mut [u8]) -> bool {
        if guest_kernel_va(address) {
            match self.host.read_kva(address, dst) {
                Ok(()) => true,
                Err(e) => {
                    self.last_error.set(Some(e));
                    false
                }
            }
        } else {
            match self.host.read_gpa(address, dst) {
                Ok(()) => true,
                Err(e) => {
                    self.last_error.set(Some(e));
                    false
                }
            }
        }
    }
    fn is_kernel_va(&self, address: u64) -> bool {
        guest_kernel_va(address)
    }
    fn is_ram_gpa(&self, address: u64) -> bool {
        self.host.is_ram_gpa(address)
    }
}

/// Capture mapper handoff registers while still on the publishing vCPU.
///
/// Call from the iosfc producer MMIO write path before scheduling the drain BH.
pub fn capture_at_producer<H: HostMemory + HostOps>(
    state: &DeviceState,
    host: &H,
    producer: u32,
) -> Option<MapperCapture> {
    if producer == 0 || state.iosfc.ring_base == 0 {
        return None;
    }
    let entry_off = mapper_request_published_entry_offset(producer)?;
    let mut e = [0u8; MAPPER_REQUEST_ENTRY_LEN];
    host.read_gpa(state.iosfc.ring_base + entry_off, &mut e)
        .ok()?;
    let request = decode_mapper_request_entry(&e).ok()?;
    if request.request_type != MAPPER_REQUEST_MAP && request.request_type != MAPPER_REQUEST_UNMAP {
        return None;
    }
    if request.mapping_id as usize >= MAX_MAPPINGS {
        return None;
    }

    // From here the ring entry is a decoded MAP/UNMAP for a valid mapping_id, so
    // any failure below is a genuine capture miss (the handoff registers do not
    // corroborate the request), not the speculative not-ready poll — log it once
    // per (mapping_id, reason). The mapping's MappingInternal never attaches and
    // downstream present/Store paints black otherwise.
    let mid = request.mapping_id;
    let mapper = match host.read_xreg(MAPPER_CAPTURE_REG_MAPPER_DEVICE) {
        Ok(value) => value,
        Err(error) => {
            let decline = MapperDecline::CaptureMapperXregRead(error);
            return capture_xreg_failed(mid, producer, decline);
        }
    };
    let rtype = match host.read_xreg(MAPPER_CAPTURE_REG_REQUEST_TYPE) {
        Ok(value) => value as u32,
        Err(error) => {
            let decline = MapperDecline::CaptureRequestTypeXregRead(error);
            return capture_xreg_failed(mid, producer, decline);
        }
    };
    let internal = match host.read_xreg(MAPPER_CAPTURE_REG_MAPPING_INTERNAL) {
        Ok(value) => value,
        Err(error) => {
            let decline = MapperDecline::CaptureInternalXregRead(error);
            return capture_xreg_failed(mid, producer, decline);
        }
    };
    if rtype != request.request_type {
        let decline = MapperDecline::CaptureRequestTypeMismatch;
        note_capture_fail(
            mid,
            crate::observe::Decline::slug(&decline),
            crate::observe::Emit::decline("mapper_capture_fail", &decline)
                .field("mapping", mid)
                .field("rtype", rtype)
                .field("request_type", request.request_type)
                .render(),
        );
        return None;
    }
    if internal == 0 {
        let decline = MapperDecline::CaptureInternalZero;
        note_capture_fail(
            mid,
            crate::observe::Decline::slug(&decline),
            crate::observe::Emit::decline("mapper_capture_fail", &decline)
                .field("mapping", mid)
                .render(),
        );
        return None;
    }
    if !guest_kernel_va(internal) {
        let decline = MapperDecline::CaptureInternalKvaInvalid;
        note_capture_fail(
            mid,
            crate::observe::Decline::slug(&decline),
            crate::observe::Emit::decline("mapper_capture_fail", &decline)
                .field("mapping", mid)
                .field("internal", format!("{internal:#x}"))
                .render(),
        );
        return None;
    }
    if mapper != 0 && !guest_kernel_va(mapper) {
        let decline = MapperDecline::CaptureMapperKvaInvalid;
        note_capture_fail(
            mid,
            crate::observe::Decline::slug(&decline),
            crate::observe::Emit::decline("mapper_capture_fail", &decline)
                .field("mapping", mid)
                .field("mapper_kva", format!("{mapper:#x}"))
                .render(),
        );
        return None;
    }

    let mem = MapperMem::new(host);
    let fields = match read_mapper_identity(&mem, internal, mapper != 0, mapper) {
        Ok(f) => f,
        Err(status) => {
            let reason = refusal_reason(&status);
            note_capture_fail(
                mid,
                reason,
                crate::observe::Emit::refusal("mapper_capture_fail", &status)
                    .expect("the error arm cannot carry Status::Ok")
                    .field("mapping", mid)
                    .field("internal", format!("{internal:#x}"))
                    .field("mapper_kva", format!("{mapper:#x}"))
                    .render(),
            );
            return None;
        }
    };
    let status = validate_mapper_internal(&mem, mid, &fields);
    if status != iosurface_pages::Status::Ok {
        let reason = refusal_reason(&status);
        note_capture_fail(
            mid,
            reason,
            crate::observe::Emit::refusal("mapper_capture_fail", &status)
                .expect("the non-Ok branch must carry a refusal")
                .field("mapping", mid)
                .field("internal", format!("{internal:#x}"))
                .render(),
        );
        return None;
    }

    Some(MapperCapture {
        producer,
        mapper_device_kva: mapper,
        request_type: rtype,
        mapping_internal: internal,
    })
}

/// Apply a capture to the mapping named by the just-drained ring entry.
pub fn apply_capture(state: &mut DeviceState, cap: &MapperCapture, mapping_id: u32) -> bool {
    if cap.request_type == MAPPER_REQUEST_UNMAP {
        // The surface is gone: eagerly release any deferred present-store window
        // pinning its resident, so the registry LRU can reclaim it. Without this
        // the pin lingers for the guest lifetime (nothing triggers the lazy
        // `map_generation_drift` drop) and the registry soft-exceeds its cap.
        #[cfg(feature = "backend-vulkan")]
        crate::runtime::import_present::drop_render_deferred_windows(state, mapping_id);
        return state.unmap_surface(mapping_id);
    }
    if cap.request_type != MAPPER_REQUEST_MAP {
        return false;
    }
    if cap.mapper_device_kva != 0 {
        state.mapper_device_kva = cap.mapper_device_kva;
    }
    // A MAP that re-backs the slot with a *different* MappingInternal is a new
    // surface (attach_mapping_internal bumps the generation and drops the old
    // geometry). Its old deferred windows now name a stale identity that no
    // access will ever flush — release their pins here too. A re-statement of
    // the same internal keeps its windows (attach early-returns unchanged).
    #[cfg(feature = "backend-vulkan")]
    {
        let internal_changed = state
            .mappings
            .get(&mapping_id)
            .map(|m| m.mapping_internal != cap.mapping_internal)
            .unwrap_or(true);
        if internal_changed {
            crate::runtime::import_present::drop_render_deferred_windows(state, mapping_id);
        }
    }
    state.attach_mapping_internal(mapping_id, cap.mapping_internal)
}

/// Resolve page table + device-descriptor geometry for a mapped slot.
///
/// Safe to call repeatedly; refreshes pages when `mapping_internal` is set.
pub fn resolve_mapping_backing<H: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &H,
    mapping_id: u32,
) -> bool {
    let Some(m) = state.mappings.get(&mapping_id) else {
        return false;
    };
    if !m.mapped || m.mapping_internal == 0 {
        return false;
    }
    let internal = m.mapping_internal;
    let mapper = state.mapper_device_kva;
    let cached_pages = m.page_entries.len();
    let cached_table = m.page_table_kva;
    let had_cached_pages = cached_pages != 0;
    let mem = MapperMem::new(host);

    let fields = match read_mapper_internal(&mem, internal, mapper != 0, mapper) {
        Ok(f) => f,
        Err(status) => {
            let reason = refusal_reason(&status);
            let host_error = mem.last_error();
            let host_reason = host_error
                .map(|error| crate::observe::Decline::slug(&error))
                .unwrap_or("none");
            if had_cached_pages {
                // QEMU can stop exposing a CPU-backed KVA alias after the
                // mapper handoff while the already-validated GPA page plan
                // remains live. Likewise, a transiently unmapped debug-read
                // alias says nothing about the cached guest-physical pages.
                // Both are the normal revalidation fallback, not a decoded
                // guest-command refusal; emitting them once per recycled
                // mapping id made a healthy boot fire dozens of new Phase-5
                // lines. Keep other cached-plan failures visible because they
                // describe malformed identity fields rather than alias
                // availability.
                if matches!(host_error, Some(MemError::NoCpu | MemError::Unmapped)) {
                    return true;
                }
                note_resolve_keep_cached(
                    mapping_id,
                    reason,
                    crate::observe::Emit::refusal("mapper_revalidate_fallback", &status)
                        .expect("the error arm cannot carry Status::Ok")
                        .field("mapping", mapping_id)
                        .field("pages", cached_pages)
                        .field("table", format!("{cached_table:#x}"))
                        .field("internal", format!("{internal:#x}"))
                        .field("mapper_kva", format!("{mapper:#x}"))
                        .field("host_reason", host_reason)
                        .render(),
                );
                return true;
            }
            // A mapped surface (m.mapped, mapping_internal != 0) whose mapper
            // internal KVA is unreadable is a genuine anomaly, not the
            // not-yet-mapped poll — every downstream present/Store for this
            // mapping then paints black with no reason.
            note_resolve_fail(
                mapping_id,
                reason,
                crate::observe::Emit::refusal("mapper_resolve_fail", &status)
                    .expect("the error arm cannot carry Status::Ok")
                    .field("mapping", mapping_id)
                    .field("internal", format!("{internal:#x}"))
                    .field("mapper_kva", format!("{mapper:#x}"))
                    .field("host_reason", host_reason)
                    .render(),
            );
            return false;
        }
    };
    let status = validate_mapper_internal(&mem, mapping_id, &fields);
    if status != iosurface_pages::Status::Ok {
        let reason = refusal_reason(&status);
        note_resolve_fail(
            mapping_id,
            reason,
            crate::observe::Emit::refusal("mapper_resolve_fail", &status)
                .expect("the non-Ok branch must carry a refusal")
                .field("mapping", mapping_id)
                .field("internal", format!("{internal:#x}"))
                .render(),
        );
        return false;
    }

    // Geometry from device descriptor when present; cache full 0x200 for
    // biplanar plane selection (sample_window_prefer_device).
    let mut width = 0u32;
    let mut height = 0u32;
    let mut format = 0u16;
    // Guest page size for *this* device — never a bare arm PAGE_SIZE constant.
    let guest_page = state.page_size();
    let mut min_size = guest_page;
    let mut device_desc: Option<Vec<u8>> = None;
    match read_internal_desc_ptr(&mem, internal) {
        Ok(desc_kva) => {
            let mut desc = [0u8; DEVICE_DESC_LEN];
            if !mem.read(desc_kva, &mut desc) {
                let decline = MapperDecline::DeviceDescriptorRead(
                    mem.last_error().unwrap_or(MemError::Unmapped),
                );
                note_resolve_fail(
                    mapping_id,
                    crate::observe::Decline::slug(&decline),
                    crate::observe::Emit::decline("mapper_device_descriptor_fallback", &decline)
                        .field("mapping", mapping_id)
                        .field("internal", format!("{internal:#x}"))
                        .field("descriptor", format!("{desc_kva:#x}"))
                        .render(),
                );
            } else {
                device_desc = Some(desc.to_vec());
                if let Some(surf) = decode_device_surface(&desc) {
                    if surf.alloc_size as u64 > 0 {
                        min_size = (surf.alloc_size as u64).max(guest_page);
                    }
                    if surf.width > 0 && surf.height > 0 {
                        width = surf.width;
                        height = surf.height;
                        format = (surf.pixel_format & 0xffff) as u16;
                        if let Some((_, _, end, _)) =
                            sample_window_prefer_device(Some(&desc), None, format, width, height)
                        {
                            min_size = min_size.max(end).max(guest_page);
                        } else if let Some((_, _, end)) = sample_window(0, format, width, height) {
                            min_size = min_size.max(end).max(guest_page);
                        }
                    }
                }
            }
        }
        Err(status) => {
            let reason = refusal_reason(&status);
            // A zero descriptor pointer is the documented "not present" state:
            // geometry can come from the texture object. A failed read or a
            // nonzero invalid pointer is a real fallback decision.
            if reason != "iosurface_mapper_device_desc_pointer_zero" {
                note_resolve_fail(
                    mapping_id,
                    reason,
                    crate::observe::Emit::refusal("mapper_device_descriptor_fallback", &status)
                        .expect("the error arm cannot carry Status::Ok")
                        .field("mapping", mapping_id)
                        .field("internal", format!("{internal:#x}"))
                        .render(),
                );
            }
        }
    }

    // Texture-path geom (type-11 object dims) refines span for single-plane; for
    // multi-plane, prefer alloc_size already latched from the device descriptor.
    if let Some(m) = state.mappings.get(&mapping_id) {
        if m.has_geom && m.width > 0 && m.height > 0 {
            width = m.width;
            height = m.height;
            format = if m.format != 0 { m.format } else { format };
            let desc_slice = device_desc.as_deref().or({
                if m.device_desc.len() >= DEVICE_DESC_LEN {
                    Some(m.device_desc.as_slice())
                } else {
                    None
                }
            });
            if let Some((_, _, end, _)) =
                sample_window_prefer_device(desc_slice, None, format, width, height)
            {
                min_size = min_size.max(end).max(guest_page);
            } else if let Some((_, _, end)) = sample_window(0, format, width, height) {
                min_size = min_size.max(end).max(guest_page);
            }
        }
    }

    let plan = match build_table_plan(&mem, mapping_id, &fields, min_size, state.page_shift) {
        Ok(p) => p,
        Err(status) => {
            // Still latch geom / device desc if we decoded them, even without pages yet.
            if let Some(ref d) = device_desc {
                let _ = state.set_mapping_device_desc(mapping_id, d);
            }
            if width > 0 && height > 0 {
                let _ = state.set_mapping_geom(mapping_id, width, height, format);
                // Geometry IS known, yet no page table covers its
                // `min_size` span — the short-page-table → black-tile class
                // (fail-closed Store writeback / sample walk while the geom is
                // set). Distinct from the dims-not-yet-landed poll (width==0),
                // which stays silent as legitimate not-ready control flow.
                let reason = refusal_reason(&status);
                note_resolve_fail(
                    mapping_id,
                    reason,
                    crate::observe::Emit::refusal("mapper_resolve_fail", &status)
                        .expect("the error arm cannot carry Status::Ok")
                        .field("mapping", mapping_id)
                        .field("width", width)
                        .field("height", height)
                        .field("format", format!("{format:#x}"))
                        .field("min_size", min_size)
                        .render(),
                );
            }
            return false;
        }
    };

    let mut retired = None;
    let mut incarnation_changed = false;
    let mut reprieved = false;
    let mut pages_changed = false;
    if let Some(m) = state.mappings.get_mut(&mapping_id) {
        // A condemned slot (trailing DeleteIOSurfaceBacking2, no resolve
        // since) compares against the stashed fingerprint: the same plan is
        // the SAME incarnation — the delete was stale, keep the generation so
        // the resident and deferred windows stay live (black-band class). A
        // different plan is a genuine new incarnation.
        let condemned = m.condemned_entries.take();
        (pages_changed, incarnation_changed, reprieved) =
            plan_adoption_decision(condemned.as_deref(), &m.page_entries, &plan.entries);
        // New page table ⇒ the contiguous view (and any Metal texture aliasing
        // it) describe the old pages; retire them before adopting the plan.
        if m.contig_ptr != 0 && pages_changed {
            retired = Some((m.contig_ptr, m.contig_len));
            m.contig_ptr = 0;
            m.contig_len = 0;
        }
        if pages_changed {
            DeviceState::bump_map_generation(mapping_id, m);
        }
        m.page_entries = plan.entries;
        m.page_table_kva = plan.page_table_kva;
        m.mapping_internal = internal;
        m.mapped = true;
        if let Some(ref d) = device_desc {
            m.device_desc = d.clone();
        }
    }
    if let Some(v) = retired {
        #[cfg(all(feature = "backend-metal", target_os = "macos"))]
        crate::backend::metal::runtime::type11_guest_texture_invalidate(mapping_id);
        state.retired_views.push(v);
    }
    if incarnation_changed {
        // The condemned backing really died and the id now carries a new
        // surface: drop the prior incarnation's deferred windows before any
        // access could flush old content through the new pages.
        crate::runtime::storage_flush::drop_windows(state, mapping_id, "incarnation_changed");
    } else if reprieved {
        // Stale trailing delete on a live incarnation — the exact black-band
        // trigger. Only note when content was actually at stake (an armed
        // deferred window survived); plain reprieves are steady id-recycle
        // control flow.
        let windows = state
            .render_deferred_flush
            .keys()
            .filter(|k| k.mapping_id == mapping_id)
            .count()
            + state
                .compute_deferred_flush
                .keys()
                .filter(|k| k.mapping_id == mapping_id)
                .count();
        if windows > 0 {
            // Condemn dropped the raw-GVA alias index; the pages just
            // re-adopted are the same ones the windows defer-armed on.
            state.index_deferred_alias_pages(mapping_id);
            crate::observe::off(format!(
                "delete_backing_reprieve mapping={mapping_id} windows={windows}"
            ));
            // Wrong-PFN guard on the REPRIEVE path — the blind spot the rewire
            // guard above cannot cover. A reprieve keeps armed deferred windows
            // WITHOUT bumping map_generation (the delete looked stale: the plan
            // still fingerprints the condemned pages). But a guest that FREED the
            // backing and handed the SAME physical pages to another surface — yet
            // has not yet rewired this mapping's GPU page table away from them —
            // fingerprints identical here, so `pages_changed` is false and the
            // rewire guard never runs. The still-armed flush would then DMA into
            // pages another live surface now owns (or recycled userspace heap =
            // the WindowServer malloc free-list corruption class). A detected
            // cross-surface alias is a proven ownership violation, so fail
            // closed: name it, drop deferred writes, and invalidate the page
            // plan. Runs only on reprieve-with-armed-windows (rare), on the drain
            // worker.
            if let Some((gpa, owner)) = first_surface_page_collision(state, mapping_id) {
                let mine_pages = state
                    .mappings
                    .get(&mapping_id)
                    .map(|m| m.page_entries.len())
                    .unwrap_or(0);
                fail_closed_surface_page_collision(
                    state, mapping_id, gpa, owner, mine_pages, "reprieve",
                );
                return false;
            }
        }
    }
    if width > 0 && height > 0 {
        let _ = state.set_mapping_geom(mapping_id, width, height, format);
    }
    // Wrong-PFN rewire-race guard: a freshly adopted page plan must not alias
    // a *different* live surface's guest pages. Two distinct live IOSurface
    // mappings backing the same physical page means one holds a stale/wrong
    // PFN — a pixel writeback through it scribbles the other surface, or (if
    // the page was recycled to userspace) guest heap: the WindowServer malloc
    // free-list corruption class. A detected alias is a proven ownership
    // violation, so fail closed: name it, drop deferred writes, invalidate the
    // adopted plan, and make this resolve fail. Runs only on a genuine rewire
    // (`pages_changed`), on the drain worker.
    if pages_changed {
        if let Some((gpa, owner)) = first_surface_page_collision(state, mapping_id) {
            let mine_pages = state
                .mappings
                .get(&mapping_id)
                .map(|m| m.page_entries.len())
                .unwrap_or(0);
            fail_closed_surface_page_collision(state, mapping_id, gpa, owner, mine_pages, "rewire");
            return false;
        }
    }
    // Resolved: re-arm the fail latch so a later genuine failure (a re-map that
    // goes bad, a corrupted descriptor) is logged again rather than swallowed.
    clear_resolve_fail(mapping_id);
    true
}

/// Detect the wrong-PFN rewire-race corruption vector: the mapping `mapping_id`
/// just adopted a fresh page plan whose page base is also owned by a
/// *different* currently-live surface mapping. Two distinct live IOSurface
/// mappings must never back the same guest physical page; if they do, one holds
/// a stale/wrong PFN and a writeback through it corrupts memory it does not own
/// (see the WindowServer heap-corruption class). Measure-only — never gates a
/// write. Cost O(this_pages + Σ other live pages); called only on a rewire.
fn first_surface_page_collision(state: &DeviceState, mapping_id: u32) -> Option<(u64, u32)> {
    let page_shift = state.page_shift;
    let page = state.page_size();
    let page_base = |gpa: u64| gpa & !(page - 1);
    let m = state.mappings.get(&mapping_id)?;
    if !m.mapped || m.page_entries.is_empty() {
        return None;
    }
    let mine: std::collections::HashSet<u64> = m
        .page_entries
        .iter()
        .filter_map(|&e| crate::contract::iosurface_pages::entry_gpa_shift(e, page_shift))
        .map(page_base)
        .collect();
    if mine.is_empty() {
        return None;
    }
    for (&other_id, other) in &state.mappings {
        if other_id == mapping_id || !other.mapped || other.page_entries.is_empty() {
            continue;
        }
        for &e in &other.page_entries {
            if let Some(gpa) = crate::contract::iosurface_pages::entry_gpa_shift(e, page_shift) {
                if mine.contains(&page_base(gpa)) {
                    return Some((page_base(gpa), other_id));
                }
            }
        }
    }
    None
}

/// Always-on, deduped-per-`(mid, owner, gpa)` fail line for a cross-surface
/// page alias. Off-main-core (drain worker resolve path). Fires zero on a
/// healthy boot (distinct live surfaces never share a physical page).
fn note_surface_page_collision(
    mapping_id: u32,
    gpa: u64,
    owner: u32,
    mine_pages: usize,
    path: &str,
) {
    use std::collections::HashSet;
    use std::sync::{Mutex, OnceLock};
    static SEEN: OnceLock<Mutex<HashSet<(u32, u32, u64)>>> = OnceLock::new();
    let seen = SEEN.get_or_init(|| Mutex::new(HashSet::new()));
    if seen
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .insert((mapping_id, owner, gpa))
    {
        crate::observe::fail(format!(
            "mapping_pages fail reason=surface_page_collision path={path} mid={mapping_id} \
             owner={owner} gpa={gpa:#x} mine_pages={mine_pages}"
        ));
    }
}

/// A cross-surface page alias is a structural ownership violation: any
/// host-authored pixel write through `mapping_id` can scribble another live
/// surface or, for the recycled-heap variant, a userspace heap page. Keep the
/// always-on forensic line, then fail closed by dropping deferred windows and
/// invalidating the adopted page plan so the next writer must re-resolve instead
/// of writing through known-bad pages.
fn fail_closed_surface_page_collision(
    state: &mut DeviceState,
    mapping_id: u32,
    gpa: u64,
    owner: u32,
    mine_pages: usize,
    path: &str,
) {
    note_surface_page_collision(mapping_id, gpa, owner, mine_pages, path);
    crate::runtime::storage_flush::drop_windows(state, mapping_id, "surface_page_collision");
    let _ = state.invalidate_mapping_pages(mapping_id);
}

/// Incarnation decision when adopting a freshly resolved page plan into a
/// mapping slot. `condemned` is the fingerprint a trailing
/// `DeleteIOSurfaceBacking2` stashed (None when the slot is not condemned).
/// Returns `(pages_changed, incarnation_changed, reprieved)`:
/// `incarnation_changed` = the condemned backing really died and the id now
/// carries different pages (drop the old windows); `reprieved` = the delete
/// was stale — the plan matches the fingerprint, the same incarnation lives
/// on (keep generation, resident, deferred windows).
pub(crate) fn plan_adoption_decision(
    condemned: Option<&[u32]>,
    current: &[u32],
    plan: &[u32],
) -> (bool, bool, bool) {
    let pages_changed = match condemned {
        Some(old) => old != plan,
        None => current != plan,
    };
    (
        pages_changed,
        condemned.is_some() && pages_changed,
        condemned.is_some() && !pages_changed,
    )
}

/// True when the cached page table covers the type-11 sample/write span for
/// the latched geom (archive table build uses the same min_size).
///
/// Early resolve often runs before type-11 object dims land (`min_size` =
/// PAGE_SIZE only). Leaving a short `page_entries` while `has_geom` is true
/// makes Store writeback and sample page walks fail-closed on tiles (Favourites
/// 249² with ~16 pages vs ~63 required) while the Metal attachment still holds
/// content. Re-resolve when the span no longer fits.
pub fn pages_cover_geom(state: &DeviceState, mapping_id: u32) -> bool {
    let Some(m) = state.mappings.get(&mapping_id) else {
        return false;
    };
    if m.page_entries.is_empty() {
        return false;
    }
    if !m.has_geom || m.width == 0 || m.height == 0 {
        // No geom yet — any non-empty table is acceptable until dims latch.
        return true;
    }
    let format = if m.format != 0 {
        m.format
    } else {
        // Match scanout/writeback default when format not latched.
        crate::contract::pixel_format::MTL_FORMAT_BGRA8_UNORM
    };
    let span_end = if let Some((_, _, end, _)) = sample_window_prefer_device(
        if m.device_desc.len() >= DEVICE_DESC_LEN {
            Some(m.device_desc.as_slice())
        } else {
            None
        },
        None,
        format,
        m.width,
        m.height,
    ) {
        end
    } else if let Some((_, _, end)) = sample_window(0, format, m.width, m.height) {
        end
    } else {
        return false;
    };
    let page_size = crate::contract::iosurface_pages::page_size_of(state.page_shift);
    let covered = (m.page_entries.len() as u64).saturating_mul(page_size);
    covered >= span_end.max(page_size)
}

/// Ensure pages (and geom if possible) before scanout paint / type-11 Store.
///
/// Re-resolves when the table is empty, geom is missing, **or** the cached page
/// count cannot cover the latched W×H sample window (stale early resolve).
pub fn ensure_resolved_for_scanout<H: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &H,
    mapping_id: u32,
) -> bool {
    let (mapped, has_internal, empty_pages, has_geom) = match state.mappings.get(&mapping_id) {
        Some(m) => (
            m.mapped,
            m.mapping_internal != 0,
            m.page_entries.is_empty(),
            m.has_geom,
        ),
        None => return false,
    };
    let needs = mapped
        && has_internal
        && (empty_pages || !has_geom || !pages_cover_geom(state, mapping_id));
    if needs {
        resolve_mapping_backing(state, host, mapping_id)
    } else {
        mapped && !empty_pages
    }
}

/// Fail-closed page-list revalidation before host writeback or import-present.
///
/// When `mapping_internal` is set **and** we previously resolved a live
/// `page_table_kva`, re-walk MappingInternal so we never write through PFNs the
/// guest recycled (zone freelist `0xff000000ff000000` class). Resolve failure
/// **invalidates** a live table rather than writing stale PFNs.
///
/// Manual / unit-test page lists (`page_table_kva == 0`) keep their entries when
/// resolve is not available — product MAP always re-resolves once KVA is known.
pub fn revalidate_mapping_pages<H: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &H,
    mapping_id: u32,
) -> bool {
    revalidate_mapping_reason(state, host, mapping_id).is_none()
}

/// Precise reason a revalidate missed, or `None` when the mapping is resolvable.
///
/// The bool [`revalidate_mapping_pages`] collapses four distinct outcomes into
/// one "false", which forces a caller that fail-logs a lost flush to emit a
/// single `reason=revalidate` slug — and a future reader then cannot tell a
/// benign teardown window from a genuine content-drop without hunting for a
/// paired `map_revalidate resolve_fail` line. This returns the specific slug so
/// the two never share a status (AGENTS.md: each distinct check owns its slug):
/// - `revalidate_gone` / `revalidate_unmapped` — the guest already dropped the
///   mapping (pageoff/unwire raced ahead of the flush trigger); nothing to write
///   back to, benign.
/// - `revalidate_no_pages` — mapped but the page list is empty after resolve (a
///   transient (re)wire gap); benign, stale-but-coherent guest bytes remain.
/// - `revalidate_resolve_fail` — a live page table turned unreadable; the real
///   content-drop risk, and the only one that also emits the `st=invalidate`
///   line below.
pub fn revalidate_mapping_reason<H: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &H,
    mapping_id: u32,
) -> Option<&'static str> {
    let Some(m) = state.mappings.get(&mapping_id) else {
        return Some("revalidate_gone");
    };
    if !m.mapped {
        return Some("revalidate_unmapped");
    }
    let has_internal = m.mapping_internal != 0;
    let had_live_table = m.page_table_kva != 0;
    let had_pages = !m.page_entries.is_empty();
    if has_internal {
        let generation_before = m.map_generation;
        let started = std::time::Instant::now();
        let resolved = resolve_mapping_backing(state, host, mapping_id);
        let elapsed_us = started.elapsed().as_micros() as u64;
        let (pages_after, generation_after) = state
            .mappings
            .get(&mapping_id)
            .map(|entry| (entry.page_entries.len(), entry.map_generation))
            .unwrap_or((0, generation_before));
        if revalidate_timing_is_slow(elapsed_us) {
            crate::observe::off(format!(
                "map_revalidate_slow mid={mapping_id} us={elapsed_us} pages={pages_after} resolved={} generation={} changed={}",
                resolved as u8,
                generation_after,
                (generation_after != generation_before) as u8
            ));
        }
        if !resolved && had_live_table {
            // Product: table was live and is now unreadable — drop PFNs.
            if had_pages {
                let _ = state.invalidate_mapping_pages(mapping_id);
                crate::observe::fail(format!(
                    "map_revalidate mid={mapping_id} st=invalidate reason=resolve_fail"
                ));
            }
            return Some("revalidate_resolve_fail");
        }
        // No prior live KVA (first resolve miss, or test fixture with manual
        // page_entries only) — fall through to accept non-empty manual list.
    }
    match state.mappings.get(&mapping_id) {
        Some(m) if m.mapped && !m.page_entries.is_empty() => None,
        Some(_) => Some("revalidate_no_pages"),
        None => Some("revalidate_gone"),
    }
}

const REVALIDATE_SLOW_US: u64 = 1_000;

#[inline]
fn revalidate_timing_is_slow(elapsed_us: u64) -> bool {
    elapsed_us >= REVALIDATE_SLOW_US
}

/// Unmap contiguous views whose page tables changed (safe point: no GPU work
/// in flight — execution is sync-per-packet and Metal objects were dropped by
/// `type11_guest_texture_invalidate` when the view was retired).
pub fn flush_retired_views<H: HostOps>(state: &mut DeviceState, host: &mut H) {
    for (ptr, len) in state.retired_views.drain(..) {
        host.unmap_pages(ptr, len);
    }
}

/// Revalidate + collect page-aligned GPAs for a mapped surface (GVA order).
///
/// Fails closed on empty / invalid entries and known transport/control-page
/// aliases. Does not invent PFNs. Every consumer immediately passes the
/// returned GPAs to `HostOps::map_pages`, whose host callback is the
/// authoritative RAM/range validator; repeating `is_ram_gpa` once per page
/// here makes full-frame surfaces perform thousands of duplicate QEMU address
/// translations before the exact same validation in `map_pages`.
pub fn mapping_page_gpas<H: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut H,
    mapping_id: u32,
) -> Option<Vec<u64>> {
    if !revalidate_mapping_pages(state, host, mapping_id) {
        return None;
    }
    let m = state.mappings.get(&mapping_id)?;
    if !m.mapped || m.page_entries.is_empty() {
        return None;
    }
    let page_shift = state.page_shift;
    let gpas: Vec<u64> = m
        .page_entries
        .iter()
        .filter_map(|&e| crate::contract::iosurface_pages::entry_gpa_shift(e, page_shift))
        .collect();
    if gpas.is_empty() || gpas.len() != m.page_entries.len() {
        return None;
    }
    if let Some((gpa, owner)) = first_control_page_collision(state, &gpas) {
        crate::observe::fail(format!(
            "mapping_pages fail reason=control_page_collision mid={mapping_id} gpa={gpa:#x} owner={owner} pages={}",
            gpas.len()
        ));
        return None;
    }
    Some(gpas)
}

/// A render surface must never alias pages that the device knows are live
/// transport or task-control structures. `is_ram_gpa` alone cannot distinguish
/// an IOSurface page from a FIFO/page-table page; reject the provable overlap
/// before either CPU or GPU writes touch it.
fn first_control_page_collision(state: &DeviceState, gpas: &[u64]) -> Option<(u64, &'static str)> {
    let page = state.page_size();
    let page_base = |gpa: u64| gpa & !(page - 1);
    // Live tasks can advertise one million object-list slots (4,096 x86
    // pages each). A linear IOSurface scan for every control page turns one
    // full-frame safety check into ~100 million comparisons. Preserve the
    // exact collision contract with one page-base membership set.
    let mapping_pages: std::collections::HashSet<u64> =
        gpas.iter().map(|&gpa| page_base(gpa)).collect();
    let contains = |gpa: u64| mapping_pages.contains(&page_base(gpa));

    if state.gfx.root_page != 0 && contains((state.gfx.root_page as u64) << state.page_shift) {
        return Some(((state.gfx.root_page as u64) << state.page_shift, "gfx_root"));
    }
    if state.gfx.fifo_base_page != 0
        && contains((state.gfx.fifo_base_page as u64) << state.page_shift)
    {
        return Some((
            (state.gfx.fifo_base_page as u64) << state.page_shift,
            "root_fifo",
        ));
    }
    if state.iosfc.ring_base != 0 && contains(state.iosfc.ring_base) {
        return Some((page_base(state.iosfc.ring_base), "iosfc_ring"));
    }
    for ring in &state.child_rings {
        for &gpa in &ring.page_gpas {
            if contains(gpa) {
                return Some((page_base(gpa), "child_fifo"));
            }
        }
    }
    for (task_idx, task) in state.tasks.iter().enumerate() {
        if !task.active {
            continue;
        }
        if task.directory_pfn != 0 {
            let gpa = (task.directory_pfn as u64) << state.page_shift;
            if contains(gpa) {
                return Some((gpa, "task_directory"));
            }
        }
        if task.object_list_pfn != 0 {
            let first = (task.object_list_pfn as u64) << state.page_shift;
            // Reserve only up to the highest `ref` this task has actually
            // registered (`state.objects`, the host's live registry), not the
            // full advertised `object_list_count`. A live task can advertise
            // up to one million slots — the guest's headroom for future
            // registrations, not a claim that all of it is populated.
            // Reserving the whole advertised capacity as one fixed 16 MiB dead
            // zone at the low end of guest RAM declined unrelated real
            // surfaces wholesale: 325 collisions across a boot, 20+ distinct
            // mappings, most of the desktop never composited (icons, windows,
            // wallpaper). `ref` is the surface_id on the type-4 present path
            // (see the module doc), so the live high-water mark is exactly
            // what future entries can already alias — nothing past it is live
            // yet.
            let task_id = task_idx as u32;
            let live_end = state
                .objects
                .range((task_id, 0)..(task_id.saturating_add(1), 0))
                .next_back()
                .map(|(&(_, max_ref), _)| {
                    (max_ref as u64).saturating_add(1) * OBJECT_LIST_ENTRY_LEN as u64
                })
                .unwrap_or(0);
            let count = live_end.saturating_add(page - 1) / page;
            for i in 0..count {
                let gpa = first.saturating_add(i.saturating_mul(page));
                if contains(gpa) {
                    // The refusal names the colliding page; without the claim
                    // behind it there is no way to tell a real transport-page
                    // alias from an object list whose advertised slot count
                    // reserves far more guest RAM than it populates.
                    crate::observe::off(format!(
                        "control_page_claim owner=task_object_list gpa={gpa:#x} \
                         list_base={first:#x} live_end={live_end:#x} pages={count} \
                         advertised_slots={} list_end={:#x}",
                        task.object_list_count,
                        first.saturating_add(count.saturating_mul(page))
                    ));
                    return Some((gpa, "task_object_list"));
                }
            }
        }
    }
    None
}

/// Contiguous host-VA view over the mapping's guest pages (unified memory).
///
/// Builds the view on first use via [`HostOps::map_pages`] (mach_vm_remap of
/// guest RAM). Returns `(ptr, len)`. The view is the single storage for
/// surface content: Metal textures are created directly on it, so there is
/// nothing to synchronize — resolve failure here must fail the caller visibly.
///
/// **Safe zero-copy contract:** always [`revalidate_mapping_pages`] first so a
/// cached contig never aliases PFNs after ReplacePhysical / guest recycle.
///
/// On Linux, only a **packed** sequential host run succeeds. Fragmented
/// IOSurface page lists must use [`write_mapping_bytes`] / [`read_mapping_bytes`]
/// or multi-run import-present.
pub fn ensure_contig_view<H: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut H,
    mapping_id: u32,
) -> Option<(usize, usize)> {
    // Always revalidate before returning a cached contig (ReplacePhysical /
    // recycle must not leave a live view over freelist PFNs).
    if !revalidate_mapping_pages(state, host, mapping_id) {
        return None;
    }
    flush_retired_views(state, host);
    {
        let m = state.mappings.get(&mapping_id)?;
        if m.contig_ptr != 0 {
            return Some((m.contig_ptr, m.contig_len));
        }
    }
    let gpas = mapping_page_gpas(state, host, mapping_id)?;
    let page_sz = crate::contract::iosurface_pages::page_size_of(state.page_shift) as usize;
    let ptr = host.map_pages(&gpas, page_sz)?;
    let len = gpas.len() * page_sz;
    let m = state.mappings.get_mut(&mapping_id)?;
    m.contig_ptr = ptr;
    m.contig_len = len;
    Some((ptr, len))
}

/// Write `buf` into mapping linear offset `off` via packed map_pages runs.
///
/// Covers fragmented page lists (Linux product): split GPAs into maximal packed
/// runs, map each, poke, unmap. No `write_gpa`. Returns false if revalidate /
/// map fails.
pub fn write_mapping_bytes<H: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut H,
    mapping_id: u32,
    off: u64,
    buf: &[u8],
) -> bool {
    if buf.is_empty() {
        return true;
    }
    // Deferred-writeback flush-on-access: land any pending resident content
    // in these pages first so this write applies on top of it, not under it.
    crate::runtime::storage_flush::flush_intersecting(
        state,
        host,
        mapping_id,
        off,
        off.saturating_add(buf.len() as u64),
    );
    // Exact-window residency invalidation: guest pages in this range no
    // longer mirror any resident storage image (disjoint windows survive).
    state.invalidate_storage_residency_window(
        mapping_id,
        off,
        off.saturating_add(buf.len() as u64),
    );
    // Fast path: one packed view covering the write.
    let need_end = off.saturating_add(buf.len() as u64);
    if let Some((ptr, len)) = ensure_contig_view(state, host, mapping_id) {
        if (len as u64) >= need_end && (off as usize) + buf.len() <= len {
            // SAFETY: view covers need_end.
            unsafe {
                std::ptr::copy_nonoverlapping(
                    buf.as_ptr(),
                    (ptr as *mut u8).add(off as usize),
                    buf.len(),
                );
            }
            return true;
        }
    }
    let gpas = match mapping_page_gpas(state, host, mapping_id) {
        Some(g) => g,
        None => {
            crate::observe::fail(format!(
                "mapping_write fail reason=revalidate mid={mapping_id} off={off:#x} len={:#x}",
                buf.len()
            ));
            return false;
        }
    };
    let page_size = state.page_size();
    let page_sz = page_size as usize;
    let span_end = (gpas.len() as u64).saturating_mul(page_size);
    if need_end > span_end {
        crate::observe::fail(format!(
            "mapping_write fail reason=short_table mid={mapping_id} off={off:#x} len={:#x} span={span_end:#x}",
            buf.len()
        ));
        return false;
    }
    flush_retired_views(state, host);
    let runs = crate::runtime::gva_view::contig_page_runs(&gpas, page_size);
    let import_started = std::time::Instant::now();
    let end = need_end;
    for run in &runs {
        let run_gpas = &gpas[run.clone()];
        let run_mlo = (run.start as u64).saturating_mul(page_size);
        let run_mhi = (run.end as u64).saturating_mul(page_size);
        let copy_lo = off.max(run_mlo);
        let copy_hi = end.min(run_mhi);
        if copy_lo >= copy_hi {
            continue;
        }
        let Some(ptr) = host.map_pages(run_gpas, page_sz) else {
            crate::observe::fail(format!(
                "mapping_write fail reason=map_pages mid={mapping_id} run_pages={} mlo={run_mlo:#x}",
                run_gpas.len()
            ));
            return false;
        };
        let total = run_gpas.len().saturating_mul(page_sz);
        let buf_off = (copy_lo - off) as usize;
        let host_off = (copy_lo - run_mlo) as usize;
        let n = (copy_hi - copy_lo) as usize;
        if host_off + n > total || buf_off + n > buf.len() {
            host.unmap_pages(ptr, total);
            return false;
        }
        // SAFETY: map covers total; host_off+n in range.
        unsafe {
            std::ptr::copy_nonoverlapping(
                buf.as_ptr().add(buf_off),
                (ptr as *mut u8).add(host_off),
                n,
            );
        }
        host.unmap_pages(ptr, total);
    }
    let import_us = import_started.elapsed().as_micros() as u64;
    if mapping_run_import_is_slow(import_us) {
        crate::observe::off(format!(
            "mapping_write_runs mid={mapping_id} us={import_us} bytes={} pages={} runs={}",
            buf.len(),
            gpas.len(),
            runs.len()
        ));
    }
    true
}

/// Read mapping linear `[off, off+buf.len())` via packed map_pages runs.
pub fn read_mapping_bytes<H: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut H,
    mapping_id: u32,
    off: u64,
    buf: &mut [u8],
) -> bool {
    if buf.is_empty() {
        return true;
    }
    // Deferred-writeback flush-on-access: this read must observe the resident
    // content, not the stale pre-dispatch guest bytes.
    crate::runtime::storage_flush::flush_intersecting(
        state,
        host,
        mapping_id,
        off,
        off.saturating_add(buf.len() as u64),
    );
    let need_end = off.saturating_add(buf.len() as u64);
    if let Some((ptr, len)) = ensure_contig_view(state, host, mapping_id) {
        if (len as u64) >= need_end && (off as usize) + buf.len() <= len {
            unsafe {
                std::ptr::copy_nonoverlapping(
                    (ptr as *const u8).add(off as usize),
                    buf.as_mut_ptr(),
                    buf.len(),
                );
            }
            return true;
        }
    }
    let gpas = match mapping_page_gpas(state, host, mapping_id) {
        Some(g) => g,
        None => return false,
    };
    let page_size = state.page_size();
    let page_sz = page_size as usize;
    let span_end = (gpas.len() as u64).saturating_mul(page_size);
    if need_end > span_end {
        return false;
    }
    flush_retired_views(state, host);
    let runs = crate::runtime::gva_view::contig_page_runs(&gpas, page_size);
    let import_started = std::time::Instant::now();
    let end = need_end;
    for run in &runs {
        let run_gpas = &gpas[run.clone()];
        let run_mlo = (run.start as u64).saturating_mul(page_size);
        let run_mhi = (run.end as u64).saturating_mul(page_size);
        let copy_lo = off.max(run_mlo);
        let copy_hi = end.min(run_mhi);
        if copy_lo >= copy_hi {
            continue;
        }
        let Some(ptr) = host.map_pages(run_gpas, page_sz) else {
            return false;
        };
        let total = run_gpas.len().saturating_mul(page_sz);
        let buf_off = (copy_lo - off) as usize;
        let host_off = (copy_lo - run_mlo) as usize;
        let n = (copy_hi - copy_lo) as usize;
        if host_off + n > total || buf_off + n > buf.len() {
            host.unmap_pages(ptr, total);
            return false;
        }
        unsafe {
            std::ptr::copy_nonoverlapping(
                (ptr as *const u8).add(host_off),
                buf.as_mut_ptr().add(buf_off),
                n,
            );
        }
        host.unmap_pages(ptr, total);
    }
    let import_us = import_started.elapsed().as_micros() as u64;
    if mapping_run_import_is_slow(import_us) {
        crate::observe::off(format!(
            "mapping_read_runs mid={mapping_id} us={import_us} bytes={} pages={} runs={}",
            buf.len(),
            gpas.len(),
            runs.len()
        ));
    }
    true
}

const MAPPING_RUN_IMPORT_SLOW_US: u64 = 1_000;

#[inline]
fn mapping_run_import_is_slow(elapsed_us: u64) -> bool {
    elapsed_us >= MAPPING_RUN_IMPORT_SLOW_US
}

/// Fields helper for tests.
#[allow(dead_code)]
pub fn fields_ok(fields: &MapperInternalFields, mapping_id: u32) -> bool {
    fields.mapping_id == mapping_id
}

#[cfg(test)]
mod revalidate_tests {
    use super::*;
    use crate::contract::iosurface_pages::{PAGE_ENTRY_PFN_SHIFT, PAGE_ENTRY_VALID};
    use crate::model::{DeviceId, PAGE_SHIFT_X86};
    use crate::runtime::host::FakeHost;

    #[test]
    fn page_table_revalidation_slow_proxy_threshold_is_explicit() {
        assert!(!revalidate_timing_is_slow(REVALIDATE_SLOW_US - 1));
        assert!(revalidate_timing_is_slow(REVALIDATE_SLOW_US));
    }

    #[test]
    fn fragmented_run_import_slow_proxy_threshold_is_explicit() {
        assert!(!mapping_run_import_is_slow(MAPPING_RUN_IMPORT_SLOW_US - 1));
        assert!(mapping_run_import_is_slow(MAPPING_RUN_IMPORT_SLOW_US));
    }

    #[test]
    fn revalidate_fail_closed_without_internal_and_empty_pages() {
        let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_X86);
        let host = FakeHost::new();
        state.map_surface(2);
        // Mapped but no MappingInternal and no pages → not writable.
        assert!(!revalidate_mapping_pages(&mut state, &host, 2));
    }

    #[test]
    fn revalidate_reason_disambiguates_the_miss() {
        let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_X86);
        let host = FakeHost::new();
        // Unknown id → the mapping was never created / already forgotten.
        assert_eq!(
            revalidate_mapping_reason(&mut state, &host, 7),
            Some("revalidate_gone")
        );
        // Mapped but no MappingInternal and no page list → the benign
        // (re)wire-gap window, NOT a live-table resolve failure. Must carry its
        // own slug so a real content drop is never masked by this case.
        state.map_surface(2);
        assert_eq!(
            revalidate_mapping_reason(&mut state, &host, 2),
            Some("revalidate_no_pages")
        );
        // A resolvable static page list → success (None).
        state.map_surface(4);
        state.mappings.get_mut(&4).unwrap().page_entries =
            vec![(0x100 << PAGE_ENTRY_PFN_SHIFT) | PAGE_ENTRY_VALID];
        assert_eq!(revalidate_mapping_reason(&mut state, &host, 4), None);
    }

    #[test]
    fn surface_page_collision_detects_only_distinct_live_alias() {
        let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_X86);
        let entry = |pfn: u32| (pfn << PAGE_ENTRY_PFN_SHIFT) | PAGE_ENTRY_VALID;
        let gpa = |pfn: u64| pfn << PAGE_SHIFT_X86;
        // Two distinct live surfaces on disjoint pages → no collision.
        state.map_surface(10);
        state.map_surface(20);
        state.mappings.get_mut(&10).unwrap().page_entries = vec![entry(0x100), entry(0x101)];
        state.mappings.get_mut(&20).unwrap().page_entries = vec![entry(0x200), entry(0x201)];
        assert_eq!(first_surface_page_collision(&state, 10), None);
        assert_eq!(first_surface_page_collision(&state, 20), None);
        // Surface 20 rewires onto a page surface 10 still owns → collision,
        // reported against the other owner (10) at the shared GPA.
        state.mappings.get_mut(&20).unwrap().page_entries = vec![entry(0x101), entry(0x201)];
        assert_eq!(
            first_surface_page_collision(&state, 20),
            Some((gpa(0x101), 10))
        );
        // A surface never collides with itself.
        assert_eq!(
            first_surface_page_collision(&state, 10),
            Some((gpa(0x101), 20))
        );
        // If the other owner is unmapped, the alias is legitimate (handoff) →
        // no collision.
        state.unmap_surface(10);
        assert_eq!(first_surface_page_collision(&state, 20), None);
        // Empty / unmapped self → None.
        state.mappings.get_mut(&20).unwrap().page_entries.clear();
        assert_eq!(first_surface_page_collision(&state, 20), None);
    }

    #[test]
    fn reprieve_with_aliasing_peer_is_a_detected_collision() {
        // The condemn/reprieve corruptor precondition: a mapping's backing was
        // deleted (condemn stashed its pages), the guest handed the SAME
        // physical pages to another live surface, but this mapping's page table
        // still resolves to them — so the resolve fingerprints identical and
        // REPRIEVES (pages_changed == false, no map_generation bump). The rewire
        // wrong-PFN guard is gated on pages_changed and would never run; the
        // reprieve-path guard must catch it. This asserts both halves the branch
        // composes fire together on that exact state.
        let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_X86);
        let entry = |pfn: u32| (pfn << PAGE_ENTRY_PFN_SHIFT) | PAGE_ENTRY_VALID;
        let gpa = |pfn: u64| pfn << PAGE_SHIFT_X86;

        // Mapping 3: condemned, its stashed pages == the plan it re-adopts.
        state.map_surface(3);
        {
            let m = state.mappings.get_mut(&3).unwrap();
            m.mapped = true;
            m.page_entries = vec![entry(0x300), entry(0x301)];
            m.map_generation = 4;
        }
        assert!(state.condemn_surface_backing(3));
        // The re-walked plan matches the condemned fingerprint → reprieve.
        let condemned = state.mappings.get(&3).unwrap().condemned_entries.clone();
        let plan = vec![entry(0x300), entry(0x301)];
        let (pages_changed, incarnation_changed, reprieved) =
            plan_adoption_decision(condemned.as_deref(), &[], &plan);
        assert!(
            reprieved,
            "same plan as condemned fingerprint must reprieve"
        );
        assert!(!pages_changed, "reprieve must not see a page change");
        assert!(!incarnation_changed);

        // Re-adopt the plan (as the resolve would) and stand up a DIFFERENT live
        // surface (20) that now also owns page 0x301 — the guest recycled it.
        {
            let m = state.mappings.get_mut(&3).unwrap();
            m.page_entries = plan.clone();
            m.condemned_entries = None;
        }
        state.map_surface(20);
        {
            let m = state.mappings.get_mut(&20).unwrap();
            m.mapped = true;
            m.page_entries = vec![entry(0x301), entry(0x999)];
        }
        // The reprieve-path guard's detector fires: mapping 3's re-adopted page
        // 0x301 is also owned by live surface 20 — the wrong-PFN write vector the
        // rewire-only guard would have missed (pages_changed was false).
        assert_eq!(
            first_surface_page_collision(&state, 3),
            Some((gpa(0x301), 20))
        );
    }

    #[test]
    fn surface_page_collision_invalidates_mapping_fail_closed() {
        let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_X86);
        let entry = |pfn: u32| (pfn << PAGE_ENTRY_PFN_SHIFT) | PAGE_ENTRY_VALID;
        let gpa = |pfn: u64| pfn << PAGE_SHIFT_X86;
        const MID: u32 = 0x0CA;
        const OWNER: u32 = 0x0BE;

        state.map_surface(MID);
        {
            let m = state.mappings.get_mut(&MID).unwrap();
            m.mapped = true;
            m.map_generation = 7;
            m.page_entries = vec![entry(0x777), entry(0x778)];
            m.page_table_kva = 0xABC0;
        }
        state.map_surface(OWNER);
        {
            let m = state.mappings.get_mut(&OWNER).unwrap();
            m.mapped = true;
            m.page_entries = vec![entry(0x778)];
        }

        let (shared_gpa, owner) =
            first_surface_page_collision(&state, MID).expect("must detect alias");
        assert_eq!((shared_gpa, owner), (gpa(0x778), OWNER));

        fail_closed_surface_page_collision(&mut state, MID, shared_gpa, owner, 2, "test");
        let m = state.mappings.get(&MID).unwrap();
        assert!(m.mapped, "surface stays mapped but unresolved");
        assert!(
            m.page_entries.is_empty(),
            "known-bad page plan must be cleared"
        );
        assert_eq!(m.page_table_kva, 0);
        assert_eq!(
            m.map_generation, 8,
            "generation bump makes any deferred writeback fail closed"
        );
        assert_eq!(first_surface_page_collision(&state, MID), None);
    }

    #[test]
    fn revalidate_accepts_static_page_list_without_internal() {
        let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_X86);
        let host = FakeHost::new();
        state.map_surface(4);
        {
            let m = state.mappings.get_mut(&4).unwrap();
            m.page_entries = vec![(0x100 << PAGE_ENTRY_PFN_SHIFT) | PAGE_ENTRY_VALID];
        }
        assert!(revalidate_mapping_pages(&mut state, &host, 4));
    }

    #[test]
    fn mapping_io_still_rejects_non_ram_page_at_map_boundary() {
        let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_X86);
        let mut host = FakeHost::new();
        host.strict_linux_map = true;
        let mid = 6;
        assert!(state.map_surface(mid));
        {
            let m = state.mappings.get_mut(&mid).unwrap();
            m.page_entries = vec![(0x7f000 << PAGE_ENTRY_PFN_SHIFT) | PAGE_ENTRY_VALID];
        }
        let mut byte = [0u8; 1];
        assert!(!read_mapping_bytes(
            &mut state, &mut host, mid, 0, &mut byte,
        ));
        assert!(!write_mapping_bytes(&mut state, &mut host, mid, 0, &[1],));
    }

    #[test]
    fn invalidate_mapping_pages_bumps_map_generation_and_clears() {
        let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_X86);
        state.map_surface(5);
        {
            let m = state.mappings.get_mut(&5).unwrap();
            m.page_entries = vec![1];
            m.contig_ptr = 0xdead;
            m.contig_len = 4096;
        }
        let gen0 = state.mappings.get(&5).unwrap().map_generation;
        assert!(state.invalidate_mapping_pages(5));
        let m = state.mappings.get(&5).unwrap();
        assert!(m.page_entries.is_empty());
        assert_eq!(m.contig_ptr, 0);
        assert!(m.map_generation != gen0);
        assert_eq!(state.retired_views, vec![(0xdead, 4096)]);
    }

    /// Product Linux: full page list is non-packed → ensure_contig_view fails;
    /// write_mapping_bytes still lands bytes via maximal packed runs.
    #[test]
    fn multi_import_fragmented_mapping_write() {
        let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_X86);
        let mut host = FakeHost::new();
        host.strict_linux_map = true;
        let page = 1u64 << PAGE_SHIFT_X86;
        // Two non-adjacent guest pages (gap in GPA → not one packed map_pages).
        let gpa0 = 0x1000_0000u64;
        let gpa1 = 0x2000_0000u64;
        host.map_range(gpa0, page as usize, 0);
        host.map_range(gpa1, page as usize, 0);
        let pfn0 = (gpa0 >> PAGE_SHIFT_X86) as u32;
        let pfn1 = (gpa1 >> PAGE_SHIFT_X86) as u32;
        let mid = 9u32;
        state.map_surface(mid);
        {
            let m = state.mappings.get_mut(&mid).unwrap();
            m.page_entries = vec![
                (pfn0 << PAGE_ENTRY_PFN_SHIFT) | PAGE_ENTRY_VALID,
                (pfn1 << PAGE_ENTRY_PFN_SHIFT) | PAGE_ENTRY_VALID,
            ];
        }
        assert!(
            ensure_contig_view(&mut state, &mut host, mid).is_none(),
            "fragmented list must not pack under strict_linux_map"
        );
        let payload = b"FRAG-MULTI-IMPORT-OK!!!!"; // 24 bytes
        assert!(write_mapping_bytes(&mut state, &mut host, mid, 0, payload));
        // Second page offset = page_size.
        let mut hi = [0u8; 8];
        assert!(read_mapping_bytes(
            &mut state, &mut host, mid, page, &mut hi
        ));
        // Write only touched page 0; page 1 still zero.
        assert_eq!(hi, [0u8; 8]);
        let mut lo = [0u8; 24];
        assert!(read_mapping_bytes(&mut state, &mut host, mid, 0, &mut lo));
        assert_eq!(&lo[..], &payload[..]);
        // Cross-page write spanning the gap.
        let cross = vec![0xABu8; 16];
        let off = page - 8;
        assert!(write_mapping_bytes(&mut state, &mut host, mid, off, &cross));
        let mut check = [0u8; 16];
        assert!(read_mapping_bytes(
            &mut state, &mut host, mid, off, &mut check
        ));
        assert_eq!(check, [0xABu8; 16]);
    }
}

/// Read ring entry at absolute producer index (for tests).
pub fn read_request_at_producer<M: HostMemory>(
    host: &M,
    ring_base: u64,
    producer: u32,
) -> Result<(u32, u32), MemError> {
    let entry_off = mapper_request_published_entry_offset(producer).ok_or(MemError::BadArgs)?;
    let mut e = [0u8; MAPPER_REQUEST_ENTRY_LEN];
    host.read_gpa(ring_base + entry_off, &mut e)?;
    let req = decode_mapper_request_entry(&e).map_err(|_| MemError::BadArgs)?;
    Ok((req.request_type, req.mapping_id))
}

#[cfg(test)]
mod tests {

    use super::*;
    use crate::contract::endian::st32;
    use crate::contract::iosurface_pages::{
        MAPPING_INTERNAL_BACKPTR, MAPPING_INTERNAL_EXPECTED_SIZE, MAPPING_INTERNAL_ID,
        MAPPING_INTERNAL_SIZE, PAGE_ENTRY_PFN_SHIFT, PAGE_ENTRY_VALID,
    };

    #[test]
    fn plan_adoption_decision_incarnation_semantics() {
        // No condemn: plain pages-changed compare against the live entries.
        assert_eq!(
            plan_adoption_decision(None, &[1, 2], &[1, 2]),
            (false, false, false)
        );
        assert_eq!(
            plan_adoption_decision(None, &[1, 2], &[1, 3]),
            (true, false, false)
        );
        // Condemned + identical plan = stale delete reprieve: the SAME
        // incarnation lives on — no bump, no drop (black-band class).
        assert_eq!(
            plan_adoption_decision(Some(&[1, 2]), &[], &[1, 2]),
            (false, false, true)
        );
        // Condemned + different plan = the backing really died and the id
        // was re-used: bump + drop the old incarnation's windows.
        assert_eq!(
            plan_adoption_decision(Some(&[1, 2]), &[], &[7, 8]),
            (true, true, false)
        );
        // The live (cleared) entries never mask the fingerprint compare.
        assert_eq!(
            plan_adoption_decision(Some(&[1, 2]), &[7, 8], &[1, 2]),
            (false, false, true)
        );
    }

    #[test]
    fn resolve_fail_latch_dedups_per_mapping_and_rearms_on_clear() {
        // Flood guard for the per-present `resolve_mapping_backing` path: a
        // genuinely-broken mapping must log each reason once, re-arm when it
        // resolves, and never bleed across mappings. Unique ids so this never
        // races real mappings across the process-global latch.
        let mid = 0xF00D_0001u32;
        let other = 0xF00D_0002u32;
        clear_resolve_fail(mid);
        clear_resolve_fail(other);
        let seen = |m: u32, r: &'static str| {
            resolve_fail_latch()
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .contains(&(m, r))
        };
        note_resolve_fail(mid, "iosurface_validate_mapping_id_mismatch", "x".into());
        assert!(seen(mid, "iosurface_validate_mapping_id_mismatch"));
        // A different reason on the same mapping is tracked independently.
        note_resolve_fail(mid, "iosurface_mapper_internal_owner_read", "x".into());
        assert!(seen(mid, "iosurface_mapper_internal_owner_read"));
        // A different mapping is untouched by mid's failures.
        assert!(!seen(other, "iosurface_validate_mapping_id_mismatch"));
        // Clearing mid re-arms both its reasons but leaves `other` alone.
        note_resolve_fail(other, "iosurface_validate_mapping_id_mismatch", "x".into());
        clear_resolve_fail(mid);
        assert!(!seen(mid, "iosurface_validate_mapping_id_mismatch"));
        assert!(!seen(mid, "iosurface_mapper_internal_owner_read"));
        assert!(seen(other, "iosurface_validate_mapping_id_mismatch"));
        clear_resolve_fail(other);
    }

    #[test]
    fn mapper_declines_are_exact_and_log_safe() {
        use crate::observe::Decline;

        let declines = [
            MapperDecline::CaptureMapperXregRead(MemError::XregUnavailable),
            MapperDecline::CaptureRequestTypeXregRead(MemError::XregUnavailable),
            MapperDecline::CaptureInternalXregRead(MemError::XregUnavailable),
            MapperDecline::CaptureRequestTypeMismatch,
            MapperDecline::CaptureInternalZero,
            MapperDecline::CaptureInternalKvaInvalid,
            MapperDecline::CaptureMapperKvaInvalid,
            MapperDecline::DeviceDescriptorRead(MemError::Unmapped),
        ];
        let mut slugs = std::collections::HashSet::new();
        for decline in declines {
            let slug = decline.slug();
            assert!(slug.starts_with("mapper_"));
            assert!(
                slug.bytes()
                    .all(|byte| byte.is_ascii_lowercase() || byte.is_ascii_digit() || byte == b'_'),
                "not log-safe: {slug}"
            );
            assert!(slugs.insert(slug), "duplicate mapper decline: {slug}");
        }
        assert_eq!(
            crate::observe::Emit::decline(
                "mapper_capture_fail",
                &MapperDecline::CaptureRequestTypeMismatch,
            )
            .field("mapping", 9)
            .render(),
            "mapper_capture_fail reason=mapper_capture_request_type_mismatch mapping=9"
        );
    }

    #[test]
    fn mapper_boundary_preserves_the_iosurface_check_reason() {
        let status =
            iosurface_pages::Status::ErrInternalRead("iosurface_mapper_internal_mapping_id_read");
        assert_eq!(
            refusal_reason(&status),
            "iosurface_mapper_internal_mapping_id_read"
        );
        assert_eq!(
            crate::observe::Emit::refusal("mapper_resolve_fail", &status)
                .unwrap()
                .field("mapping", 4)
                .render(),
            "mapper_resolve_fail reason=iosurface_mapper_internal_mapping_id_read \
             class=internal_read mapping=4"
        );
    }
    use crate::model::{DeviceId, PAGE_SHIFT_ARM64E, PAGE_SIZE_ARM64E};
    use crate::runtime::host::FakeHost;

    /// arm64e kernel VA base used by the contract.
    const KVA: u64 = 0xfffffe00_10000000;

    fn put_u32(h: &mut FakeHost, gpa: u64, v: u32) {
        h.map_range(gpa, 4, 0);
        h.put_u32(gpa, v);
    }
    fn put_u64(h: &mut FakeHost, gpa: u64, v: u64) {
        h.map_range(gpa, 8, 0);
        let b = v.to_le_bytes();
        let _ = h.write_gpa(gpa, &b);
    }

    #[test]
    fn capture_validates_identity_and_ring() {
        let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
        let mut host = FakeHost::new();
        let ring = 0x7000_0000u64;
        state.iosfc.ring_base = ring;

        // producer=1 → entry 0: MAP mapping_id=7
        let mut entry = [0u8; 16];
        st32(&mut entry[0..], MAPPER_REQUEST_MAP);
        st32(&mut entry[4..], 7);
        host.map_range(ring, 16, 0);
        let _ = host.write_gpa(ring, &entry);

        let internal = KVA;
        let mapper = KVA + 0x1000;
        // MappingInternal identity fields
        put_u64(&mut host, internal + MAPPING_INTERNAL_BACKPTR, mapper);
        put_u32(&mut host, internal + MAPPING_INTERNAL_ID, 7);
        put_u32(
            &mut host,
            internal + MAPPING_INTERNAL_SIZE,
            MAPPING_INTERNAL_EXPECTED_SIZE,
        );

        host.set_xreg(MAPPER_CAPTURE_REG_MAPPER_DEVICE, mapper);
        host.set_xreg(MAPPER_CAPTURE_REG_REQUEST_TYPE, MAPPER_REQUEST_MAP as u64);
        host.set_xreg(MAPPER_CAPTURE_REG_MAPPING_INTERNAL, internal);

        let cap = capture_at_producer(&state, &host, 1).expect("capture");
        assert_eq!(cap.producer, 1);
        assert_eq!(cap.mapping_internal, internal);
        assert!(apply_capture(&mut state, &cap, 7));
        assert_eq!(state.mappings.get(&7).unwrap().mapping_internal, internal);
    }

    #[test]
    fn capture_handoff_mismatch_is_fail_visible_and_latched() {
        // A decoded MAP request whose captured handoff registers disagree with
        // the ring (wrong request-type in the xreg) is a genuine capture miss:
        // the mapping never attaches → downstream black. It must return None,
        // latch its reason once (no per-publish flood), and re-arm on clear.
        let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
        let mut host = FakeHost::new();
        let ring = 0x7100_0000u64;
        state.iosfc.ring_base = ring;

        // producer=1 → entry 0: MAP mapping_id=9
        let mut entry = [0u8; 16];
        st32(&mut entry[0..], MAPPER_REQUEST_MAP);
        st32(&mut entry[4..], 9);
        host.map_range(ring, 16, 0);
        let _ = host.write_gpa(ring, &entry);

        let internal = KVA;
        // xreg request-type disagrees with the ring's MAP → handoff mismatch.
        host.set_xreg(MAPPER_CAPTURE_REG_MAPPER_DEVICE, 0);
        host.set_xreg(MAPPER_CAPTURE_REG_REQUEST_TYPE, MAPPER_REQUEST_UNMAP as u64);
        host.set_xreg(MAPPER_CAPTURE_REG_MAPPING_INTERNAL, internal);

        clear_resolve_fail(9);
        let seen = |m: u32, r: &'static str| {
            resolve_fail_latch()
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .contains(&(m, r))
        };
        assert!(capture_at_producer(&state, &host, 1).is_none());
        assert!(seen(9, "mapper_capture_request_type_mismatch"));
        // A second identical publish must not add a duplicate (still one entry).
        assert!(capture_at_producer(&state, &host, 1).is_none());
        // A clean resolve of the same mapping re-arms the capture reason.
        clear_resolve_fail(9);
        assert!(!seen(9, "mapper_capture_request_type_mismatch"));
    }

    #[test]
    fn resolve_builds_page_entries() {
        let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
        let mut host = FakeHost::new();
        let internal = KVA;
        let mapper = KVA + 0x1000;
        let page_obj = KVA + 0x2000;
        let table = KVA + 0x3000;
        let pfn = 0x1e88c_u32;
        let page_gpa = (pfn as u64) << PAGE_SHIFT_ARM64E;

        put_u64(&mut host, internal + MAPPING_INTERNAL_BACKPTR, mapper);
        put_u32(&mut host, internal + MAPPING_INTERNAL_ID, 3);
        put_u32(
            &mut host,
            internal + MAPPING_INTERNAL_SIZE,
            MAPPING_INTERNAL_EXPECTED_SIZE,
        );
        // page fields: 0x48 points at page_obj which has table ptr at +0xb8
        put_u64(
            &mut host,
            internal + iosurface_pages::MAPPING_INTERNAL_PAGE_FIELD_48,
            page_obj,
        );
        put_u64(
            &mut host,
            internal + iosurface_pages::MAPPING_INTERNAL_PAGE_FIELD_50,
            0,
        );
        put_u64(
            &mut host,
            internal + iosurface_pages::MAPPING_INTERNAL_PAGE_COUNT,
            1,
        );
        put_u64(
            &mut host,
            page_obj + iosurface_pages::MAPPING_PAGE_TABLE_FROM_F48,
            table,
        );
        let entry = (pfn << PAGE_ENTRY_PFN_SHIFT) | PAGE_ENTRY_VALID;
        put_u32(&mut host, table, entry);
        // one page of guest RAM for the surface
        host.map_range(page_gpa, PAGE_SIZE_ARM64E as usize, 0x55);

        state.mapper_device_kva = mapper;
        assert!(state.attach_mapping_internal(3, internal));
        assert!(resolve_mapping_backing(&mut state, &host, 3));
        let m = state.mappings.get(&3).unwrap();
        assert_eq!(m.page_entries.len(), 1);
        assert_eq!(m.page_entries[0], entry);
    }

    struct FailingKvaHost {
        inner: FakeHost,
        err: MemError,
    }

    impl HostMemory for FailingKvaHost {
        fn read_gpa(&self, gpa: u64, buf: &mut [u8]) -> Result<(), MemError> {
            self.inner.read_gpa(gpa, buf)
        }

        fn write_gpa(&mut self, gpa: u64, buf: &[u8]) -> Result<(), MemError> {
            self.inner.write_gpa(gpa, buf)
        }
    }

    impl HostOps for FailingKvaHost {
        fn mono_ns(&self) -> u64 {
            0
        }

        fn enqueue(&mut self, _action: crate::runtime::host::HostAction) {}

        fn schedule_bh(&mut self) {}

        fn read_kva(&self, _kva: u64, _buf: &mut [u8]) -> Result<(), MemError> {
            Err(self.err)
        }

        fn map_pages(&mut self, gpas: &[u64], page_size: usize) -> Option<usize> {
            self.inner.map_pages(gpas, page_size)
        }

        fn unmap_pages(&mut self, ptr: usize, len: usize) {
            self.inner.unmap_pages(ptr, len);
        }

        fn is_ram_gpa(&self, gpa: u64) -> bool {
            self.inner.is_ram_gpa(gpa)
        }
    }

    fn assert_revalidate_error_preserves_cached_page_plan(err: MemError) {
        let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
        let entry = (0x444u32 << PAGE_ENTRY_PFN_SHIFT) | PAGE_ENTRY_VALID;
        assert!(state.attach_mapping_internal(3, KVA));
        assert!(state.set_mapping_geom(
            3,
            64,
            64,
            crate::contract::pixel_format::MTL_FORMAT_BGRA8_UNORM
        ));
        {
            let m = state.mappings.get_mut(&3).unwrap();
            m.mapped = true;
            m.page_entries = vec![entry];
            m.page_table_kva = KVA + 0x3000;
        }

        let host = FailingKvaHost {
            inner: FakeHost::new(),
            err,
        };
        clear_resolve_fail(3);
        let log_before = std::fs::read_to_string(crate::observe::fail_log_path())
            .unwrap_or_default()
            .len();
        assert_eq!(revalidate_mapping_reason(&mut state, &host, 3), None);
        let m = state.mappings.get(&3).unwrap();
        assert_eq!(m.page_entries, vec![entry]);
        assert_eq!(m.page_table_kva, KVA + 0x3000);
        let log_after =
            std::fs::read_to_string(crate::observe::fail_log_path()).unwrap_or_default();
        assert!(
            !log_after[log_before..].contains("mapper_revalidate_fallback"),
            "an expected cached-plan alias fallback must stay silent: {}",
            &log_after[log_before..]
        );
    }

    #[test]
    fn revalidate_no_cpu_preserves_cached_page_plan() {
        assert_revalidate_error_preserves_cached_page_plan(MemError::NoCpu);
    }

    #[test]
    fn revalidate_unmapped_read_preserves_cached_page_plan() {
        assert_revalidate_error_preserves_cached_page_plan(MemError::Unmapped);
    }

    /// qemu-shim: early page resolve + late geom must re-expand the table.
    /// IOSurface PAGE_SIZE is 16 KiB (arm64e). 1440×1080 BGRA needs
    /// ALIGN_UP(1440×4,128)×1080 = 6 220 800 bytes ≈ 380 pages; a 1-page stale
    /// table must not cover (dual-mid Store after mode switch).
    #[test]
    fn pages_cover_geom_false_when_table_shorter_than_span() {
        let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
        assert!(state.attach_mapping_internal(8, KVA));
        {
            let m = state.mappings.get_mut(&8).unwrap();
            m.mapped = true;
            // Stale early resolve: single PAGE_SIZE before geom latched.
            m.page_entries = vec![0x11; 1];
        }
        assert!(state.set_mapping_geom(
            8,
            1440,
            1080,
            crate::contract::pixel_format::MTL_FORMAT_BGRA8_UNORM
        ));
        assert!(
            !pages_cover_geom(&state, 8),
            "1×16KiB page cannot cover 1440×1080 BGRA sample window"
        );
        let host = FakeHost::new();
        let _ = ensure_resolved_for_scanout(&mut state, &host, 8);
        assert!(!pages_cover_geom(&state, 8));
    }

    #[test]
    fn pages_cover_geom_true_for_full_table() {
        let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
        assert!(state.attach_mapping_internal(3, KVA));
        assert!(state.set_mapping_geom(
            3,
            64,
            64,
            crate::contract::pixel_format::MTL_FORMAT_BGRA8_UNORM
        ));
        // 64×64 BGRA packed bpr 256 → 16 KiB → 1×16KiB page covers.
        {
            let m = state.mappings.get_mut(&3).unwrap();
            m.mapped = true;
            m.page_entries = vec![0x22; 1];
        }
        assert!(pages_cover_geom(&state, 3));
    }

    /// 249² Favourites-class tiles fit in 16×16KiB pages; short-table proxy is
    /// desktop dual-mid, not tile size alone.
    #[test]
    fn pages_cover_geom_249_tile_fits_in_sixteen_16k_pages() {
        let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
        assert!(state.attach_mapping_internal(8, KVA));
        {
            let m = state.mappings.get_mut(&8).unwrap();
            m.mapped = true;
            m.page_entries = vec![0x11; 16];
        }
        assert!(state.set_mapping_geom(
            8,
            249,
            249,
            crate::contract::pixel_format::MTL_FORMAT_BGRA8_UNORM
        ));
        assert!(
            pages_cover_geom(&state, 8),
            "live Favourites pages=16 is enough for 249² BGRA at 16KiB pages"
        );
    }

    #[test]
    fn render_pages_reject_known_device_and_task_control_pages() {
        let mut state = DeviceState::new(DeviceId(1), crate::model::PAGE_SHIFT_X86);
        state.gfx.root_page = 0x120;
        state.child_rings[2].page_gpas = vec![0x330_000];
        assert!(state.define_task(1, 0x4000_0000, 0x440));
        assert!(state.set_object_list(1, 0x550, 1024));
        // ref=341 is the lowest ref whose 12-byte entry lands in the object
        // list's second page (0x550000 + 0x1000): offset 341*12=4092..4104
        // crosses the 4096 page boundary. Registering it is what makes
        // 0x551_000 live — see the reservation test below for the case where
        // nothing is registered.
        assert!(state.insert_object(1, 341, crate::model::ObjectEntry::default()));

        assert_eq!(
            first_control_page_collision(&state, &[0x120_000]),
            Some((0x120_000, "gfx_root"))
        );
        assert_eq!(
            first_control_page_collision(&state, &[0x330_000]),
            Some((0x330_000, "child_fifo"))
        );
        assert_eq!(
            first_control_page_collision(&state, &[0x440_000]),
            Some((0x440_000, "task_directory"))
        );
        assert_eq!(
            first_control_page_collision(&state, &[0x551_000]),
            Some((0x551_000, "task_object_list"))
        );
        assert_eq!(first_control_page_collision(&state, &[0x660_000]), None);
    }

    /// The bug this closes: the reservation used to span `object_list_count`
    /// (the guest's advertised ceiling — observed at 1,048,576, a 16 MiB dead
    /// zone) regardless of how many refs were actually live. A live task with
    /// zero registered objects reserved the same 16 MiB as one with a full
    /// desktop's worth, declining unrelated real surfaces that merely landed
    /// inside the advertised-but-unpopulated range. The reservation must
    /// track `state.objects`, the host's actual live registry, not the
    /// guest's ceiling.
    #[test]
    fn object_list_reservation_tracks_live_refs_not_advertised_capacity() {
        let mut state = DeviceState::new(DeviceId(1), crate::model::PAGE_SHIFT_X86);
        assert!(state.define_task(1, 0x4000_0000, 0));
        // The advertised ceiling used in the pre-fix code (DEFAULT_OBJECT_LIST_COUNT).
        assert!(state.set_object_list(1, 0x1, 1_048_576));

        // No object registered yet: nothing in the advertised range is live,
        // so a real surface anywhere in it — even one page in — must not
        // collide.
        assert_eq!(first_control_page_collision(&state, &[0x2_000]), None);

        // Registering ref=0 makes only its own page live.
        assert!(state.insert_object(1, 0, crate::model::ObjectEntry::default()));
        assert_eq!(
            first_control_page_collision(&state, &[0x1_000]),
            Some((0x1_000, "task_object_list"))
        );
        // A page far past the one live entry is still free.
        assert_eq!(first_control_page_collision(&state, &[0x100_000]), None);
    }
}

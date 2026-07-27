//! Host-pointer import-window geometry for `VK_EXT_external_memory_host`.
//!
//! Importing registers (pins) host pages for GPU DMA; importing a whole QEMU
//! RAMBlock VMA would pin all guest RAM for the VM lifetime. These helpers bucket
//! a span into a `HOST_IMPORT_WINDOW_CAP`-bounded, VMA-relative aligned window so
//! the pinning stays bounded while a handful of windows amortize steady state.
//! Pure geometry plus the per-host "which mapping holds this pointer?" query
//! (`/proc/self/maps` on Linux, `mach_vm_region` on macOS) — no `ResourcePools`
//! state — so it lives apart from the pool bookkeeping in the parent module.

/// Importing registers (pins) host pages for GPU DMA — importing the whole
/// QEMU RAMBlock VMA (16 GiB+) pins the guest's entire RAM on the host for
/// the VM lifetime. The driver accepts it, but the host pays gigabytes of
/// locked memory. Windows bound the pinning while keeping the amortization:
/// spans bucket into VMA-relative aligned windows, so steady state reuses a
/// handful of windows instead of importing per span.
/// Standing rule (AGENTS.md): never raise this back to whole-VMA.
///
/// Prefer few large windows over many small ones. A touched page pulls its
/// whole bucket, so a smaller window looks like it should buy more distinct hot
/// regions for the same pinned bytes — but every live import is a buffer the
/// kernel revalidates on **each** queue submit, and userptr imports are the
/// expensive kind. Measured on RADV (Renoir) with a 4 GiB budget held constant,
/// same guest, same workload:
///
/// | window | live regions | `engine_submit_us` per submit |
/// |---|---|---|
/// | 256 MiB | 16 | ~41 ms |
/// | 1 GiB | 4 | ~4.1 ms |
///
/// Submission fell from 77% of the tranche to 45%, and the tranche from ~30 to
/// ~4.9 ms per draw. Coverage is the byte cap's job; the window size is what
/// the submit path pays for, so keep it at the coarse end of what the caps
/// allow. 1 GiB is also a multiple of every `minImportedHostPointerAlignment`
/// in the support matrix (4 KiB / 16 KiB guest pages).
pub(super) const HOST_IMPORT_WINDOW_CAP: u64 = 1 << 30;

/// Compute the capped import window inside `[vma_base, vma_base+vma_len)`
/// covering `[ptr, end)`: the `HOST_IMPORT_WINDOW_CAP`-aligned bucket
/// (relative to the VMA base) containing `ptr`, extended when the span
/// crosses the bucket edge, clamped to the VMA. A VMA at or under the cap
/// imports whole. `align` (the driver's min imported-host-pointer
/// alignment) divides the result because VMA bounds are page-aligned and
/// the cap is a page multiple; the caller re-checks before importing.
pub(super) fn capped_import_window(
    vma_base: usize,
    vma_len: u64,
    ptr: usize,
    end: u64,
    align: u64,
) -> (usize, u64) {
    debug_assert!(
        ptr >= vma_base && (ptr as u64) < vma_base as u64 + vma_len,
        "ptr {ptr:#x} must lie inside the VMA {vma_base:#x}+{vma_len:#x}; both \
         callers establish this by finding the mapping that contains it, and \
         without it the bucket subtraction below underflows"
    );
    if vma_len <= HOST_IMPORT_WINDOW_CAP {
        return (vma_base, vma_len);
    }
    let vma_end = vma_base as u64 + vma_len;
    let bucket = (ptr as u64 - vma_base as u64) / HOST_IMPORT_WINDOW_CAP;
    let base = vma_base as u64 + bucket * HOST_IMPORT_WINDOW_CAP;
    let span_end = end.div_ceil(align) * align;
    let win_end = (base + HOST_IMPORT_WINDOW_CAP).max(span_end).min(vma_end);
    (base as usize, win_end - base)
}

/// Bounds of the host mapping containing `ptr`, or `None` if it cannot be
/// determined.
///
/// `None` is not benign: without VMA bounds `capped_import_window` has nothing to
/// bucket against, so `host_import_resolve` falls through to its bare
/// aligned-span candidate and imports **one region per span**. On the arm64 guest
/// that is one 16 KiB region per page touched, which walks the 512-region cap in
/// ~128 MiB — two orders of magnitude below the 1 GiB byte cap that is supposed
/// to be the binding constraint — and every import after that declines, leaving
/// fresh surfaces at zeros. See [[failure-logging]] for the boot that showed it.
#[cfg(not(target_os = "macos"))]
pub(super) fn vma_bounds(ptr: usize) -> Option<(usize, usize)> {
    let maps = std::fs::read_to_string("/proc/self/maps").ok()?;
    vma_bounds_in(&maps, ptr)
}

/// macOS has no `/proc`, so the bounds come from the Mach VM map instead.
///
/// This is a *query about a pointer we already hold*, not a scan: one
/// `mach_vm_region` call asking which region contains `ptr`. Mach coalesces
/// adjacent mappings that share protection, inheritance and behaviour, so the
/// reported region can be **wider** than QEMU's RAMBlock. That is safe here and
/// only here, because the caller never imports the region — it imports the
/// `HOST_IMPORT_WINDOW_CAP` bucket of it containing `ptr`, so a too-wide answer
/// shifts bucket boundaries without ever growing what gets pinned.
#[cfg(target_os = "macos")]
pub(super) fn vma_bounds(ptr: usize) -> Option<(usize, usize)> {
    // `mach_vm_region` lives in libSystem; `libc` exposes the task port and the
    // Mach integer types but not this call.
    extern "C" {
        // `mach_task_self()` is a C macro over this global; `libc`'s function
        // wrapper for it is deprecated in favour of the `mach2` crate, and
        // taking a dependency to read one `int` is not worth it.
        static mach_task_self_: libc::c_uint;
        fn mach_vm_region(
            target_task: libc::c_uint,
            address: *mut u64,
            size: *mut u64,
            flavor: libc::c_int,
            info: *mut libc::c_int,
            info_count: *mut libc::c_uint,
            object_name: *mut libc::c_uint,
        ) -> libc::c_int;
    }
    // `VM_REGION_BASIC_INFO_64` and the `int`-count of
    // `vm_region_basic_info_data_64_t` (40 bytes / 4), from `mach/vm_region.h`.
    const VM_REGION_BASIC_INFO_64: libc::c_int = 9;
    const VM_REGION_BASIC_INFO_COUNT_64: libc::c_uint = 10;
    const KERN_SUCCESS: libc::c_int = 0;
    const VM_PROT_READ: libc::c_int = 0x01;

    let mut address = ptr as u64;
    let mut size = 0u64;
    let mut info = [0i32; VM_REGION_BASIC_INFO_COUNT_64 as usize];
    let mut count = VM_REGION_BASIC_INFO_COUNT_64;
    let mut object_name: libc::c_uint = 0;
    // SAFETY: all out-params are owned locals of the sizes the flavor declares,
    // and `count` says how many `int`s `info` can hold.
    let kr = unsafe {
        mach_vm_region(
            mach_task_self_,
            &mut address,
            &mut size,
            VM_REGION_BASIC_INFO_64,
            info.as_mut_ptr(),
            &mut count,
            &mut object_name,
        )
    };
    if kr != KERN_SUCCESS || size == 0 {
        return None;
    }
    // `mach_vm_region` returns the region at or *after* the address it is given,
    // so a hit has to be re-confirmed: an unmapped `ptr` yields the next region up.
    let base = usize::try_from(address).ok()?;
    let len = usize::try_from(size).ok()?;
    if ptr < base || ptr >= base.checked_add(len)? {
        return None;
    }
    // Same check as the Linux parser: the GPU must be able to read the pages, so
    // a non-readable region can never be a guest-RAM block. `protection` is the
    // first `int` of `vm_region_basic_info_data_64_t`.
    if info[0] & VM_PROT_READ == 0 {
        return None;
    }
    Some((base, len))
}

/// Pure parser over `/proc/self/maps` content: find the readable mapping
/// containing `ptr`. Line shape: `start-end perms offset dev inode [path]`.
///
/// Compiled on every host so its tests run everywhere, not only on the Linux
/// rows — the parser is where the Linux answer's correctness lives.
#[cfg_attr(
    target_os = "macos",
    allow(dead_code, reason = "compiled for its tests; the caller here is Mach")
)]
fn vma_bounds_in(maps: &str, ptr: usize) -> Option<(usize, usize)> {
    for line in maps.lines() {
        let mut fields = line.split_whitespace();
        let range = fields.next()?;
        let perms = fields.next().unwrap_or("");
        let (start_hex, end_hex) = range.split_once('-')?;
        let start = usize::from_str_radix(start_hex, 16).ok()?;
        let end = usize::from_str_radix(end_hex, 16).ok()?;
        if ptr >= start && ptr < end {
            // The GPU must be able to read the pages; a non-readable VMA
            // (guard region) can never be a guest-RAM block.
            if !perms.starts_with('r') {
                return None;
            }
            return Some((start, end - start));
        }
    }
    None
}

#[cfg(test)]
mod host_import_window_tests {
    use super::{capped_import_window, HOST_IMPORT_WINDOW_CAP};

    /// The bug this closes: `vma_bounds` was `/proc/self/maps`-only, so on macOS
    /// it answered `None` for *every* pointer. With no VMA to bucket against,
    /// `host_import_resolve` fell through to its bare aligned-span candidate and
    /// created one region per span — on a live arm64 boot, 310 of 513 regions were
    /// exactly one 16 KiB guest page. The 512-region cap then fired at 128 MiB,
    /// against a 1 GiB byte cap that never came close, and every later import
    /// declined.
    #[cfg(target_os = "macos")]
    #[test]
    fn mach_answers_bounds_that_contain_a_live_mapping() {
        let buf = vec![0u8; 1 << 20];
        let ptr = buf.as_ptr() as usize;
        let (base, len) = super::vma_bounds(ptr).expect(
            "a live readable mapping must have bounds — None here is what \
             collapses the window resolver to one region per span",
        );
        assert!(
            base <= ptr && ptr < base + len,
            "bounds {base:#x}+{len:#x} must contain {ptr:#x}"
        );
        assert!(
            len >= buf.len(),
            "the region must cover the whole allocation, got {len:#x}"
        );
    }

    /// A pointer past the end of its region must not inherit that region's bounds.
    /// `mach_vm_region` answers with the region at or *after* the address it is
    /// given, so without the containment re-check an unmapped address silently
    /// returns the next mapping up — and the caller would bucket a window that
    /// does not hold the span at all.
    #[cfg(target_os = "macos")]
    #[test]
    fn mach_rejects_an_address_outside_every_readable_region() {
        // `__PAGEZERO` covers the low address space with `VM_PROT_NONE`, so this
        // exercises the readability check rather than the containment one.
        assert_eq!(super::vma_bounds(0x1000), None);
    }

    /// With real bounds a single guest page buckets into a whole window instead
    /// of its own 16 KiB region — the property whose absence walked the region
    /// cap. 16 KiB is the arm64 guest page (`PAGE_SIZE_ARM64E`).
    #[test]
    fn one_guest_page_buckets_into_a_window_rather_than_its_own_region() {
        const GUEST_PAGE: u64 = 0x4000;
        let ptr = VMA_BASE + 0x3_0000_0000;
        let (base, len) =
            capped_import_window(VMA_BASE, VMA_LEN, ptr, ptr as u64 + GUEST_PAGE, ALIGN);
        assert_eq!(
            len, HOST_IMPORT_WINDOW_CAP,
            "a page must pull a whole window"
        );
        assert!(base <= ptr && (ptr as u64) < base as u64 + len);
        // 512 spans strided across the whole VMA collapse to one region per
        // bucket they land in — never one per span, which is what walked the
        // region cap. The bound follows the window size rather than fixing a
        // count, so it holds when the cap is retuned.
        const STRIDE: usize = 0x0200_0000;
        let windows: std::collections::BTreeSet<usize> = (0..512)
            .map(|i| {
                let p = VMA_BASE + i * STRIDE;
                capped_import_window(VMA_BASE, VMA_LEN, p, p as u64 + GUEST_PAGE, ALIGN).0
            })
            .collect();
        let spanned = (512 * STRIDE as u64).min(VMA_LEN);
        assert!(
            (windows.len() as u64) <= spanned.div_ceil(HOST_IMPORT_WINDOW_CAP),
            "scattered pages must collapse into their buckets, got {} for {} bytes at {HOST_IMPORT_WINDOW_CAP:#x}",
            windows.len(),
            spanned
        );
        assert!(windows.len() < 512, "one region per span is the bug");
    }

    const ALIGN: u64 = 0x1000;
    const VMA_BASE: usize = 0x7f3a_0000_0000;
    const VMA_LEN: u64 = 0x4_0000_0000; // 16 GiB QEMU RAMBlock shape

    /// A VMA at or under the cap imports whole — small blocks (ROMs, option
    /// RAM) keep the one-import-serves-all behavior.
    #[test]
    fn small_vma_imports_whole() {
        let (base, len) = capped_import_window(
            VMA_BASE,
            HOST_IMPORT_WINDOW_CAP,
            VMA_BASE + 0x5000,
            VMA_BASE as u64 + 0x9000,
            ALIGN,
        );
        assert_eq!((base, len), (VMA_BASE, HOST_IMPORT_WINDOW_CAP));
    }

    /// A span inside a big VMA gets its own bucket, never the whole VMA.
    #[test]
    fn big_vma_is_capped_to_bucket() {
        // Bucket 4 of the 16 GiB VMA, offset well inside it at any window size.
        let ptr = VMA_BASE + (4 * HOST_IMPORT_WINDOW_CAP) as usize + 0x345_6000;
        let (base, len) =
            capped_import_window(VMA_BASE, VMA_LEN, ptr, ptr as u64 + 0x80_0000, ALIGN);
        assert_eq!(base as u64, VMA_BASE as u64 + 4 * HOST_IMPORT_WINDOW_CAP);
        assert_eq!(len, HOST_IMPORT_WINDOW_CAP);
        assert!(base <= ptr);
        assert!(ptr as u64 + 0x80_0000 <= base as u64 + len);
    }

    /// A span crossing the bucket edge extends the window to cover it — the
    /// caller requires full coverage of [ptr, end).
    #[test]
    fn span_crossing_bucket_edge_extends() {
        let ptr = VMA_BASE + (HOST_IMPORT_WINDOW_CAP as usize) - 0x2000;
        let end = ptr as u64 + 0x8000; // 8 KiB before the edge into 24 KiB after
        let (base, len) = capped_import_window(VMA_BASE, VMA_LEN, ptr, end, ALIGN);
        assert_eq!(base, VMA_BASE);
        assert!(end <= base as u64 + len);
        assert_eq!(len % ALIGN, 0);
    }

    /// The last bucket clamps to the VMA end — never import past the block.
    #[test]
    fn last_bucket_clamps_to_vma_end() {
        let vma_len = 15 * HOST_IMPORT_WINDOW_CAP + 0x10_0000; // non-multiple
        let ptr = VMA_BASE + (15 * HOST_IMPORT_WINDOW_CAP) as usize + 0x8000;
        let (base, len) = capped_import_window(VMA_BASE, vma_len, ptr, ptr as u64 + 0x1000, ALIGN);
        assert_eq!(base as u64, VMA_BASE as u64 + 15 * HOST_IMPORT_WINDOW_CAP);
        assert_eq!(base as u64 + len, VMA_BASE as u64 + vma_len);
    }

    /// Window base and length stay aligned for page-aligned VMAs.
    #[test]
    fn window_stays_aligned() {
        for off in [0usize, 0x1000, 0x3fff_f000, 0x2_0000_0000] {
            let ptr = VMA_BASE + off;
            let (base, len) =
                capped_import_window(VMA_BASE, VMA_LEN, ptr, ptr as u64 + 0x4000, ALIGN);
            assert_eq!(base as u64 % ALIGN, 0);
            assert_eq!(len % ALIGN, 0);
            assert!(len <= HOST_IMPORT_WINDOW_CAP + 0x4000);
        }
    }
}

#[cfg(test)]
mod vma_bounds_tests {
    use super::vma_bounds_in;

    const MAPS: &str = "\
00400000-00452000 r-xp 00000000 08:02 173521 /usr/bin/dbus-daemon
7f3a00000000-7f3c00000000 rw-p 00000000 00:00 0
7f3c10000000-7f3c10021000 ---p 00000000 00:00 0
7ffc04b1c000-7ffc04b3d000 rw-p 00000000 00:00 0 [stack]";

    /// A pointer inside the big anonymous rw mapping (the QEMU RAMBlock
    /// shape) must resolve to that mapping's exact bounds.
    #[test]
    fn pointer_inside_ram_block_resolves() {
        let ptr = 0x7f3a_8000_0000usize;
        assert_eq!(
            vma_bounds_in(MAPS, ptr),
            Some((0x7f3a_0000_0000, 0x2_0000_0000))
        );
    }

    /// A pointer in an unmapped hole must not resolve — importing unmapped
    /// VA would be undefined behavior at allocate time.
    #[test]
    fn pointer_in_hole_is_none() {
        assert_eq!(vma_bounds_in(MAPS, 0x7f3c_0800_0000), None);
    }

    /// A non-readable guard mapping must refuse (the GPU cannot DMA from it).
    #[test]
    fn non_readable_vma_refused() {
        assert_eq!(vma_bounds_in(MAPS, 0x7f3c_1000_0000), None);
    }

    /// Boundary conditions: first byte in, last byte in, first byte past.
    #[test]
    fn bounds_are_half_open() {
        assert!(vma_bounds_in(MAPS, 0x7f3a_0000_0000).is_some());
        assert!(vma_bounds_in(MAPS, 0x7f3b_ffff_ffff).is_some());
        assert!(vma_bounds_in(MAPS, 0x7f3c_0000_0000).is_none());
    }
}

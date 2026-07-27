impl ResourcePools {
    /// Resolve a guest-RAM host span to a cached VK_EXT_external_memory_host
    /// import, creating the import on first sight. Returns the TRANSFER_SRC
    /// buffer bound over the region and the span's byte offset within it.
    ///
    /// Import candidates, in order: a capped window of the containing VMA
    /// (see [`host_import::HOST_IMPORT_WINDOW_CAP`] — never the whole QEMU RAMBlock;
    /// importing pins host pages for DMA, so a whole-block import would pin
    /// all guest RAM for the VM lifetime) then the aligned span itself
    /// (fallback when the driver rejects the window import). `None` = no
    /// import possible; callers fall back to the CPU byte path.
    /// One flood latch per distinct cause.
    ///
    /// Keyed on the variant rather than on "have we logged a host-import failure",
    /// because a shared latch would report whichever cause happened to fire first
    /// and silence the rest — the precise defect that made 190 declines
    /// undiagnosable. The `match` is exhaustive, so a new cause cannot be added
    /// without deciding where its latch lives.
    fn host_import_first_time(&mut self, reason: HostImportDecline) -> bool {
        let seen = match reason {
            HostImportDecline::RegionCount => &mut self.host_import_count_cap_logged,
            HostImportDecline::TotalBytes => &mut self.host_import_byte_cap_logged,
            HostImportDecline::ZeroLength => &mut self.host_import_zero_len_logged,
            HostImportDecline::ExtensionAbsent => &mut self.host_import_no_ext_logged,
            HostImportDecline::PointerMisaligned { .. }
            | HostImportDecline::SizeMisaligned { .. }
            | HostImportDecline::RangeOverflow { .. }
            | HostImportDecline::NoValidWindow { .. } => {
                unreachable!("non-latched import declines are emitted at their attempted site")
            }
        };
        let first = !*seen;
        *seen = true;
        first
    }

    pub(crate) unsafe fn host_import_resolve(
        &mut self,
        ctx: &DeviceContext,
        ptr: usize,
        len: u64,
    ) -> Option<(vk::Buffer, u64)> {
        // Both of these used to be one unlogged `return None`. They are latched,
        // not per-call: a zero-length span recurs per present and an absent
        // extension refuses every span for the device's lifetime, so an unlatched
        // line would flood. The latch is per cause, so one does not mask the other.
        if len == 0 {
            if self.host_import_first_time(HostImportDecline::ZeroLength) {
                crate::observe::Emit::decline("host_import_fail", &HostImportDecline::ZeroLength)
                    .field("ptr", format!("{ptr:#x}"))
                    .fail();
            }
            return None;
        }
        if ctx.ext_external_memory_host.is_none() {
            if self.host_import_first_time(HostImportDecline::ExtensionAbsent) {
                crate::observe::Emit::decline(
                    "host_import_fail",
                    &HostImportDecline::ExtensionAbsent,
                )
                .field("len", format!("{len:#x}"))
                .fail();
            }
            return None;
        }
        let Some(end) = (ptr as u64).checked_add(len) else {
            let reason = HostImportDecline::RangeOverflow { host_ptr: ptr, len };
            crate::observe::Emit::decline("host_import_fail", &reason).fail_once(0);
            return None;
        };
        self.host_import_clock = self.host_import_clock.wrapping_add(1);
        let now = self.host_import_clock;
        if let Some(r) = self
            .host_imports
            .iter_mut()
            .find(|r| r.base as u64 <= ptr as u64 && end <= r.base as u64 + r.len)
        {
            // Stamp on hit: eviction reads these, so a window the guest keeps
            // touching must not look as cold as one abandoned after boot.
            r.last_used = now;
            return Some((r.buffer, ptr as u64 - r.base as u64));
        }
        let align = ctx.min_imported_host_pointer_alignment.max(1);
        let mut candidates: Vec<(usize, u64)> = Vec::new();
        if let Some((vma_base, vma_len)) = vma_bounds(ptr) {
            candidates.push(capped_import_window(
                vma_base,
                vma_len as u64,
                ptr,
                end,
                align,
            ));
        }
        let span_base = ((ptr as u64) / align * align) as usize;
        let span_len = (end - span_base as u64).div_ceil(align) * align;
        candidates.push((span_base, span_len));
        let mut last_error = None;
        for (base, region_len) in candidates {
            if !(base as u64).is_multiple_of(align)
                || region_len % align != 0
                || base > ptr
                || end > base as u64 + region_len
            {
                continue;
            }
            let imported_bytes = self
                .host_imports
                .iter()
                .fold(0u64, |total, region| total.saturating_add(region.len));
            if let Err(reason) =
                host_import_budget(self.host_imports.len(), imported_bytes, region_len)
            {
                // The budget binds a working set, not a lifetime: make room by
                // dropping the coldest windows rather than refusing every new
                // bucket for the rest of the session.
                let stamps: Vec<(u64, u64)> = self
                    .host_imports
                    .iter()
                    .map(|region| (region.len, region.last_used))
                    .collect();
                let Some(victims) = host_import_eviction_plan(&stamps, region_len) else {
                    if self.host_import_first_time(reason) {
                        crate::observe::Emit::decline("host_import_fail", &reason)
                            .field("regions", self.host_imports.len())
                            .field("imported_bytes", format!("{imported_bytes:#x}"))
                            .field("candidate_bytes", format!("{region_len:#x}"))
                            .field("cap", format!("{HOST_IMPORT_TOTAL_BYTE_CAP:#x}"))
                            .fail();
                    }
                    return None;
                };
                // Descending so each removal leaves the lower indices valid.
                let mut ordered = victims;
                ordered.sort_unstable_by(|a, b| b.cmp(a));
                for index in ordered {
                    let region = self.host_imports.remove(index);
                    crate::observe::off(format!(
                        "host_import_evict base={:#x} len={:#x} last_used={} regions={}",
                        region.base,
                        region.len,
                        region.last_used,
                        self.host_imports.len()
                    ));
                    self.dispose(
                        &ctx.device,
                        DeferredHandle::HostImport {
                            buffer: region.buffer,
                            memory: region.memory,
                        },
                    );
                }
            }
            let memory = match ctx.import_host_ptr(base as *mut std::ffi::c_void, region_len) {
                Ok(memory) => memory,
                Err(error) => {
                    last_error = Some(error);
                    continue;
                }
            };
            let info = vk::BufferCreateInfo::default()
                .size(region_len)
                .usage(vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST)
                .sharing_mode(vk::SharingMode::EXCLUSIVE);
            let buffer = match ctx.device.create_buffer(&info, None) {
                Ok(b) => b,
                Err(result) => {
                    ctx.device.free_memory(memory, None);
                    last_error = Some(DrawError::VkCall(VkCall::new(
                        VkOp::PoolsHostImportCreateBuffer,
                        result,
                    )));
                    continue;
                }
            };
            if let Err(result) = ctx.device.bind_buffer_memory(buffer, memory, 0) {
                ctx.device.destroy_buffer(buffer, None);
                ctx.device.free_memory(memory, None);
                last_error = Some(DrawError::VkCall(VkCall::new(
                    VkOp::PoolsHostImportBindBuffer,
                    result,
                )));
                continue;
            }
            crate::observe::off(format!(
                "host_import_region base={base:#x} len={region_len:#x} regions={}",
                self.host_imports.len() + 1
            ));
            self.host_imports.push(HostImportRegion {
                base,
                len: region_len,
                memory,
                buffer,
                last_used: now,
            });
            let r = self.host_imports.last().unwrap();
            return Some((r.buffer, ptr as u64 - r.base as u64));
        }
        let error = terminal_host_import_error(last_error, ptr, len, align);
        crate::observe::Emit::decline("host_import_fail", &error)
            .field("ptr", format!("{ptr:#x}"))
            .field("len", format!("{len:#x}"))
            .fail_once(0);
        None
    }

    pub(crate) unsafe fn destroy_all(&mut self, device: &ash::Device) {
        // An open (never-submitted) batch dies with the pool: its CB belongs
        // to cmd_pool (destroyed below) and its dsets to desc_pool; the
        // accumulated transients are already in the live lists.
        self.open_batch = None;
        // Best-effort quiesce: wait every in-flight fence so no CB references
        // what we are about to destroy. On device loss the waits fail — the
        // teardown proceeds regardless, matching the recreate path.
        for slot in &self.slots {
            if slot.pending.is_some() {
                if let Err(result) = device.wait_for_fences(&[slot.fence], true, FENCE_TIMEOUT_NS) {
                    let decline =
                        DrawError::VkCall(VkCall::new(VkOp::PoolsWaitFencesDestroy, result));
                    crate::observe::Emit::decline("vk_pools_destroy", &decline).fail_once(0);
                }
            }
        }
        for slot in &mut self.slots {
            if let Some(pending) = slot.pending.take() {
                // The descriptor pool is destroyed below (frees every set);
                // move the owed transients into the live lists so the drains
                // below destroy them.
                self.staging_live.extend(pending.staging);
                self.readback_multi_live.extend(pending.readback);
                self.sampled_live.extend(pending.sampled);
                self.storage_image_live.extend(pending.storage_images);
            }
        }
        self.in_flight = 0;
        self.drain_graveyard(device);
        for list in self.staging_free.values_mut() {
            for s in list.drain(..) {
                device.destroy_buffer(s.buffer, None);
                device.free_memory(s.memory, None);
            }
        }
        for s in self.staging_live.drain(..) {
            device.destroy_buffer(s.buffer, None);
            device.free_memory(s.memory, None);
        }
        for list in self.readback_free.values_mut() {
            for s in list.drain(..) {
                device.destroy_buffer(s.buffer, None);
                device.free_memory(s.memory, None);
            }
        }
        if let Some(s) = self.readback_live.take() {
            device.destroy_buffer(s.buffer, None);
            device.free_memory(s.memory, None);
        }
        for s in self.readback_multi_live.drain(..) {
            device.destroy_buffer(s.buffer, None);
            device.free_memory(s.memory, None);
        }
        // Sampled / target / registry images are slab-backed: destroy the image
        // + view handles here, but their memory belongs to shared blocks freed
        // once by `self.slab.destroy_all(device)` at the end — never a per-image
        // `vkFreeMemory` (that would double-free a block many images share).
        for list in self.sampled_free.values_mut() {
            for s in list.drain(..) {
                device.destroy_image_view(s.view, None);
                device.destroy_image(s.image, None);
            }
        }
        for list in self.target_free.values_mut() {
            for img in list.drain(..) {
                device.destroy_image_view(img.view, None);
                device.destroy_image(img.image, None);
            }
        }
        for s in self.sampled_live.drain(..) {
            device.destroy_image_view(s.view, None);
            device.destroy_image(s.image, None);
        }
        for s in self.sampled_cache.drain(..) {
            device.destroy_image_view(s.slot.view, None);
            device.destroy_image(s.slot.image, None);
        }
        self.sampled_cache_bytes = 0;
        for list in self.storage_image_free.values_mut() {
            for s in list.drain(..) {
                device.destroy_image_view(s.view, None);
                device.destroy_image(s.image, None);
                device.free_memory(s.memory, None);
            }
        }
        for s in self.storage_image_live.drain(..) {
            device.destroy_image_view(s.view, None);
            device.destroy_image(s.image, None);
            device.free_memory(s.memory, None);
        }
        for (_, resident) in self.compute_storage_registry.drain() {
            device.destroy_image_view(resident.slot.view, None);
            device.destroy_image(resident.slot.image, None);
            device.free_memory(resident.slot.memory, None);
        }
        self.compute_storage_order.clear();
        for (_, t) in self.targets.drain() {
            device.destroy_framebuffer(t.framebuffer, None);
            device.destroy_image_view(t.view, None);
            device.destroy_image(t.image, None);
        }
        self.target_order.clear();
        for (_, t) in self.registry.drain() {
            device.destroy_framebuffer(t.framebuffer, None);
            device.destroy_image_view(t.view, None);
            device.destroy_image(t.image, None);
        }
        self.registry_order.clear();
        // Free every slab block now that all slab-backed images are destroyed.
        self.slab.destroy_all(device);
        // Host-import regions: guest RAM outlives the device; only the Vulkan
        // handles are ours to free (in-flight fences were waited above).
        for r in self.host_imports.drain(..) {
            device.destroy_buffer(r.buffer, None);
            device.free_memory(r.memory, None);
        }
        // Prefetch slots own dedicated fences + host-coherent buffers; their CBs
        // are freed with `cmd_pool` below. Destroy them before the pool so the
        // best-effort in-flight fence waits happen while the device is live.
        // Same rule for the stats-reduction pool: dedicated fences, a
        // persistently-mapped buffer, a private descriptor pool and a
        // sampler. Its CBs come from `cmd_pool`, so hand it over here and
        // destroy before the pool itself goes.
        self.stats_reduce.destroy_all(device, self.cmd_pool);
        self.host_scatter.destroy_all(device, self.cmd_pool);
        for slot in self.slots.drain(..) {
            device.destroy_fence(slot.fence, None);
        }
        self.cur = 0;
        self.desc_arena.destroy(device);
        if self.cmd_pool != vk::CommandPool::null() {
            device.destroy_command_pool(self.cmd_pool, None);
            self.cmd_pool = vk::CommandPool::null();
        }
        self.initialized = false;
    }
}


#[cfg(test)]
mod host_import_split_tests {
    use super::host_import_budget;

    #[test]
    fn small_arm_aliases_fit_under_shared_byte_ceiling() {
        assert_eq!(host_import_budget(128, 64 << 20, 16 << 10), Ok(()));
    }
}

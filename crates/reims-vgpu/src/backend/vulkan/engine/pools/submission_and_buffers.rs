impl ResourcePools {
    pub(crate) fn guest_reset_counts(&self) -> (usize, usize, usize, usize) {
        let sampled = self.sampled_live.len()
            + self.sampled_free.values().map(Vec::len).sum::<usize>()
            + self.sampled_cache.len();
        let storage = self.storage_image_live.len()
            + self
                .storage_image_free
                .values()
                .map(Vec::len)
                .sum::<usize>()
            + self.compute_storage_registry.len();
        (self.registry.len(), self.targets.len(), sampled, storage)
    }

    pub(crate) fn new() -> Self {
        Self {
            staging_free: HashMap::new(),
            staging_live: Vec::new(),
            targets: HashMap::new(),
            target_order: Vec::new(),
            readback_free: HashMap::new(),
            readback_live: None,
            readback_multi_live: Vec::new(),
            sampled_free: HashMap::new(),
            sampled_live: Vec::new(),
            sampled_cache: Vec::new(),
            sampled_cache_bytes: 0,
            storage_image_free: HashMap::new(),
            storage_image_live: Vec::new(),
            compute_storage_registry: HashMap::new(),
            compute_storage_order: VecDeque::new(),
            registry: HashMap::new(),
            registry_order: VecDeque::new(),
            idle_clock_ms: 0,
            last_drain_ms: 0,
            settled_drain_passes: 0,
            cmd_pool: vk::CommandPool::null(),
            desc_arena: DescriptorArena::empty(),
            slots: Vec::new(),
            cur: 0,
            in_flight: 0,
            graveyard: Vec::new(),
            sampled_free_hits: 0,
            sampled_free_allocs: 0,
            sampled_recycle_admits: 0,
            sampled_recycle_cap_drops: 0,
            target_free: HashMap::new(),
            target_free_hits: 0,
            target_free_allocs: 0,
            target_recycle_admits: 0,
            target_recycle_cap_drops: 0,
            storage_recycle_admits: 0,
            storage_recycle_cap_drops: 0,
            host_imports: Vec::new(),
            host_import_clock: 0,
            host_import_count_cap_logged: false,
            host_import_zero_len_logged: false,
            host_import_no_ext_logged: false,
            host_import_byte_cap_logged: false,
            stats_reduce: stats_reduce::StatsReducePool::new(),
            host_scatter: host_scatter::HostScatterPool::default(),
            open_batch: None,
            slab: super::slab::SlabPool::new(),
            initialized: false,
        }
    }
    /// Arm a GPU-side present-proxy stats reduction for `(identity, seq)`.
    ///
    /// This is the zero-copy oracle: it dispatches the reduction kernel over the
    /// resident and returns immediately, so the proxies get their measurements
    /// without any full-frame GPU→CPU copy. Returns whether it armed; `false`
    /// means the caller simply has no stats for this present (never a fallback
    /// to reading the frame back).
    ///
    /// Requires the resident to be content-ready **and** BGRA: the kernel reads
    /// `.rgb` as colour channels and packs `px0` in memory order, so an RGBA
    /// resident would report channel-swapped `px0`.
    pub(crate) unsafe fn arm_present_stats(
        &mut self,
        ctx: &DeviceContext,
        counters: &EngineCounters,
        caches: &mut ObjectCaches,
        identity: &TargetIdentity,
        seq: u64,
    ) -> bool {
        // Same ordering rule as the prefetch pool: this submits on its own CB,
        // bypassing begin_entry, so flush any open draw batch first or the
        // dispatch would be queued ahead of the draws producing this content.
        if let Err(error) = self.batch_flush(ctx, counters) {
            crate::observe::Emit::decline("stats_reduce", &error).fail_once(0);
            return false;
        }
        let (image, view, old_layout, width, height, generation) = {
            let slot = match self.registry.get(identity) {
                Some(s) if s.content_ready && s.bgra => s,
                _ => return false,
            };
            (
                slot.image,
                slot.view,
                slot.layout,
                slot.width,
                slot.height,
                slot.generation,
            )
        };
        // Pipeline comes from the shared, content-keyed caches — a host-authored
        // kernel is just another SPIR-V digest to them.
        let words = &super::present_stats_spv::PRESENT_STATS_SPIRV;
        let (_digest, module) = match caches.get_or_create_shader(ctx, words, counters, self) {
            Ok(value) => value,
            Err(error) => {
                let setup = PresentStatsSetup::Shader;
                present_stats_setup_decline(setup, &error).fail_once(setup.discriminant());
                return false;
            }
        };
        let layout_key = LayoutKey {
            bindings: vec![
                BindingSig {
                    binding: 0,
                    ty: vk::DescriptorType::SAMPLED_IMAGE.as_raw() as u32,
                    stages: vk::ShaderStageFlags::COMPUTE.as_raw(),
                },
                BindingSig {
                    binding: 1,
                    ty: vk::DescriptorType::SAMPLER.as_raw() as u32,
                    stages: vk::ShaderStageFlags::COMPUTE.as_raw(),
                },
                BindingSig {
                    binding: 2,
                    ty: vk::DescriptorType::STORAGE_BUFFER.as_raw() as u32,
                    stages: vk::ShaderStageFlags::COMPUTE.as_raw(),
                },
            ],
        };
        let (set_layout, pipeline_layout) =
            match caches.get_or_create_layout(ctx, &layout_key, counters, self) {
                Ok(value) => value,
                Err(error) => {
                    let setup = PresentStatsSetup::Layout;
                    present_stats_setup_decline(setup, &error).fail_once(setup.discriminant());
                    return false;
                }
            };
        let pkey = ComputePipelineKey {
            spirv: _digest,
            entry: "main".to_string(),
            layout: layout_key,
        };
        let pipeline = match caches.get_or_create_compute_pipeline(
            ctx,
            &pkey,
            module,
            pipeline_layout,
            counters,
            self,
        ) {
            Ok(value) => value,
            Err(error) => {
                let setup = PresentStatsSetup::Pipeline;
                present_stats_setup_decline(setup, &error).fail_once(setup.discriminant());
                return false;
            }
        };
        let cmd_pool = self.cmd_pool;
        let armed = self.stats_reduce.arm(
            ctx,
            counters,
            cmd_pool,
            pipeline,
            pipeline_layout,
            set_layout,
            identity,
            generation,
            seq,
            image,
            view,
            old_layout,
            width,
            height,
        );
        if armed {
            // The dispatch leaves the resident in SHADER_READ_ONLY_OPTIMAL; the
            // pool cannot reach the registry, so record it here.
            self.registry_set_layout(identity, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);
        }
        armed
    }

    /// Scatter a resident present straight into imported guest pages, with no
    /// full-frame CPU copy anywhere in the path.
    ///
    /// `Ok(stats)` means the guest pages hold the frame and `stats` carries
    /// the content measurement (from the GPU reduction, when `measure_content`).
    /// `Err` names the exact setup/record/submit/state refusal. The caller fails
    /// the Store rather than publishing a possibly partial guest-page write;
    /// hosts without `VK_EXT_external_memory_host` select the CPU path before
    /// entering this rail.
    #[allow(
        clippy::too_many_arguments,
        reason = "scatter presentation mirrors the source image and guest run layout"
    )]
    pub(crate) unsafe fn present_scatter_gpu(
        &mut self,
        ctx: &DeviceContext,
        counters: &EngineCounters,
        caches: &mut ObjectCaches,
        identity: &TargetIdentity,
        runs: &[host_scatter::ScatterRun],
        spans: &[host_scatter::ScatterSpan],
        measure_content: bool,
        stats_seq: u64,
    ) -> Result<Option<super::content_stats::Color8ContentStats>, DrawError> {
        if spans.is_empty() {
            return Err(DrawError::Present(
                super::reason::HostPresentDecline::RunsEmpty,
            ));
        }
        if ctx.ext_external_memory_host.is_none() {
            return Err(DrawError::Unsupported(
                super::reason::DrawReason::PresentHostPtrImportUnavailable,
            ));
        }
        // Same ordering rule as the prefetch/stats pools: this submits on its
        // own CB, so flush any open draw batch or the copy would be queued
        // ahead of the draws producing this frame.
        if let Err(error) = self.batch_flush(ctx, counters) {
            self.host_scatter.fallback_submit = self.host_scatter.fallback_submit.wrapping_add(1);
            return Err(error);
        }

        // Resolve every run up front: a partial scatter would leave the guest
        // surface torn, so this is all-or-nothing.
        let mut bufs: Vec<(vk::Buffer, u64)> = Vec::with_capacity(runs.len());
        for run in runs {
            match self.host_import_resolve(ctx, run.host_ptr, run.ptr_len) {
                Some(b) => bufs.push(b),
                None => {
                    self.host_scatter.fallback_unresolved =
                        self.host_scatter.fallback_unresolved.wrapping_add(1);
                    // `host_import_resolve` already emitted the exact typed
                    // leaf (extension, cap, alignment, range, or Vulkan call).
                    // This line is the aggregate failure counter, so it names
                    // the class without minting a second coarse reason.
                    crate::observe::off(format!(
                        "store_scatter_fallback class=run_unimportable runs={} ptr_len={}",
                        runs.len(),
                        run.ptr_len
                    ));
                    return Err(DrawError::Unsupported(
                        super::reason::DrawReason::PresentHostImportResolve,
                    ));
                }
            }
        }

        let (image, old_layout, generation, width, height) = {
            let slot = self.registry.get(identity).ok_or_else(|| {
                DrawError::Facade(
                    super::facade_decline::EngineFacadeDecline::ScatterPresentUnknownIdentity {
                        identity: identity.clone(),
                    },
                )
            })?;
            if !slot.content_ready {
                return Err(DrawError::Facade(
                    super::facade_decline::EngineFacadeDecline::ScatterPresentNotReady {
                        identity: identity.clone(),
                    },
                ));
            }
            if !slot.bgra {
                return Err(DrawError::Unsupported(
                    super::reason::DrawReason::PresentScatterResidentNotBgra,
                ));
            }
            (
                slot.image,
                slot.layout,
                slot.generation,
                slot.width,
                slot.height,
            )
        };

        let regions = resolve_scatter_regions(runs, spans, &bufs, width, height)
            .map_err(DrawError::Present)?;

        // Arm the stats reduction BEFORE the scatter: it reads the resident as
        // SHADER_READ_ONLY_OPTIMAL, and the scatter then transitions to
        // TRANSFER_SRC_OPTIMAL. Ordering them the other way would make the
        // dispatch read an image the copy had already re-laid-out.
        let armed =
            measure_content && self.arm_present_stats(ctx, counters, caches, identity, stats_seq);

        let cmd_pool = self.cmd_pool;
        if let Err(error) = self
            .host_scatter
            .scatter(ctx, cmd_pool, image, old_layout, &regions)
        {
            self.host_scatter.fallback_submit = self.host_scatter.fallback_submit.wrapping_add(1);
            // The copy may have partially landed, so the caller must fail the
            // Store rather than publish those guest pages as complete.
            return Err(error);
        }
        self.registry_set_layout(identity, vk::ImageLayout::TRANSFER_SRC_OPTIMAL);
        self.host_scatter.gpu_stores = self.host_scatter.gpu_stores.wrapping_add(1);

        let content = if armed {
            self.stats_reduce
                .consume_blocking(ctx, identity, generation, stats_seq)
                .map(Into::into)
        } else {
            None
        };
        Ok(content)
    }

    /// `(gpu_stores, fallback_unresolved, fallback_submit)` for the always-on
    /// store-path proxy line.
    pub(crate) fn host_scatter_counters(&self) -> (u64, u64, u64) {
        (
            self.host_scatter.gpu_stores,
            self.host_scatter.fallback_unresolved,
            self.host_scatter.fallback_submit,
        )
    }

    /// Advance the wall-clock idle-drain clock to `now_ms`, keep the presented
    /// target alive (`display`), and — if an idle pass is due — select a bounded
    /// set of aged non-pinned residents to reclaim. Pure — no GPU work — so the
    /// aging + throttle logic is unit-testable without a device.
    ///
    /// Returns `None` when no pass is due (before the first age window, or inside
    /// the throttle interval); `Some(victims)` (possibly empty) when a pass fired.
    /// The `Some`/`None` distinction is load-bearing: the caller also trims the
    /// recycle pools on a fired pass, and an empty registry with a full recycle
    /// pool must still trim.
    ///
    /// `display` is the currently-presented target's identity: it is resolved via
    /// `registry_get` (a read that does not stamp), so at true idle — where the
    /// poll heartbeat still ticks this clock but no publish re-touches the frame —
    /// it would otherwise age out from under the display. Stamping it here every
    /// call makes it un-ageable while it is on screen.
    fn plan_idle_drain(
        &mut self,
        now_ms: u64,
        display: Option<&TargetIdentity>,
    ) -> Option<Vec<TargetIdentity>> {
        if now_ms > self.idle_clock_ms {
            self.idle_clock_ms = now_ms;
        }
        let now = self.idle_clock_ms;
        if let Some(id) = display {
            if let Some(slot) = self.registry.get_mut(id) {
                slot.last_touch_ms = now;
            }
        }
        if now < IDLE_TARGET_AGE_MS
            || now.saturating_sub(self.last_drain_ms) < IDLE_DRAIN_INTERVAL_MS
        {
            return None;
        }
        self.last_drain_ms = now;
        let cutoff = now - IDLE_TARGET_AGE_MS;
        let mut victims = Vec::new();
        for k in &self.registry_order {
            if victims.len() >= IDLE_TARGET_DRAIN_MAX_PER_CALL {
                break;
            }
            if self
                .registry
                .get(k)
                .is_some_and(|s| s.pin_count == 0 && s.last_touch_ms <= cutoff)
            {
                victims.push(k.clone());
            }
        }
        Some(victims)
    }

    /// Destroy up to `max` images from the image recycle pools (`sampled_free`
    /// then `target_free`) — freeing their slab sub-allocations — and up to `max`
    /// buffers from the HOST_VISIBLE `staging_free`/`readback_free` pools. Called
    /// only on a fired idle pass. Returns the total count destroyed. Terminal
    /// destroy (not re-recycle): at idle these are pure retained memory.
    ///
    /// The buffer pools matter as much as the image pools for "least host memory":
    /// they are never re-evaluated on the hot path, so at idle they hold a whole
    /// video session's upload working set forever (measured `staging_mb=177` +
    /// `readback_mb=61` frozen for 30 s+ after the tab closed). On a discrete GPU
    /// that is system RAM; on an iGPU it is shared guest RAM (portability target).
    /// Trimming here (gradual, like the image pools; they refill a-few-per-frame
    /// when uploads resume) returns it once the session ends without any hot-path
    /// re-alloc churn.
    ///
    /// `trim_buffers` gates ONLY the HOST_VISIBLE buffer pools: they re-alloc via
    /// full `vkAllocateMemory` on the upload hot path, so we only free them once
    /// idle has *settled* (see [`SETTLED_PASSES_FOR_BUFFER_TRIM`]). The image
    /// pools always trim — they refill via cheap slab suballocation.
    unsafe fn trim_recycle_pools(
        &mut self,
        device: &ash::Device,
        max: usize,
        trim_buffers: bool,
    ) -> usize {
        let mut trimmed = 0;
        while trimmed < max {
            let Some(slot) = pop_any_pool_entry(&mut self.sampled_free) else {
                break;
            };
            self.destroy_deferred_handle(device, DeferredHandle::RecycleSampled(slot));
            trimmed += 1;
        }
        while trimmed < max {
            let Some(img) = pop_any_pool_entry(&mut self.target_free) else {
                break;
            };
            self.destroy_deferred_handle(
                device,
                DeferredHandle::Image {
                    image: img.image,
                    view: img.view,
                    memory: img.memory,
                },
            );
            trimmed += 1;
        }
        // Buffer pools get their own budget so image trimming never starves them,
        // and only trim once idle has settled (they re-alloc via full
        // vkAllocateMemory on the upload hot path — trimming mid-video hitches it).
        let mut buf_trimmed = 0;
        if trim_buffers {
            while buf_trimmed < max {
                let Some(slot) = pop_any_pool_entry(&mut self.staging_free) else {
                    break;
                };
                device.destroy_buffer(slot.buffer, None);
                device.free_memory(slot.memory, None);
                buf_trimmed += 1;
            }
            while buf_trimmed < max {
                let Some(slot) = pop_any_pool_entry(&mut self.readback_free) else {
                    break;
                };
                device.destroy_buffer(slot.buffer, None);
                device.free_memory(slot.memory, None);
                buf_trimmed += 1;
            }
            // The standalone (non-slab) compute-storage recycle pool re-allocs via
            // a full `vkAllocateMemory` on the next dispatch — like the buffer
            // pools above and unlike the slab-backed image pools (cheap
            // suballocation refill) — so gate its trim on settled idle too: a brief
            // pause between compute dispatches must not steal a pooled storage
            // image and spike the next dispatch with a fresh allocation. Its own
            // budget so it never starves the buffer trim.
            let mut storage_trimmed = 0;
            while storage_trimmed < max {
                let Some(slot) = pop_any_pool_entry(&mut self.storage_image_free) else {
                    break;
                };
                self.destroy_deferred_handle(
                    device,
                    DeferredHandle::Image {
                        image: slot.image,
                        view: slot.view,
                        memory: slot.memory,
                    },
                );
                storage_trimmed += 1;
            }
            buf_trimmed += storage_trimmed;
        }
        trimmed + buf_trimmed
    }

    /// Reclaim up to `max` sampled-cache entries whose last use is at least
    /// `IDLE_TARGET_AGE_MS` behind the idle clock (`cutoff`) — the upload-side
    /// analogue of the resident-target idle drain. At idle a video session's
    /// retained frame textures (the ≤128 MiB `sampled_cache`) are otherwise
    /// pinned until new content evicts them by LRU, which never happens on a
    /// static desktop, so they hold VRAM for the guest lifetime (measured
    /// `sampled=64` / resident +~96 MiB frozen after a video tab closed).
    /// Age-gating means an actively-sampled entry (hit every frame, so
    /// `last_touch_ms` fresh) is never trimmed — only entries idle past the age
    /// fall out, so a live session is undisturbed (no re-upload hitch). Terminal
    /// destroy via `dispose(Image)` (not `RecycleSampled`): an idle-stale frame
    /// will not be re-sampled, so caching it in `sampled_free` would defeat the
    /// drain; freeing the image releases its slab sub-range exactly like the
    /// target drain. In-flight-safe (deferred until the referencing CB retires).
    /// Returns the count trimmed, reflected in the always-on `sampled=` census.
    unsafe fn trim_aged_sampled_cache(
        &mut self,
        device: &ash::Device,
        cutoff: u64,
        max: usize,
    ) -> usize {
        let aged = self.take_aged_sampled_slots(cutoff, max);
        let trimmed = aged.len();
        for slot in aged {
            self.dispose(
                device,
                DeferredHandle::Image {
                    image: slot.image,
                    view: slot.view,
                    memory: slot.memory,
                },
            );
        }
        trimmed
    }

    /// Device-free half of [`Self::trim_aged_sampled_cache`]: remove up to `max`
    /// sampled-cache entries whose `last_touch_ms <= cutoff`, decrement the byte
    /// accounting, and return their slots for the caller to dispose. Split out so
    /// the age selection + byte bookkeeping is unit-testable without a GPU
    /// (mirrors [`Self::plan_idle_drain`] returning victims for the caller to
    /// dispose).
    fn take_aged_sampled_slots(&mut self, cutoff: u64, max: usize) -> Vec<SampledSlot> {
        let mut taken = Vec::new();
        let mut i = 0;
        while i < self.sampled_cache.len() && taken.len() < max {
            if self.sampled_cache[i].last_touch_ms <= cutoff {
                let evicted = self.sampled_cache.remove(i);
                self.sampled_cache_bytes =
                    self.sampled_cache_bytes.saturating_sub(evicted.content_len);
                taken.push(evicted.slot);
                // `remove(i)` shifted the next entry into slot `i`; do not advance.
            } else {
                i += 1;
            }
        }
        taken
    }

    /// Reclaim up to `max` compute-storage residents whose last use is at least
    /// `IDLE_TARGET_AGE_MS` behind the idle clock (`cutoff`) — the compute analogue
    /// of the resident-target idle drain. Each resident is a standalone (non-slab)
    /// `VkDeviceMemory`, so a settled compute-heavy session (blur passes, a decode
    /// storage image) otherwise pins up to `COMPUTE_STORAGE_REGISTRY_CAP` whole
    /// allocations until an LRU eviction that never comes on a static desktop.
    /// Age-gating leaves an actively-dispatched resident (touched every pass, so
    /// `last_touch_ms` fresh) untouched and skips pinned residents (deferred
    /// writeback, only-copy-on-GPU) exactly like the LRU sweep. Terminal destroy
    /// via `dispose(Image)`; in-flight-safe (deferred until the referencing CB
    /// retires). Returns the count trimmed, reflected in the `st_res` census.
    unsafe fn trim_aged_compute_storage(
        &mut self,
        device: &ash::Device,
        cutoff: u64,
        max: usize,
    ) -> usize {
        let aged = self.take_aged_storage_residents(cutoff, max);
        let trimmed = aged.len();
        for slot in aged {
            self.dispose(
                device,
                DeferredHandle::Image {
                    image: slot.image,
                    view: slot.view,
                    memory: slot.memory,
                },
            );
        }
        trimmed
    }

    /// Device-free half of [`Self::trim_aged_compute_storage`]: remove up to `max`
    /// non-pinned compute-storage residents whose `last_touch_ms <= cutoff` from
    /// the registry and its LRU order, and return their slots for the caller to
    /// dispose. Split out so the age/pin selection is unit-testable without a GPU
    /// (mirrors [`Self::take_aged_sampled_slots`]).
    fn take_aged_storage_residents(&mut self, cutoff: u64, max: usize) -> Vec<StorageImageSlot> {
        let victims: Vec<ComputeStorageResidencyKey> = self
            .compute_storage_order
            .iter()
            .filter(|k| {
                self.compute_storage_registry
                    .get(*k)
                    .is_some_and(|r| !r.pinned && r.last_touch_ms <= cutoff)
            })
            .take(max)
            .copied()
            .collect();
        let mut taken = Vec::with_capacity(victims.len());
        for k in victims {
            self.compute_storage_order.retain(|entry| entry != &k);
            if let Some(resident) = self.compute_storage_registry.remove(&k) {
                taken.push(resident.slot);
            }
        }
        taken
    }

    /// Update the consecutive-settled-pass counter for a fired idle-drain pass
    /// that reclaimed `drained` registry victims, and return whether the
    /// HOST_VISIBLE buffer pools may be trimmed this pass. A pass that drained
    /// ≥1 victim means the working set is still churning (active video ages out
    /// old frame RTs each pass), so the counter resets. The buffer trim is only
    /// permitted after `SETTLED_PASSES_FOR_BUFFER_TRIM` consecutive zero-victim
    /// passes, so a single quiet pass mid-playback cannot steal a staging buffer
    /// and spike the next upload's latency with a full `vkAllocateMemory`.
    fn note_drain_settled(&mut self, drained: usize) -> bool {
        if drained == 0 {
            self.settled_drain_passes = self.settled_drain_passes.saturating_add(1);
        } else {
            self.settled_drain_passes = 0;
        }
        self.settled_drain_passes >= SETTLED_PASSES_FOR_BUFFER_TRIM
    }

    /// Advance the wall-clock idle-drain clock and reclaim a bounded number of
    /// non-pinned residents untouched for `IDLE_TARGET_AGE_MS`. Called from the
    /// poll heartbeat (ticks even when the guest stops publishing) and each
    /// publish. This is the mechanism that lets `REGISTRY_CAP` sit high enough to
    /// *absorb* a compositing burst (no eviction thrash) while still returning
    /// VRAM to the baseline working set once the burst ends — even on a static
    /// page where no further publishes occur: a burst's targets are all
    /// recently-touched (kept), and its stale leftovers age out ~2 s later. Pinned
    /// slots (deferred-write windows) are never drained — they leave via their own
    /// window lifecycle. Reclaimed images route through the same in-flight-safe
    /// recycle/dispose path as an LRU eviction; they are NOT counted as
    /// `target_evicts` (that counter is the thrash signal, and idle reclamation is
    /// not thrash). On a fired pass it also trims the recycle pools
    /// ([`Self::trim_recycle_pools`]) — at idle those are pure retained VRAM.
    /// Returns the count of registry residents drained this call, for the
    /// always-on census.
    pub(crate) unsafe fn advance_registry_touch_and_drain(
        &mut self,
        ctx: &DeviceContext,
        now_ms: u64,
        display: Option<&TargetIdentity>,
    ) -> usize {
        let Some(victims) = self.plan_idle_drain(now_ms, display) else {
            return 0;
        };
        let drained = victims.len();
        for k in victims {
            if let Some(pos) = self.registry_order.iter().position(|x| x == &k) {
                self.registry_order.remove(pos);
            }
            if let Some(old) = self.registry.remove(&k) {
                self.dispose(&ctx.device, DeferredHandle::Framebuffer(old.framebuffer));
                // Terminal DESTROY, not RecycleTarget: an idle-drained resident is
                // stale by `IDLE_TARGET_AGE_MS` — it is not being actively recycled
                // (that is the capacity-eviction path's job for a per-frame video
                // output), so caching it in `target_free` would defeat the whole
                // point of the drain. `RecycleTarget` keeps the image's slab
                // sub-allocation live, so a diverse burst (hundreds of distinct
                // geometries, ≤ `TARGET_FREE_CAP_PER_KEY` each) would cache every
                // image and no slab block could ever empty — measured VRAM stuck at
                // 1532 MiB after the registry drained to ~22. Freeing the image
                // releases its slab sub-range; when a block's last sub-allocation
                // leaves, the slab returns it to the driver (`SLAB_KEEP_EMPTY`), so
                // VRAM returns to the idle baseline.
                self.dispose(
                    &ctx.device,
                    DeferredHandle::Image {
                        image: old.image,
                        view: old.view,
                        memory: old.memory,
                    },
                );
            }
        }
        // Track settled-ness and decide whether this pass may trim the expensive
        // HOST_VISIBLE buffer pools.
        let trim_buffers = self.note_drain_settled(drained);
        // A fired pass also returns the recycle pools' idle VRAM to the driver.
        self.trim_recycle_pools(&ctx.device, IDLE_RECYCLE_TRIM_PER_PASS, trim_buffers);
        // …and ages out the sampled-content cache, the upload-side pool the
        // recycle/buffer trims above do not cover. A settled video session's
        // frame textures (≤128 MiB) are pinned until LRU eviction that never
        // comes on a static desktop; age-gating on the same `IDLE_TARGET_AGE_MS`
        // cutoff frees them once idle without touching an actively-sampled entry.
        let sampled_cutoff = self.idle_clock_ms.saturating_sub(IDLE_TARGET_AGE_MS);
        self.trim_aged_sampled_cache(&ctx.device, sampled_cutoff, IDLE_RECYCLE_TRIM_PER_PASS);
        // …and the compute-storage residents, the standalone-VkDeviceMemory pool the
        // render-registry / sampled-cache drains above do not cover. A settled
        // compute-heavy session's blur/decode storage images are pinned until an
        // LRU eviction that never comes on a static desktop; the same age cutoff
        // frees them once idle without touching an actively-dispatched resident.
        self.trim_aged_compute_storage(&ctx.device, sampled_cutoff, IDLE_RECYCLE_TRIM_PER_PASS);
        // …and releases the empty slab blocks the hot release path retains as a
        // churn buffer, which otherwise sit resident forever at idle (no image
        // release fires to trigger their free). Down to one spare.
        self.slab
            .trim_empty_blocks(&ctx.device, IDLE_SLAB_KEEP_EMPTY);
        drained
    }

    /// Count of registry residents NOT held by a deferred-write pin — the
    /// LRU-evictable (active) working set the `REGISTRY_CAP` bounds. Pinned slots
    /// are bounded separately (`RENDER_DEFERRED_WINDOW_CAP`) and excluded so a
    /// pinned burst cannot force the active set into eviction thrash.
    fn non_pinned_registry_len(&self) -> usize {
        let pinned = self
            .registry
            .values()
            .filter(|slot| slot.pin_count > 0)
            .count();
        self.registry_order.len().saturating_sub(pinned)
    }

    /// Evict non-pinned resident targets (LRU, oldest at the front of
    /// `registry_order`) until the non-pinned population is at or below
    /// [`REGISTRY_CAP`]. Pinned slots (deferred render Stores whose only copy is
    /// on the GPU) and an optional `protect`ed identity rotate to the back
    /// instead of evicting — they are bounded separately and must not count
    /// toward the cap, or a pinned burst would force the active set out (thrash).
    /// One full rotation is the budget. `registry_order` is a `VecDeque`, so each
    /// front pop / rotate-to-back is O(1) and the whole sweep is O(n) — not the
    /// O(n²) a `Vec` front-`remove(0)` per rotation would cost under a large
    /// pinned population (`reg=512` measured under multi-4K load).
    ///
    /// Shared by both admit paths (`registry_ensure` passes the just-resolved
    /// identity as `protect`; `registry_ensure_color` passes `None`).
    unsafe fn evict_registry_to_cap(
        &mut self,
        ctx: &DeviceContext,
        counters: &EngineCounters,
        protect: Option<&TargetIdentity>,
    ) {
        let mut non_pinned = self.non_pinned_registry_len();
        let mut rotations = self.registry_order.len();
        while non_pinned > REGISTRY_CAP && rotations > 0 {
            rotations -= 1;
            let Some(old_k) = self.registry_order.front().cloned() else {
                break;
            };
            if self
                .registry
                .get(&old_k)
                .is_some_and(|slot| slot.pin_count > 0)
                || protect == Some(&old_k)
            {
                self.registry_order.pop_front();
                self.registry_order.push_back(old_k);
                continue;
            }
            if let Some(old) = self.registry.remove(&old_k) {
                if old.framebuffer != vk::Framebuffer::null() {
                    self.dispose(&ctx.device, DeferredHandle::Framebuffer(old.framebuffer));
                }
                self.dispose(
                    &ctx.device,
                    DeferredHandle::RecycleTarget(FreeTargetImage {
                        image: old.image,
                        memory: old.memory,
                        view: old.view,
                        width: old.width,
                        height: old.height,
                        format: old.color_format,
                    }),
                );
                counters
                    .target_evicts
                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            }
            self.registry_order.pop_front();
            non_pinned = non_pinned.saturating_sub(1);
        }
    }

    /// Refresh a resident's idle-drain timestamp without going through the draw
    /// path. The present/export path resolves a target via `registry_get` (a
    /// read) rather than `registry_ensure` (which stamps), so a target that is
    /// re-presented but not re-drawn would otherwise age out from under the
    /// display — this keeps the currently-presented target alive.
    pub(crate) fn registry_touch(&mut self, identity: &TargetIdentity) {
        self.registry_touch_at(identity, self.idle_clock_ms);
    }

    /// Refresh a resident's idle-drain timestamp to at least `now_ms`. Used by
    /// host-window direct present before the export attempt so offscreen
    /// compositor peers needed for route-B tile compositing do not age out while
    /// the displayed member remains active.
    pub(crate) fn registry_touch_at(&mut self, identity: &TargetIdentity, now_ms: u64) {
        if now_ms > self.idle_clock_ms {
            self.idle_clock_ms = now_ms;
        }
        let touch = self.idle_clock_ms;
        if let Some(slot) = self.registry.get_mut(identity) {
            slot.last_touch_ms = touch;
        }
    }

    pub(crate) fn cap_pressure_occupancy(&self) -> CapPressureOccupancy {
        let pinned = self
            .registry
            .values()
            .filter(|slot| slot.pin_count > 0)
            .count();
        CapPressureOccupancy {
            registry_len: self.registry_order.len(),
            registry_cap: REGISTRY_CAP,
            registry_pinned: pinned,
            desc_blocks: self.desc_arena.block_count(),
            sampled_len: self.sampled_cache.len(),
            sampled_cap: SAMPLED_CACHE_CAP,
            sampled_bytes: self.sampled_cache_bytes,
            sampled_byte_cap: SAMPLED_CACHE_BYTE_CAP,
            graveyard_len: self.graveyard.len(),
            slab: self.slab.occupancy(),
            target_free_imgs: self.target_free.values().map(Vec::len).sum(),
            sampled_free_imgs: self.sampled_free.values().map(Vec::len).sum(),
            storage_free_imgs: self.storage_image_free.values().map(Vec::len).sum(),
            storage_recycle_admits: self.storage_recycle_admits,
            storage_recycle_cap_drops: self.storage_recycle_cap_drops,
            storage_resident: self.compute_storage_registry.len(),
            storage_resident_pinned: self
                .compute_storage_registry
                .values()
                .filter(|r| r.pinned)
                .count(),
            staging_free_bytes: self
                .staging_free
                .values()
                .flat_map(|v| v.iter())
                .map(|s| s.size)
                .sum(),
            readback_free_bytes: self
                .readback_free
                .values()
                .flat_map(|v| v.iter())
                .map(|s| s.size)
                .sum(),
        }
    }

    /// Take a completed stats reduction for `(identity, generation, seq)`.
    /// `None` = not armed, or not finished yet (non-blocking; try again later).
    pub(crate) unsafe fn take_present_stats(
        &mut self,
        ctx: &DeviceContext,
        identity: &TargetIdentity,
        generation: u64,
        seq: u64,
    ) -> Option<stats_reduce::PresentStats> {
        self.stats_reduce.consume(ctx, identity, generation, seq)
    }

    /// Blocking consume of a store-path stats reduction (waits the slot's fence).
    /// Mirrors `present_scatter_gpu`'s stats collection for the packed-contig
    /// store path, which needs this frame's content stats before it returns.
    pub(crate) unsafe fn consume_store_stats_blocking(
        &mut self,
        ctx: &DeviceContext,
        identity: &TargetIdentity,
        generation: u64,
        seq: u64,
    ) -> Option<super::content_stats::Color8ContentStats> {
        self.stats_reduce
            .consume_blocking(ctx, identity, generation, seq)
            .map(Into::into)
    }

    /// Make an armed stats reduction unmatchable (its present is gone).
    pub(crate) fn cancel_present_stats(&mut self, seq: u64) {
        self.stats_reduce.cancel(seq);
    }

    /// `(arms, hits, misses, not_ready, saturated)` for the always-on census.
    pub(crate) fn present_stats_counters(&self) -> (u64, u64, u64, u64, u64) {
        self.stats_reduce.stats()
    }

    /// Cumulative sampled-cache pool recycle diagnostics:
    /// `(free_hits, free_allocs, recycle_admits, recycle_cap_drops)`.
    /// Merged into `CounterSnapshot` by `engine::counter_snapshot`.
    pub(crate) fn recycle_stats(&self) -> (u64, u64, u64, u64) {
        (
            self.sampled_free_hits,
            self.sampled_free_allocs,
            self.sampled_recycle_admits,
            self.sampled_recycle_cap_drops,
        )
    }

    pub(crate) unsafe fn ensure_init(
        &mut self,
        ctx: &DeviceContext,
        counters: &EngineCounters,
    ) -> Result<(), DrawError> {
        if self.initialized {
            return Ok(());
        }
        let cmd_pool = ctx
            .device
            .create_command_pool(
                &vk::CommandPoolCreateInfo::default()
                    .queue_family_index(ctx.gq)
                    .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER),
                None,
            )
            .map_err(|e| DrawError::VkCall(VkCall::new(VkOp::PoolsCreateCommandPool, e)))?;
        counters.note_create();
        let cmd_bufs = ctx
            .device
            .allocate_command_buffers(
                &vk::CommandBufferAllocateInfo::default()
                    .command_pool(cmd_pool)
                    .level(vk::CommandBufferLevel::PRIMARY)
                    .command_buffer_count(RING_DEPTH as u32),
            )
            .map_err(|e| {
                ctx.device.destroy_command_pool(cmd_pool, None);
                DrawError::VkCall(VkCall::new(VkOp::PoolsAllocCommandBuffers, e))
            })?;
        // Growable descriptor arena: block 0 up front, more blocks on demand.
        // Free sets after each draw/dispatch; exhaustion grows rather than drops.
        let mut desc_arena = DescriptorArena::empty();
        if let Err(e) = desc_arena.create_first_block(&ctx.device) {
            ctx.device.destroy_command_pool(cmd_pool, None);
            return Err(e);
        }
        counters.note_create();
        let mut slots = Vec::with_capacity(RING_DEPTH);
        for cmd_buf in cmd_bufs.into_iter() {
            // Fences start unsignaled: a slot with no pending cleanup is never
            // waited on, and a submit requires an unsignaled fence.
            match ctx
                .device
                .create_fence(&vk::FenceCreateInfo::default(), None)
            {
                Ok(fence) => {
                    counters.note_create();
                    slots.push(CmdSlot {
                        cmd_buf,
                        fence,
                        pending: None,
                    });
                }
                Err(e) => {
                    for slot in &slots {
                        ctx.device.destroy_fence(slot.fence, None);
                    }
                    desc_arena.destroy(&ctx.device);
                    ctx.device.destroy_command_pool(cmd_pool, None);
                    return Err(DrawError::VkCall(VkCall::new(VkOp::PoolsCreateFence, e)));
                }
            }
        }
        self.cmd_pool = cmd_pool;
        self.desc_arena = desc_arena;
        self.slots = slots;
        self.cur = 0;
        self.in_flight = 0;
        self.initialized = true;
        Ok(())
    }

    /// Allocate one descriptor set for `dsl` from the arena, growing a new pool
    /// block on exhaustion (rather than dropping the draw). Returns the set and
    /// its owning pool — pair the pool with the set so a later free routes to
    /// the allocating block. Emits the always-on `desc_arena_grow` cap-pressure
    /// proxy on growth (a rare event; zero under normal load).
    pub(crate) unsafe fn alloc_descriptor_set(
        &mut self,
        device: &ash::Device,
        dsl: vk::DescriptorSetLayout,
        counters: &EngineCounters,
    ) -> Result<(vk::DescriptorSet, vk::DescriptorPool), DrawError> {
        let (set, pool, grew) = self.desc_arena.allocate(device, dsl)?;
        if grew {
            counters.desc_pool_grow.fetch_add(1, Ordering::Relaxed);
            crate::observe::off(format!(
                "desc_arena_grow blocks={} block_max_sets={DESC_BLOCK_MAX_SETS} cause=pool_exhausted",
                self.desc_arena.block_count()
            ));
        }
        Ok((set, pool))
    }

    /// Free `(set, owning_pool)` pairs back to their allocating blocks.
    pub(crate) unsafe fn free_descriptor_sets(
        &self,
        device: &ash::Device,
        sets: &[(vk::DescriptorSet, vk::DescriptorPool)],
    ) {
        self.desc_arena.free(device, sets);
    }

    /// True while any recorded GPU work may still reference pool objects: a
    /// submitted-but-unretired CB, or the open draw batch's still-recording CB
    /// (unsubmitted, but it already references images/buffers — destroying
    /// them before its flush would be a use-after-free at submit).
    fn gpu_work_open(&self) -> bool {
        self.in_flight > 0 || self.open_batch.is_some()
    }

    /// Destroy `handle` now if no CB is in flight, else park it in the
    /// graveyard until every in-flight fence retires.
    pub(crate) unsafe fn dispose(&mut self, device: &ash::Device, handle: DeferredHandle) {
        if !self.gpu_work_open() {
            self.destroy_or_recycle(device, handle);
        } else {
            self.graveyard.push(handle);
        }
    }

    /// Acquire + bind DEVICE_LOCAL image memory from the slab suballocator,
    /// registering the sub-allocation against `image` on success. On bind
    /// failure the slab range is released (the caller destroys the `image`).
    /// Returns the backing `VkDeviceMemory` (shared across the block's other
    /// images) for the pool's image struct; the image was bound at the slab
    /// offset, not offset 0.
    unsafe fn bind_image_slab(
        &mut self,
        ctx: &DeviceContext,
        image: vk::Image,
        ireq: &vk::MemoryRequirements,
        bind_op: VkOp,
        counters: &EngineCounters,
    ) -> Result<vk::DeviceMemory, DrawError> {
        self.slab
            .ensure_image_unregistered(image)
            .map_err(DrawError::Slab)?;
        let token = self.slab.acquire(ctx, ireq, counters)?;
        match ctx
            .device
            .bind_image_memory(image, token.memory, token.offset())
        {
            Ok(()) => {
                self.slab.register(image, token);
                Ok(token.memory)
            }
            Err(e) => {
                self.slab.release_token(&ctx.device, token);
                Err(DrawError::VkCall(VkCall::new(bind_op, e)))
            }
        }
    }

    /// Release the slab sub-allocation backing `image` (the caller destroys the
    /// `image`/view handles). No-op for a non-slab image. This replaces the raw
    /// `vkFreeMemory` at every DEVICE_LOCAL-image free site: the memory belongs
    /// to a shared block, not the image.
    unsafe fn free_image_slab(&mut self, device: &ash::Device, image: vk::Image) {
        self.slab.free_image(device, image);
    }

    /// Terminal handling for a deferred handle once it is safe (in_flight == 0):
    /// a `RecycleSampled` slot rejoins `sampled_free` (bounded per key) for reuse
    /// instead of being destroyed; every other handle is destroyed.
    unsafe fn destroy_or_recycle(&mut self, device: &ash::Device, handle: DeferredHandle) {
        match handle {
            DeferredHandle::RecycleSampled(slot) => {
                if let Some(slot) = self.try_recycle_sampled(slot) {
                    self.destroy_deferred_handle(device, DeferredHandle::RecycleSampled(slot));
                }
            }
            DeferredHandle::RecycleTarget(img) => {
                if let Some(img) = self.try_recycle_target(img) {
                    self.destroy_deferred_handle(device, DeferredHandle::RecycleTarget(img));
                }
            }
            other => self.destroy_deferred_handle(device, other),
        }
    }

    /// Return an evicted sampled slot to `sampled_free` for reuse by a later
    /// same-geometry `acquire_sampled`. `None` means it was recycled; `Some(slot)`
    /// means the per-key cap is full and the caller must destroy it. Device-free
    /// so the cap/routing is unit-testable without a GPU.
    fn try_recycle_sampled(&mut self, slot: SampledSlot) -> Option<SampledSlot> {
        // Global cap first: a diverse burst (hundreds of distinct sampled keys,
        // each ≤ the per-key cap) would otherwise fill the pool unboundedly and
        // pin every slab block. Past the global cap the slot is destroyed.
        if self.sampled_free.values().map(Vec::len).sum::<usize>() >= SAMPLED_FREE_CAP_TOTAL {
            self.sampled_recycle_cap_drops += 1;
            return Some(slot);
        }
        let list = self.sampled_free.entry(slot.key()).or_default();
        if list.len() < SAMPLED_FREE_CAP_PER_KEY {
            list.push(slot);
            self.sampled_recycle_admits += 1;
            None
        } else {
            // Per-key cap full: this evicted slot is destroyed, not reused. A
            // high count here (with high sampled_free_allocs) means the cap is
            // the recycle limiter.
            self.sampled_recycle_cap_drops += 1;
            Some(slot)
        }
    }

    /// Return a displaced resident-target image to `target_free` for reuse by a
    /// later same-(geometry, format) `registry_ensure`/`registry_ensure_color`
    /// create. `None` means it was recycled; `Some(img)` means the per-key cap
    /// is full and the caller must destroy it. Device-free so the cap/routing is
    /// unit-testable without a GPU (mirrors [`Self::try_recycle_sampled`]).
    fn try_recycle_target(&mut self, img: FreeTargetImage) -> Option<FreeTargetImage> {
        // Global cap first (mirrors try_recycle_sampled): bound a diverse burst.
        if self.target_free.values().map(Vec::len).sum::<usize>() >= TARGET_FREE_CAP_TOTAL {
            self.target_recycle_cap_drops += 1;
            return Some(img);
        }
        let list = self.target_free.entry(img.key()).or_default();
        if list.len() < TARGET_FREE_CAP_PER_KEY {
            list.push(img);
            self.target_recycle_admits += 1;
            None
        } else {
            // Per-key cap full: this displaced image is destroyed, not reused. A
            // high count here (with high target_free_allocs) means the cap is
            // the recycle limiter.
            self.target_recycle_cap_drops += 1;
            Some(img)
        }
    }

    /// Return a retired transient compute-storage image to `storage_image_free`
    /// for reuse by a later same-geometry `acquire_storage_image`. `None` means
    /// it was recycled; `Some(slot)` means a per-key or the global cap is full and
    /// the caller must destroy it (freeing its standalone `VkDeviceMemory`).
    /// Device-free so the cap/routing is unit-testable without a GPU (mirrors
    /// [`Self::try_recycle_sampled`] / [`Self::try_recycle_target`]).
    fn try_recycle_storage_image(&mut self, slot: StorageImageSlot) -> Option<StorageImageSlot> {
        // Global cap first (mirrors try_recycle_sampled): bound a diverse burst so
        // an all-new-geometry compute workload cannot leak unbounded standalone
        // device allocations.
        if self
            .storage_image_free
            .values()
            .map(Vec::len)
            .sum::<usize>()
            >= STORAGE_IMAGE_FREE_CAP_TOTAL
        {
            self.storage_recycle_cap_drops += 1;
            return Some(slot);
        }
        let list = self.storage_image_free.entry(slot.key).or_default();
        if list.len() < STORAGE_IMAGE_FREE_CAP_PER_KEY {
            list.push(slot);
            self.storage_recycle_admits += 1;
            None
        } else {
            self.storage_recycle_cap_drops += 1;
            Some(slot)
        }
    }

    /// Pop a recycled resident-target image for `(width, height, format)` if one
    /// is available, else `None`. Splits the reuse (`target_free_hits`) vs
    /// fresh-alloc (`target_free_allocs`) census so a boot can prove the
    /// per-frame realloc storm collapsed (allocs ≈ 0 under video).
    fn take_free_target(
        &mut self,
        width: u32,
        height: u32,
        format: vk::Format,
    ) -> Option<FreeTargetImage> {
        let key = TargetRecycleKey {
            width,
            height,
            format,
        };
        let img = self.target_free.get_mut(&key).and_then(Vec::pop);
        if img.is_some() {
            self.target_free_hits += 1;
        } else {
            self.target_free_allocs += 1;
        }
        img
    }

    /// Cumulative resident-target recycle diagnostics:
    /// `(free_hits, free_allocs, recycle_admits, recycle_cap_drops)`.
    pub(crate) fn target_recycle_stats(&self) -> (u64, u64, u64, u64) {
        (
            self.target_free_hits,
            self.target_free_allocs,
            self.target_recycle_admits,
            self.target_recycle_cap_drops,
        )
    }

    /// Cumulative compute-storage recycle diagnostics: `(admits, cap_drops)`.
    /// A nonzero `cap_drops` means a per-key or global cap actively bounded the
    /// pool (an all-new-geometry compute burst) — the "the cap is biting" signal
    /// surfaced as `st_drop` on the `vram` census.
    #[cfg(test)]
    pub(crate) fn storage_recycle_stats(&self) -> (u64, u64) {
        (self.storage_recycle_admits, self.storage_recycle_cap_drops)
    }

    unsafe fn drain_graveyard(&mut self, device: &ash::Device) {
        // Collect first: recycling a `RecycleSampled` handle borrows
        // `self.sampled_free`, which conflicts with draining `self.graveyard` in
        // place.
        let handles: Vec<DeferredHandle> = self.graveyard.drain(..).collect();
        for handle in handles {
            self.destroy_or_recycle(device, handle);
        }
    }

    fn wait_error(counters: &EngineCounters, e: vk::Result, op: DeviceLostOp) -> DrawError {
        if e == vk::Result::TIMEOUT {
            counters.fence_timeouts.fetch_add(1, Ordering::Relaxed);
            DrawError::FenceTimeout
        } else if e == vk::Result::ERROR_DEVICE_LOST {
            DrawError::DeviceLost(DeviceLostDecline::Driver { op, result: e })
        } else {
            DrawError::VkCall(VkCall::new(op.vk_op(), e))
        }
    }

    /// Retire one slot: wait its fence, reset it, and drain the cleanup it
    /// owes. No-op for a slot with nothing pending.
    unsafe fn retire_slot(
        &mut self,
        ctx: &DeviceContext,
        counters: &EngineCounters,
        index: usize,
    ) -> Result<(), DrawError> {
        if self.slots[index].pending.is_none() {
            return Ok(());
        }
        let fence = self.slots[index].fence;
        ctx.device
            .wait_for_fences(&[fence], true, FENCE_TIMEOUT_NS)
            .map_err(|e| Self::wait_error(counters, e, DeviceLostOp::PoolsWaitFencesRetire))?;
        ctx.device
            .reset_fences(&[fence])
            .map_err(|e| DrawError::VkCall(VkCall::new(VkOp::PoolsResetFencesRetire, e)))?;
        let pending = self.slots[index].pending.take().expect("checked above");
        self.in_flight = self.in_flight.saturating_sub(1);
        self.drain_cleanup(&ctx.device, pending);
        if !self.gpu_work_open() {
            self.drain_graveyard(&ctx.device);
        }
        Ok(())
    }

    /// Start a new entry (draw / dispatch / sync helper): advance to the next
    /// ring slot, retiring it first if its CB is still in flight (this is the
    /// only place a full ring blocks). Returns the slot's CB + fence; the CB
    /// is ready to reset/record and the fence is unsignaled, ready to submit.
    pub(crate) unsafe fn begin_entry(
        &mut self,
        ctx: &DeviceContext,
        counters: &EngineCounters,
    ) -> Result<(vk::CommandBuffer, vk::Fence), DrawError> {
        // Any path that claims a slot first submits the deferred batch — the
        // single choke point that keeps queue order = record order for every
        // reader/compute/prefetch path (and prevents ring wrap from resetting
        // the still-recording batch CB).
        self.batch_flush(ctx, counters)?;
        let next = (self.cur + 1) % self.slots.len();
        if self.slots[next].pending.is_some() {
            // Count as a "block" only when the fence is genuinely unsignaled
            // (the GPU still owns the slot); reclaiming a finished slot on
            // advance is bookkeeping, not a stall.
            let still_running = !ctx
                .device
                .get_fence_status(self.slots[next].fence)
                .map_err(|e| {
                    Self::wait_error(counters, e, DeviceLostOp::PoolsFenceStatusBeginEntry)
                })?;
            let retire_started = Instant::now();
            self.retire_slot(ctx, counters, next)?;
            if still_running {
                counters.retire_wait_us.fetch_add(
                    retire_started.elapsed().as_micros() as u64,
                    Ordering::Relaxed,
                );
                counters.ring_retire_blocks.fetch_add(1, Ordering::Relaxed);
            }
        }
        // Safety valve: a long pure-async streak keeps in_flight above 0 at
        // every retire, so eviction-displaced handles could pile up. Under
        // real workloads presents/readbacks quiesce the ring long before this
        // fires.
        if self.graveyard.len() >= GRAVEYARD_FORCE_DRAIN {
            self.retire_all(ctx, counters)?;
        }
        self.cur = next;
        Ok((self.slots[next].cmd_buf, self.slots[next].fence))
    }

    /// [`Self::begin_entry`] for fully synchronous paths (target reads,
    /// presents, imports): additionally retires EVERY in-flight slot, so the
    /// caller records against a quiesced device — required by paths whose
    /// barriers assume no concurrent CB (e.g. UNDEFINED-layout seeds of an
    /// existing registry image).
    pub(crate) unsafe fn begin_entry_sync(
        &mut self,
        ctx: &DeviceContext,
        counters: &EngineCounters,
    ) -> Result<(vk::CommandBuffer, vk::Fence), DrawError> {
        let retire_started = Instant::now();
        let waited = self.in_flight > 0;
        self.retire_all(ctx, counters)?;
        if waited {
            counters.retire_wait_us.fetch_add(
                retire_started.elapsed().as_micros() as u64,
                Ordering::Relaxed,
            );
        }
        self.begin_entry(ctx, counters)
    }

    /// Seal the current entry's transient resources: move every live pool slot
    /// (which the just-recorded CB references) out of the shared live lists so
    /// a concurrent entry cannot recycle them, bundled with the descriptor set
    /// and deferred sampled-cache admissions.
    pub(crate) fn seal_entry(
        &mut self,
        dsets: Vec<(vk::DescriptorSet, vk::DescriptorPool)>,
        sampled_retains: Vec<(
            vk::Image,
            std::sync::Arc<Vec<u8>>,
            Option<crate::backend::vulkan::engine::SampledContentIdentity>,
        )>,
    ) -> PendingGpuCleanup {
        let mut readback: Vec<BufferSlot> = self.readback_live.take().into_iter().collect();
        readback.append(&mut self.readback_multi_live);
        PendingGpuCleanup {
            dsets,
            staging: std::mem::take(&mut self.staging_live),
            readback,
            sampled: std::mem::take(&mut self.sampled_live),
            storage_images: std::mem::take(&mut self.storage_image_live),
            sampled_retains,
        }
    }

    /// Park the sealed cleanup on the current slot: its CB was submitted with
    /// the slot fence and the entry returns without waiting.
    pub(crate) fn finish_entry_async(&mut self, cleanup: PendingGpuCleanup) {
        debug_assert!(
            self.slots[self.cur].pending.is_none(),
            "current slot already owes cleanup"
        );
        self.slots[self.cur].pending = Some(cleanup);
        self.in_flight += 1;
    }

    /// The open batch's (CB, fence) when a draw at (identity, geometry, bgra)
    /// can append to it. `None` means the caller must claim its own slot (and
    /// `begin_entry` will flush the batch first).
    ///
    /// A full batch (BATCH_MAX_DRAWS) also answers `None`, turning the next
    /// same-target draw into a flush-then-reopen. Unbounded batches destroyed
    /// the pipeline (live A/B 2026-07-19): the GPU idled while the CPU
    /// recorded the whole run — the present then blocked behind the entire
    /// batch executing from scratch (presents 38.7 -> 27/s) — and every
    /// draw's staging slots stayed hoarded in ONE pending ring entry until
    /// its fence retired, starving the free lists into per-bind
    /// vkCreateBuffer/vkAllocateMemory churn (setup_bufs 50 -> 108 us/draw).
    /// The cap keeps the GPU fed every ~N draws while still amortizing the
    /// per-draw submit+fence cost N-fold.
    pub(crate) fn batch_slot(
        &self,
        identity: &TargetIdentity,
        width: u32,
        height: u32,
        bgra: bool,
    ) -> Option<(vk::CommandBuffer, vk::Fence)> {
        let b = self.open_batch.as_ref()?;
        (b.draws < BATCH_MAX_DRAWS
            && b.identity == *identity
            && b.width == width
            && b.height == height
            && b.bgra == bgra)
            .then_some((b.cb, b.fence))
    }

    /// Record a batch-deferred draw's completion: open the batch on its ring
    /// slot (opener) or extend it (joiner), accumulating the per-draw
    /// descriptor set and sampled-cache admissions for the single flush-time
    /// seal. The CB stays in recording state; submit happens at
    /// [`Self::batch_flush`].
    #[allow(
        clippy::too_many_arguments,
        reason = "batch ownership tracks every Vulkan object and resident pin explicitly"
    )]
    pub(crate) fn batch_append(
        &mut self,
        cb: vk::CommandBuffer,
        fence: vk::Fence,
        identity: TargetIdentity,
        width: u32,
        height: u32,
        bgra: bool,
        dset: Option<(vk::DescriptorSet, vk::DescriptorPool)>,
        sampled_retains: Vec<(
            vk::Image,
            std::sync::Arc<Vec<u8>>,
            Option<crate::backend::vulkan::engine::SampledContentIdentity>,
        )>,
        counters: &EngineCounters,
    ) {
        match self.open_batch.as_mut() {
            Some(b) => {
                debug_assert!(b.cb == cb, "joiner recorded into a foreign CB");
                b.draws += 1;
                b.dsets.extend(dset);
                b.sampled_retains.extend(sampled_retains);
                counters.batch_joins.fetch_add(1, Ordering::Relaxed);
            }
            None => {
                debug_assert!(
                    self.slots[self.cur].pending.is_none(),
                    "batch opener's slot already owes cleanup"
                );
                self.open_batch = Some(OpenBatch {
                    cb,
                    fence,
                    identity,
                    width,
                    height,
                    bgra,
                    draws: 1,
                    dsets: dset.into_iter().collect(),
                    sampled_retains,
                });
                counters.batch_opens.fetch_add(1, Ordering::Relaxed);
            }
        }
    }

    /// Submit the open batch (if any): end its CB, queue it on the batch
    /// fence, and park the accumulated cleanup on its ring slot. No-op when no
    /// batch is open. On submit failure the descriptor sets are freed
    /// immediately (the CB never reached the queue) and the pool-slot lives
    /// stay for the next seal, matching the per-draw submit-error path.
    pub(crate) unsafe fn batch_flush(
        &mut self,
        ctx: &DeviceContext,
        counters: &EngineCounters,
    ) -> Result<(), DrawError> {
        let Some(mut batch) = self.open_batch.take() else {
            return Ok(());
        };
        counters.batch_flushes.fetch_add(1, Ordering::Relaxed);
        counters
            .batch_flush_draws
            .fetch_add(batch.draws, Ordering::Relaxed);
        let submit = (|| -> Result<(), DrawError> {
            ctx.device
                .end_command_buffer(batch.cb)
                .map_err(|e| DrawError::VkCall(VkCall::new(VkOp::PoolsEndCbBatch, e)))?;
            let cbs = [batch.cb];
            let si = vk::SubmitInfo::default().command_buffers(&cbs);
            match ctx.device.queue_submit(ctx.queue(), &[si], batch.fence) {
                Ok(()) => Ok(()),
                Err(e) if e == vk::Result::ERROR_DEVICE_LOST => {
                    Err(DrawError::DeviceLost(DeviceLostDecline::Driver {
                        op: DeviceLostOp::PoolsSubmitBatch,
                        result: e,
                    }))
                }
                Err(e) => Err(DrawError::VkCall(VkCall::new(VkOp::PoolsSubmitBatch, e))),
            }
        })();
        match submit {
            Ok(()) => {
                let cleanup = self.seal_entry(
                    std::mem::take(&mut batch.dsets),
                    std::mem::take(&mut batch.sampled_retains),
                );
                self.finish_entry_async(cleanup);
                Ok(())
            }
            Err(e) => {
                self.desc_arena.free(&ctx.device, &batch.dsets);
                Err(e)
            }
        }
    }

    /// Wait a single already-submitted entry fence WITHOUT retiring the ring.
    ///
    /// A synchronous reader (e.g. [`super::read_target_inner`]) that submitted
    /// its own copy CB with `fence` only needs *that* copy to finish before it
    /// maps the readback — it does not need to quiesce unrelated in-flight
    /// draws. The copy's `ALL_COMMANDS → TRANSFER` barrier plus single-queue
    /// submission order already guarantee it observes every prior-submitted
    /// draw's writes (the same argument the async prefetch path relies on), so
    /// waiting the whole ring here would just serialize the guest-blocking
    /// readback behind an unrelated heavy draw — the `finish_us` tail. The
    /// caller must have parked its cleanup with [`Self::finish_entry_async`]
    /// first; the slot stays pending and the ring retires it later (its fence
    /// is already signaled, so that retire is a no-wait drain).
    pub(crate) unsafe fn wait_entry_fence(
        &self,
        ctx: &DeviceContext,
        counters: &EngineCounters,
        fence: vk::Fence,
    ) -> Result<(), DrawError> {
        ctx.device
            .wait_for_fences(&[fence], true, FENCE_TIMEOUT_NS)
            .map_err(|e| Self::wait_error(counters, e, DeviceLostOp::PoolsWaitFencesEntry))
    }

    /// Wait + retire every in-flight slot and drain the graveyard. Callers
    /// park their own cleanup with [`Self::finish_entry_async`] right after
    /// submit, then call this for a synchronous result — a failed wait leaves
    /// the slot pending, and whichever entry claims it next re-waits, so no
    /// path ever submits on an unretired fence.
    pub(crate) unsafe fn retire_all(
        &mut self,
        ctx: &DeviceContext,
        counters: &EngineCounters,
    ) -> Result<(), DrawError> {
        // Quiesce includes deferred batched work: submit it so the retire
        // below actually waits it out.
        self.batch_flush(ctx, counters)?;
        for index in 0..self.slots.len() {
            self.retire_slot(ctx, counters, index)?;
        }
        if !self.gpu_work_open() {
            self.drain_graveyard(&ctx.device);
        }
        Ok(())
    }

    /// Free/recycle the resources owed by a retired entry.
    ///
    /// # Safety
    /// The CB that referenced these resources must have retired.
    unsafe fn drain_cleanup(&mut self, device: &ash::Device, mut pending: PendingGpuCleanup) {
        self.desc_arena.free(device, &pending.dsets);
        // Cache admissions first (they move slots into the cache and may evict
        // images — deferred via dispose() while others are in flight), then
        // recycle what remains.
        for (image, content, identity) in pending.sampled_retains.drain(..) {
            if let Some(index) = pending.sampled.iter().position(|slot| slot.image == image) {
                let slot = pending.sampled.remove(index);
                self.admit_sampled_slot(device, slot, &content, identity);
            }
        }
        for slot in pending.staging.drain(..) {
            let bucket = Self::bucket(slot.size);
            self.staging_free.entry(bucket).or_default().push(slot);
        }
        for slot in pending.readback.drain(..) {
            let bucket = Self::bucket(slot.size);
            self.readback_free.entry(bucket).or_default().push(slot);
        }
        for slot in pending.sampled.drain(..) {
            // Respect the same global + per-key cap as the eviction-recycle path
            // (`destroy_or_recycle`). This per-frame retire path previously pushed
            // unconditionally, so a diverse-content workload (many distinct sampled
            // keys — measured 4×4K video) grew `sampled_free` past
            // `SAMPLED_FREE_CAP_TOTAL` (live `sfree=203`), each slot pinning a slab
            // sub-allocation. Past the cap the slot is destroyed, not recycled.
            if let Some(slot) = self.try_recycle_sampled(slot) {
                self.destroy_deferred_handle(device, DeferredHandle::RecycleSampled(slot));
            }
        }
        for slot in pending.storage_images.drain(..) {
            // Respect the per-key + global cap (mirrors the sampled retire path
            // above). This path previously pushed unconditionally, so an all-new-
            // geometry compute workload (diff-heavy / CoreImage / blur burst) grew
            // `storage_image_free` without bound — and because storage images are
            // standalone (non-slab) allocations, each hoarded slot pins a whole
            // VkDeviceMemory (invisible to the slab `resident_mb`/`live_subs`
            // census). Past the cap the slot is destroyed, freeing its device
            // memory.
            if let Some(slot) = self.try_recycle_storage_image(slot) {
                self.destroy_deferred_handle(
                    device,
                    DeferredHandle::Image {
                        image: slot.image,
                        view: slot.view,
                        memory: slot.memory,
                    },
                );
            }
        }
    }

    fn bucket(size: u64) -> u64 {
        // Power-of-two bucket, min 64.
        let mut b = 64u64;
        while b < size {
            b = b.saturating_mul(2);
            if b == 0 {
                return u64::MAX;
            }
        }
        b
    }

    pub(crate) unsafe fn acquire_staging(
        &mut self,
        ctx: &DeviceContext,
        size: u64,
        usage: vk::BufferUsageFlags,
        counters: &EngineCounters,
    ) -> Result<BufferSlot, DrawError> {
        let need = size.max(4);
        let bucket = Self::bucket(need);
        // Prefer exact-usage free slots in this bucket; usage is OR'd broadly so reuse is fine.
        if let Some(list) = self.staging_free.get_mut(&bucket) {
            if let Some(slot) = list.pop() {
                self.staging_live.push(BufferSlot {
                    buffer: slot.buffer,
                    memory: slot.memory,
                    size: slot.size,
                });
                return Ok(BufferSlot {
                    buffer: slot.buffer,
                    memory: slot.memory,
                    size: slot.size,
                });
            }
        }
        let buffer = ctx
            .device
            .create_buffer(
                &vk::BufferCreateInfo::default()
                    .size(bucket)
                    .usage(
                        usage
                            | vk::BufferUsageFlags::TRANSFER_SRC
                            | vk::BufferUsageFlags::TRANSFER_DST
                            | vk::BufferUsageFlags::VERTEX_BUFFER
                            | vk::BufferUsageFlags::INDEX_BUFFER
                            | vk::BufferUsageFlags::STORAGE_BUFFER,
                    )
                    .sharing_mode(vk::SharingMode::EXCLUSIVE),
                None,
            )
            .map_err(|e| DrawError::VkCall(VkCall::new(VkOp::PoolsCreateStaging, e)))?;
        counters.note_create();
        let req = ctx.device.get_buffer_memory_requirements(buffer);
        let mt = ctx
            .memory_type_for(req.memory_type_bits, MemoryClass::Upload)
            .ok_or_else(|| {
                DrawError::Unsupported(super::reason::DrawReason::NoHostVisibleMemoryForStaging {
                    memory_type_bits: req.memory_type_bits,
                })
            })?;
        let memory = allocate_memory_timed(
            ctx,
            &vk::MemoryAllocateInfo::default()
                .allocation_size(req.size)
                .memory_type_index(mt),
            counters,
        )
        .map_err(|e| {
            ctx.device.destroy_buffer(buffer, None);
            DrawError::VkCall(VkCall::new(VkOp::PoolsAllocStaging, e))
        })?;
        counters.note_alloc();
        ctx.device
            .bind_buffer_memory(buffer, memory, 0)
            .map_err(|e| {
                ctx.device.free_memory(memory, None);
                ctx.device.destroy_buffer(buffer, None);
                DrawError::VkCall(VkCall::new(VkOp::PoolsBindStaging, e))
            })?;
        let slot = BufferSlot {
            buffer,
            memory,
            size: bucket,
        };
        self.staging_live.push(BufferSlot {
            buffer: slot.buffer,
            memory: slot.memory,
            size: slot.size,
        });
        Ok(slot)
    }

    pub(crate) unsafe fn write_staging(
        &self,
        ctx: &DeviceContext,
        slot: &BufferSlot,
        bytes: &[u8],
    ) -> Result<(), DrawError> {
        let size = bytes.len().max(4) as u64;
        let ptr = ctx
            .device
            .map_memory(slot.memory, 0, size, vk::MemoryMapFlags::empty())
            .map_err(|e| DrawError::VkCall(VkCall::new(VkOp::PoolsMapStaging, e)))?
            as *mut u8;
        unsafe {
            if bytes.is_empty() {
                // Nothing to copy — the mapped span is the 4-byte minimum; zero it
                // so the bind reads defined memory.
                std::ptr::write_bytes(ptr, 0, size as usize);
            } else {
                // The mapped span is exactly `bytes.len()` (`size == bytes.len()`
                // here), so the copy overwrites every mapped byte — a preceding
                // full-span zeroing would just be overwritten. Copy only.
                std::ptr::copy_nonoverlapping(bytes.as_ptr(), ptr, bytes.len());
            }
        }
        ctx.device.unmap_memory(slot.memory);
        Ok(())
    }

    /// Snapshot guest-run spans directly into a mapped staging slot.
    ///
    /// The deferred-submit snapshot path used to `cpu_bytes()` the runs into a
    /// heap `Vec` and then `write_staging` that `Vec` into the mapped buffer —
    /// two full copies plus an allocation per bind, ~4.8 binds/draw under
    /// compositing. This copies each run's guest RAM straight into the mapped
    /// staging span at its running offset (one copy, no intermediate `Vec`), and
    /// zeroes only the tail if the runs underfill `total_len` (short read). The
    /// freshness contract is identical to `cpu_bytes` — the read races guest CPU
    /// writes exactly as the encode-time staging read does.
    pub(crate) unsafe fn write_staging_from_runs(
        &self,
        ctx: &DeviceContext,
        slot: &BufferSlot,
        runs: &[super::types::GuestRun],
        total_len: u64,
    ) -> Result<(), DrawError> {
        let size = total_len.max(4);
        let ptr = ctx
            .device
            .map_memory(slot.memory, 0, size, vk::MemoryMapFlags::empty())
            .map_err(|e| DrawError::VkCall(VkCall::new(VkOp::PoolsMapStaging, e)))?
            as *mut u8;
        let total = total_len as usize;
        let mut off = 0usize;
        unsafe {
            for run in runs {
                if off >= total {
                    break;
                }
                let n = (run.len as usize).min(total - off);
                // SAFETY: `host_ptr` is a stable RAMBlock alias from
                // `HostOps::map_pages`, valid for the VM lifetime; `ptr` is the
                // mapped staging span of `size >= total` bytes and `off + n <=
                // total`, so the destination stays in bounds.
                std::ptr::copy_nonoverlapping(run.host_ptr as *const u8, ptr.add(off), n);
                off += n;
            }
            // Runs underfilled the span (short read) or the 4-byte minimum tail:
            // zero the remainder so the bind reads defined memory.
            if off < size as usize {
                std::ptr::write_bytes(ptr.add(off), 0, size as usize - off);
            }
        }
        ctx.device.unmap_memory(slot.memory);
        Ok(())
    }

    pub(crate) fn recycle_staging(&mut self) {
        for slot in self.staging_live.drain(..) {
            let bucket = Self::bucket(slot.size);
            self.staging_free.entry(bucket).or_default().push(slot);
        }
    }

    pub(crate) unsafe fn acquire_readback(
        &mut self,
        ctx: &DeviceContext,
        size: u64,
        counters: &EngineCounters,
    ) -> Result<BufferSlot, DrawError> {
        let bucket = Self::bucket(size.max(4));
        if let Some(list) = self.readback_free.get_mut(&bucket) {
            if let Some(slot) = list.pop() {
                self.readback_live = Some(BufferSlot {
                    buffer: slot.buffer,
                    memory: slot.memory,
                    size: slot.size,
                });
                return Ok(slot);
            }
        }
        let buffer = ctx
            .device
            .create_buffer(
                &vk::BufferCreateInfo::default()
                    .size(bucket)
                    .usage(vk::BufferUsageFlags::TRANSFER_DST)
                    .sharing_mode(vk::SharingMode::EXCLUSIVE),
                None,
            )
            .map_err(|e| DrawError::VkCall(VkCall::new(VkOp::PoolsCreateReadback, e)))?;
        counters.note_create();
        let req = ctx.device.get_buffer_memory_requirements(buffer);
        let mt = ctx
            .memory_type_for(req.memory_type_bits, MemoryClass::Readback)
            .ok_or_else(|| {
                DrawError::Unsupported(super::reason::DrawReason::NoHostVisibleMemoryForReadback {
                    memory_type_bits: req.memory_type_bits,
                })
            })?;
        let memory = allocate_memory_timed(
            ctx,
            &vk::MemoryAllocateInfo::default()
                .allocation_size(req.size)
                .memory_type_index(mt),
            counters,
        )
        .map_err(|e| {
            ctx.device.destroy_buffer(buffer, None);
            DrawError::VkCall(VkCall::new(VkOp::PoolsAllocReadback, e))
        })?;
        counters.note_alloc();
        ctx.device
            .bind_buffer_memory(buffer, memory, 0)
            .map_err(|e| {
                ctx.device.free_memory(memory, None);
                ctx.device.destroy_buffer(buffer, None);
                DrawError::VkCall(VkCall::new(VkOp::PoolsBindReadback, e))
            })?;
        let slot = BufferSlot {
            buffer,
            memory,
            size: bucket,
        };
        self.readback_live = Some(BufferSlot {
            buffer: slot.buffer,
            memory: slot.memory,
            size: slot.size,
        });
        Ok(slot)
    }

    pub(crate) fn recycle_readback(&mut self) {
        if let Some(slot) = self.readback_live.take() {
            let bucket = Self::bucket(slot.size);
            self.readback_free.entry(bucket).or_default().push(slot);
        }
        for slot in self.readback_multi_live.drain(..) {
            let bucket = Self::bucket(slot.size);
            self.readback_free.entry(bucket).or_default().push(slot);
        }
    }

    /// Acquire an additional readback buffer without replacing the primary live slot.
    pub(crate) unsafe fn acquire_readback_extra(
        &mut self,
        ctx: &DeviceContext,
        size: u64,
        counters: &EngineCounters,
    ) -> Result<BufferSlot, DrawError> {
        let bucket = Self::bucket(size.max(4));
        if let Some(list) = self.readback_free.get_mut(&bucket) {
            if let Some(slot) = list.pop() {
                self.readback_multi_live.push(BufferSlot {
                    buffer: slot.buffer,
                    memory: slot.memory,
                    size: slot.size,
                });
                return Ok(slot);
            }
        }
        let buffer = ctx
            .device
            .create_buffer(
                &vk::BufferCreateInfo::default()
                    .size(bucket)
                    .usage(vk::BufferUsageFlags::TRANSFER_DST)
                    .sharing_mode(vk::SharingMode::EXCLUSIVE),
                None,
            )
            .map_err(|e| DrawError::VkCall(VkCall::new(VkOp::PoolsCreateReadbackExtra, e)))?;
        counters.note_create();
        let req = ctx.device.get_buffer_memory_requirements(buffer);
        let mt = ctx
            .memory_type_for(req.memory_type_bits, MemoryClass::Readback)
            .ok_or_else(|| {
                DrawError::Unsupported(super::reason::DrawReason::NoHostVisibleMemoryForReadback {
                    memory_type_bits: req.memory_type_bits,
                })
            })?;
        let memory = allocate_memory_timed(
            ctx,
            &vk::MemoryAllocateInfo::default()
                .allocation_size(req.size)
                .memory_type_index(mt),
            counters,
        )
        .map_err(|e| {
            ctx.device.destroy_buffer(buffer, None);
            DrawError::VkCall(VkCall::new(VkOp::PoolsAllocReadbackExtra, e))
        })?;
        counters.note_alloc();
        ctx.device
            .bind_buffer_memory(buffer, memory, 0)
            .map_err(|e| {
                ctx.device.free_memory(memory, None);
                ctx.device.destroy_buffer(buffer, None);
                DrawError::VkCall(VkCall::new(VkOp::PoolsBindReadbackExtra, e))
            })?;
        let slot = BufferSlot {
            buffer,
            memory,
            size: bucket,
        };
        self.readback_multi_live.push(BufferSlot {
            buffer: slot.buffer,
            memory: slot.memory,
            size: slot.size,
        });
        Ok(slot)
    }

    pub(crate) unsafe fn acquire_target(
        &mut self,
        ctx: &DeviceContext,
        key: TargetKey,
        render_pass: vk::RenderPass,
        counters: &EngineCounters,
    ) -> Result<&TargetSlot, DrawError> {
        let map_key = (key, render_pass.as_raw());
        if self.targets.contains_key(&map_key) {
            return Ok(self.targets.get(&map_key).unwrap());
        }
        let usage = vk::ImageUsageFlags::COLOR_ATTACHMENT
            | vk::ImageUsageFlags::INPUT_ATTACHMENT
            | vk::ImageUsageFlags::TRANSFER_SRC
            | if key.with_transfer_dst {
                vk::ImageUsageFlags::TRANSFER_DST
            } else {
                vk::ImageUsageFlags::empty()
            };
        let image = ctx
            .device
            .create_image(
                &vk::ImageCreateInfo::default()
                    .image_type(vk::ImageType::TYPE_2D)
                    .format(translate::pixel::RESIDENT_RGBA_FORMAT)
                    .extent(vk::Extent3D {
                        width: key.width,
                        height: key.height,
                        depth: 1,
                    })
                    .mip_levels(1)
                    .array_layers(1)
                    .samples(vk::SampleCountFlags::TYPE_1)
                    .tiling(vk::ImageTiling::OPTIMAL)
                    .usage(usage)
                    .initial_layout(vk::ImageLayout::UNDEFINED),
                None,
            )
            .map_err(|e| DrawError::VkCall(VkCall::new(VkOp::PoolsCreateTargetImage, e)))?;
        counters.note_create();
        let ireq = ctx.device.get_image_memory_requirements(image);
        let memory = match self.bind_image_slab(ctx, image, &ireq, VkOp::PoolsBindTarget, counters)
        {
            Ok(m) => m,
            Err(error) => {
                ctx.device.destroy_image(image, None);
                return Err(error);
            }
        };
        let view = match ctx.device.create_image_view(
            &vk::ImageViewCreateInfo::default()
                .image(image)
                .view_type(vk::ImageViewType::TYPE_2D)
                .format(translate::pixel::RESIDENT_RGBA_FORMAT)
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                }),
            None,
        ) {
            Ok(v) => v,
            Err(e) => {
                self.free_image_slab(&ctx.device, image);
                ctx.device.destroy_image(image, None);
                return Err(DrawError::VkCall(VkCall::new(
                    VkOp::PoolsCreateTargetView,
                    e,
                )));
            }
        };
        counters.note_create();
        let attachments = [view];
        let framebuffer = match ctx.device.create_framebuffer(
            &vk::FramebufferCreateInfo::default()
                .render_pass(render_pass)
                .attachments(&attachments)
                .width(key.width)
                .height(key.height)
                .layers(1),
            None,
        ) {
            Ok(fb) => fb,
            Err(e) => {
                ctx.device.destroy_image_view(view, None);
                self.free_image_slab(&ctx.device, image);
                ctx.device.destroy_image(image, None);
                return Err(DrawError::VkCall(VkCall::new(
                    VkOp::PoolsCreateFramebuffer,
                    e,
                )));
            }
        };
        counters.note_create();
        // Cap target pool
        if self.target_order.len() >= 32 {
            if let Some(old_k) = self.target_order.first().cloned() {
                if let Some(old) = self.targets.remove(&old_k) {
                    self.dispose(&ctx.device, DeferredHandle::Framebuffer(old.framebuffer));
                    self.dispose(
                        &ctx.device,
                        DeferredHandle::Image {
                            image: old.image,
                            view: old.view,
                            memory: old.memory,
                        },
                    );
                }
                self.target_order.remove(0);
            }
        }
        self.targets.insert(
            map_key,
            TargetSlot {
                image,
                memory,
                view,
                framebuffer,
            },
        );
        self.target_order.push(map_key);
        Ok(self.targets.get(&map_key).unwrap())
    }

    #[allow(
        clippy::too_many_arguments,
        reason = "sampled-image acquisition mirrors its Vulkan format and content identity"
    )]
    pub(crate) unsafe fn acquire_sampled(
        &mut self,
        ctx: &DeviceContext,
        width: u32,
        height: u32,
        layers: u32,
        volume: bool,
        cube: bool,
        arrayed: bool,
        format: ash::vk::Format,
        swizzle: crate::contract::pixel_format::SwizzlePlan,
        counters: &EngineCounters,
    ) -> Result<SampledSlot, DrawError> {
        let sk = SampledKey {
            width,
            height,
            layers,
            volume,
            cube,
            arrayed,
            format,
            swizzle,
        };
        if let Some(list) = self.sampled_free.get_mut(&sk) {
            if let Some(slot) = list.pop() {
                let handles = slot.handles();
                self.sampled_live.push(slot);
                // Reused a recycled slot — no vkAllocateMemory this acquire.
                self.sampled_free_hits += 1;
                return Ok(handles);
            }
        }
        let image_type = if volume {
            vk::ImageType::TYPE_3D
        } else {
            vk::ImageType::TYPE_2D
        };
        let view_type = if volume {
            vk::ImageViewType::TYPE_3D
        } else if cube {
            vk::ImageViewType::CUBE
        } else if arrayed {
            vk::ImageViewType::TYPE_2D_ARRAY
        } else {
            vk::ImageViewType::TYPE_2D
        };
        let extent_depth = if volume { layers } else { 1 };
        let array_layers = if volume { 1 } else { layers };
        let flags = if cube {
            vk::ImageCreateFlags::CUBE_COMPATIBLE
        } else {
            vk::ImageCreateFlags::empty()
        };
        let vk_format = format;
        let image = ctx
            .device
            .create_image(
                &vk::ImageCreateInfo::default()
                    .flags(flags)
                    .image_type(image_type)
                    .format(vk_format)
                    .extent(vk::Extent3D {
                        width,
                        height,
                        depth: extent_depth,
                    })
                    .mip_levels(1)
                    .array_layers(array_layers)
                    .samples(vk::SampleCountFlags::TYPE_1)
                    .tiling(vk::ImageTiling::OPTIMAL)
                    .usage(vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST)
                    .initial_layout(vk::ImageLayout::UNDEFINED),
                None,
            )
            .map_err(|e| DrawError::VkCall(VkCall::new(VkOp::PoolsCreateSampledImage, e)))?;
        counters.note_create();
        let req = ctx.device.get_image_memory_requirements(image);
        // Cold acquire: the per-geometry free list was empty so we pay a fresh
        // sub-allocation (a block alloc only when no block has room). This is
        // the lag-tail driver; recycling aims to keep it low relative to
        // sampled_free_hits.
        self.sampled_free_allocs += 1;
        let memory = match self.bind_image_slab(ctx, image, &req, VkOp::PoolsBindSampled, counters)
        {
            Ok(m) => m,
            Err(error) => {
                ctx.device.destroy_image(image, None);
                return Err(error);
            }
        };
        let view = match ctx.device.create_image_view(
            &vk::ImageViewCreateInfo::default()
                .image(image)
                .view_type(view_type)
                .format(vk_format)
                // The decoded type-8 view swizzle, performed by the hardware at
                // sample time. Identity for every ordinary bind. The format
                // contributes no mapping of its own: `translate::pixel`'s
                // sampled rail admits only formats whose Metal channels sit
                // identically on their Vulkan ones, and declines the rest by
                // name rather than binding a plan it cannot carry.
                .components(translate::pixel::vk_component_mapping(&swizzle))
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: array_layers,
                }),
            None,
        ) {
            Ok(v) => v,
            Err(e) => {
                self.free_image_slab(&ctx.device, image);
                ctx.device.destroy_image(image, None);
                return Err(DrawError::VkCall(VkCall::new(
                    VkOp::PoolsCreateSampledView,
                    e,
                )));
            }
        };
        counters.note_create();
        let slot = SampledSlot {
            image,
            memory,
            view,
            width,
            height,
            layers,
            volume,
            cube,
            arrayed,
            format,
            swizzle,
        };
        let handles = slot.handles();
        self.sampled_live.push(slot);
        Ok(handles)
    }

    #[allow(
        clippy::too_many_arguments,
        reason = "sampled-cache lookup mirrors the complete image and content key"
    )]
    pub(crate) fn find_cached_sampled(
        &mut self,
        width: u32,
        height: u32,
        layers: u32,
        volume: bool,
        cube: bool,
        arrayed: bool,
        format: ash::vk::Format,
        swizzle: crate::contract::pixel_format::SwizzlePlan,
        content: &[u8],
        identity: Option<crate::backend::vulkan::engine::SampledContentIdentity>,
        counters: &EngineCounters,
    ) -> Option<SampledSlot> {
        let key = SampledKey {
            width,
            height,
            layers,
            volume,
            cube,
            arrayed,
            format,
            swizzle,
        };
        // Identity fast path: same producer key + generation means the bytes
        // are unchanged by the producer's coherence model — bind the retained
        // image without hashing or comparing content.
        if let Some(id) = identity {
            if let Some(index) = self
                .sampled_cache
                .iter()
                .position(|entry| entry.slot.key() == key && entry.identity == Some(id))
            {
                let mut entry = self.sampled_cache.remove(index);
                entry.last_touch_ms = self.idle_clock_ms;
                let handles = entry.slot.handles();
                self.sampled_cache.push(entry);
                counters
                    .sampled_identity_hits
                    .fetch_add(1, Ordering::Relaxed);
                return Some(handles);
            }
        }
        let content_hash = sampled_content_hash(content);
        // 128-bit fingerprint match binds the retained image directly — the
        // former full-frame `entry.content == content` compare (a cold DRAM
        // read of the retained copy on every hit) is dropped in favour of the
        // wider digest.
        let found = self
            .sampled_cache
            .iter()
            .position(|entry| entry.slot.key() == key && entry.content_hash == content_hash);
        let Some(index) = found else {
            counters
                .sampled_cache_misses
                .fetch_add(1, Ordering::Relaxed);
            return None;
        };
        let mut entry = self.sampled_cache.remove(index);
        // Same content re-presented under a new identity/generation: adopt it
        // so the fast path serves the follow-up draws.
        if identity.is_some() {
            entry.identity = identity;
        }
        entry.last_touch_ms = self.idle_clock_ms;
        let handles = entry.slot.handles();
        self.sampled_cache.push(entry);
        counters.sampled_cache_hits.fetch_add(1, Ordering::Relaxed);
        counters
            .sampled_cache_hit_bytes
            .fetch_add(content.len() as u64, Ordering::Relaxed);
        Some(handles)
    }

    /// Admit one detached sampled slot into the exact-content cache. A slot
    /// whose content duplicates an existing entry returns to the live list
    /// (recycled later); cap evictions go through dispose() so an image a
    /// concurrent in-flight CB samples is never destroyed under it.
    unsafe fn admit_sampled_slot(
        &mut self,
        device: &ash::Device,
        slot: SampledSlot,
        content: &std::sync::Arc<Vec<u8>>,
        identity: Option<crate::backend::vulkan::engine::SampledContentIdentity>,
    ) {
        if content.len() > SAMPLED_CACHE_BYTE_CAP {
            self.sampled_live.push(slot);
            return;
        }
        let content_hash = sampled_content_hash(content);
        if let Some(existing) = self
            .sampled_cache
            .iter_mut()
            .find(|entry| entry.slot.key() == slot.key() && entry.content_hash == content_hash)
        {
            if identity.is_some() {
                existing.identity = identity;
            }
            self.sampled_live.push(slot);
            return;
        }
        self.sampled_cache_bytes = self.sampled_cache_bytes.saturating_add(content.len());
        let touch = self.idle_clock_ms;
        self.sampled_cache.push(ResidentSampledSlot {
            slot,
            content_hash,
            content_len: content.len(),
            identity,
            last_touch_ms: touch,
        });
        while self.sampled_cache.len() > SAMPLED_CACHE_CAP
            || self.sampled_cache_bytes > SAMPLED_CACHE_BYTE_CAP
        {
            let evicted = self.sampled_cache.remove(0);
            self.sampled_cache_bytes = self.sampled_cache_bytes.saturating_sub(evicted.content_len);
            // Recycle rather than destroy: a content-changing sampled input
            // (live tile / video frame) re-uploads into this same-geometry image
            // next frame instead of a fresh vkAllocateMemory. Routed through the
            // in-flight-safe deferral (an in-flight CB may still sample it).
            self.dispose(device, DeferredHandle::RecycleSampled(evicted.slot));
        }
    }

    pub(crate) fn recycle_sampled(&mut self) {
        for slot in self.sampled_live.drain(..) {
            let sk = slot.key();
            self.sampled_free.entry(sk).or_default().push(slot);
        }
    }
}

#[cfg(test)]
mod recycle_tests {
    use super::*;

    fn null_slot(w: u32, h: u32) -> SampledSlot {
        SampledSlot {
            image: vk::Image::null(),
            memory: vk::DeviceMemory::null(),
            view: vk::ImageView::null(),
            width: w,
            height: h,
            layers: 1,
            volume: false,
            cube: false,
            arrayed: false,
            format: crate::backend::vulkan::translate::pixel::vk_texel_layout(
                crate::contract::pixel_format::TexelLayout::Bgra8,
            ),
            swizzle: Default::default(),
        }
    }

    /// A *diverse* burst — many distinct geometries, each ≤ the per-key cap —
    /// must not grow `sampled_free` past the GLOBAL cap: this is the measured
    /// VRAM-return stall (`sfree=593` pinning every slab block). Each distinct
    /// key admits until the pool total hits `SAMPLED_FREE_CAP_TOTAL`, then every
    /// further eviction is destroyed (returns Some) regardless of its key.
    #[test]
    fn sampled_free_global_cap_bounds_a_diverse_burst() {
        let mut pools = ResourcePools::new();
        // One eviction per distinct 1-pixel-taller geometry, more than the global
        // cap. Each key is fresh so the per-key cap never bites — only the global
        // cap can bound this.
        let mut admitted = 0;
        for i in 0..(SAMPLED_FREE_CAP_TOTAL + 40) {
            if pools
                .try_recycle_sampled(null_slot(16, 16 + i as u32))
                .is_none()
            {
                admitted += 1;
            }
        }
        assert_eq!(
            admitted, SAMPLED_FREE_CAP_TOTAL,
            "global cap bounds the diverse burst"
        );
        assert_eq!(
            pools.sampled_free.values().map(Vec::len).sum::<usize>(),
            SAMPLED_FREE_CAP_TOTAL,
            "pool total pinned at the global cap"
        );
    }

    /// `target_free` has the same global cap for the same reason.
    #[test]
    fn target_free_global_cap_bounds_a_diverse_burst() {
        let mut pools = ResourcePools::new();
        let mut admitted = 0;
        for i in 0..(TARGET_FREE_CAP_TOTAL + 40) {
            if pools
                .try_recycle_target(null_target(
                    16,
                    16 + i as u32,
                    translate::pixel::SCANOUT_FORMAT,
                ))
                .is_none()
            {
                admitted += 1;
            }
        }
        assert_eq!(
            admitted, TARGET_FREE_CAP_TOTAL,
            "global cap bounds the burst"
        );
    }

    fn null_storage_slot(w: u32, h: u32) -> StorageImageSlot {
        StorageImageSlot {
            image: vk::Image::null(),
            memory: vk::DeviceMemory::null(),
            view: vk::ImageView::null(),
            key: StorageImageKey {
                width: w,
                height: h,
                layers: 1,
                format: StorageImageFormat::default(),
                one_dim: false,
                arrayed: false,
                volume: false,
                sampled_only: false,
            },
            array_layers: 1,
            extent_depth: 1,
        }
    }

    /// The compute-storage recycle pool (`storage_image_free`) had NO cap before
    /// this fix, so an all-new-geometry compute burst (each a standalone, non-slab
    /// `vkAllocateMemory`) grew it without bound. A diverse burst — many distinct
    /// geometries, each ≤ the per-key cap — must now stop admitting at the GLOBAL
    /// cap; past it every slot is returned (Some) for the caller to destroy.
    #[test]
    fn storage_free_global_cap_bounds_a_diverse_burst() {
        let mut pools = ResourcePools::new();
        let mut admitted = 0;
        for i in 0..(STORAGE_IMAGE_FREE_CAP_TOTAL + 40) {
            if pools
                .try_recycle_storage_image(null_storage_slot(16, 16 + i as u32))
                .is_none()
            {
                admitted += 1;
            }
        }
        assert_eq!(
            admitted, STORAGE_IMAGE_FREE_CAP_TOTAL,
            "global cap bounds the diverse burst"
        );
        assert_eq!(
            pools
                .storage_image_free
                .values()
                .map(Vec::len)
                .sum::<usize>(),
            STORAGE_IMAGE_FREE_CAP_TOTAL,
            "pool total pinned at the global cap"
        );
    }

    /// Within one geometry the pool recycles up to the per-key cap (reuse instead
    /// of a fresh allocation); beyond the cap the slot is returned for the caller
    /// to destroy, and the admits/cap-drops counters split the two so a leak is
    /// diagnosable (`st_drop` on the census).
    #[test]
    fn storage_free_recycle_up_to_per_key_cap_then_drops() {
        let mut pools = ResourcePools::new();
        let key = null_storage_slot(512, 512).key;
        for i in 0..STORAGE_IMAGE_FREE_CAP_PER_KEY {
            assert!(
                pools
                    .try_recycle_storage_image(null_storage_slot(512, 512))
                    .is_none(),
                "recycle {i} within the per-key cap must be admitted"
            );
        }
        assert_eq!(
            pools.storage_image_free.get(&key).map(Vec::len),
            Some(STORAGE_IMAGE_FREE_CAP_PER_KEY)
        );
        assert!(
            pools
                .try_recycle_storage_image(null_storage_slot(512, 512))
                .is_some(),
            "over the per-key cap the slot is returned for destroy"
        );
        let (admits, cap_drops) = pools.storage_recycle_stats();
        assert_eq!(admits, STORAGE_IMAGE_FREE_CAP_PER_KEY as u64);
        assert_eq!(cap_drops, 1);
    }

    /// `pop_any_pool_entry` drains a keyed pool one entry at a time across all
    /// buckets and removes empties, so the idle trim can empty the whole pool.
    #[test]
    fn pop_any_pool_entry_drains_all_buckets() {
        let mut pool: HashMap<u32, Vec<u32>> = HashMap::new();
        pool.insert(1, vec![10, 11]);
        pool.insert(2, vec![20]);
        let mut popped = Vec::new();
        while let Some(v) = pop_any_pool_entry(&mut pool) {
            popped.push(v);
        }
        popped.sort_unstable();
        assert_eq!(popped, vec![10, 11, 20]);
        assert!(pool.is_empty(), "emptied buckets are removed");
    }

    /// Evicted sampled-cache slots rejoin `sampled_free` for reuse (no fresh
    /// `vkAllocateMemory` next frame) up to a per-key cap; beyond the cap the
    /// caller must destroy so a one-off geometry cannot pin memory for the whole
    /// guest lifetime. Device-free: exercises only the routing/cap decision.
    #[test]
    fn evicted_sampled_slots_recycle_into_free_list_up_to_cap() {
        let mut pools = ResourcePools::new();
        let hd = null_slot(1920, 1080).key();

        // The first CAP evictions of one geometry recycle (return None) and are
        // available for a later same-geometry acquire.
        for i in 0..SAMPLED_FREE_CAP_PER_KEY {
            assert!(
                pools.try_recycle_sampled(null_slot(1920, 1080)).is_none(),
                "eviction {i} within cap must recycle"
            );
        }
        assert_eq!(
            pools.sampled_free.get(&hd).map(Vec::len),
            Some(SAMPLED_FREE_CAP_PER_KEY)
        );

        // Over the cap: caller must destroy (returns the slot); free list is
        // bounded, not grown.
        assert!(
            pools.try_recycle_sampled(null_slot(1920, 1080)).is_some(),
            "over-cap eviction must not recycle"
        );
        assert_eq!(
            pools.sampled_free.get(&hd).map(Vec::len),
            Some(SAMPLED_FREE_CAP_PER_KEY)
        );

        // A different geometry has an independent cap.
        let small = null_slot(64, 64).key();
        assert!(pools.try_recycle_sampled(null_slot(64, 64)).is_none());
        assert_eq!(pools.sampled_free.get(&small).map(Vec::len), Some(1));
    }

    /// The idle sampled-cache trim reclaims only entries idle past
    /// `IDLE_TARGET_AGE_MS` (an actively-sampled entry is touched every frame and
    /// must survive), decrements the byte accounting by exactly the trimmed
    /// entries, and is bounded per pass — so a live video session is never
    /// disturbed while a settled one's ≤128 MiB of frame textures return to the
    /// driver.
    #[test]
    fn idle_trim_reclaims_only_aged_sampled_cache_entries() {
        let mut pools = ResourcePools::new();
        let push = |pools: &mut ResourcePools, w: u32, h: u32, touch: u64, len: usize| {
            pools.sampled_cache_bytes = pools.sampled_cache_bytes.saturating_add(len);
            pools.sampled_cache.push(ResidentSampledSlot {
                slot: null_slot(w, h),
                content_hash: ((w as u128) << 64) | h as u128,
                content_len: len,
                identity: None,
                last_touch_ms: touch,
            });
        };
        push(&mut pools, 1920, 1080, 1_000, 8_000_000); // aged
        push(&mut pools, 1280, 720, 1_500, 4_000_000); // aged
        push(&mut pools, 640, 480, 9_000, 1_000_000); // freshly touched
        assert_eq!(pools.sampled_cache.len(), 3);
        assert_eq!(pools.sampled_cache_bytes, 13_000_000);

        // Idle clock at 10_000 ms → cutoff = 10_000 - IDLE_TARGET_AGE_MS.
        let cutoff = 10_000u64.saturating_sub(IDLE_TARGET_AGE_MS);
        let taken = pools.take_aged_sampled_slots(cutoff, IDLE_RECYCLE_TRIM_PER_PASS);

        assert_eq!(taken.len(), 2, "only the two aged entries are taken");
        assert_eq!(pools.sampled_cache.len(), 1, "the fresh entry stays cached");
        assert_eq!(
            pools.sampled_cache[0].slot.width, 640,
            "the surviving entry is the freshly-touched one"
        );
        assert_eq!(
            pools.sampled_cache_bytes, 1_000_000,
            "byte accounting drops exactly the two aged entries"
        );

        // Per-pass bound: three more aged entries, max=2 → only two taken.
        push(&mut pools, 100, 100, 0, 500_000);
        push(&mut pools, 101, 101, 0, 500_000);
        push(&mut pools, 102, 102, 0, 500_000);
        assert_eq!(pools.take_aged_sampled_slots(cutoff, 2).len(), 2);
    }

    /// The compute-storage residents are standalone (non-slab) VkDeviceMemory, so a
    /// settled compute session that leaves stale residents pins whole allocations
    /// until an LRU eviction that never comes on a static desktop. The idle drain
    /// must reclaim exactly the non-pinned residents idle past the age cutoff, leave
    /// a freshly-touched or pinned resident alone, and bound the batch per pass.
    #[test]
    fn idle_trim_reclaims_only_aged_non_pinned_compute_storage() {
        let mut pools = ResourcePools::new();
        let admit = |pools: &mut ResourcePools, tex: u32, touch: u64, pinned: bool| {
            let id = ComputeStorageResidencyKey::linear(0, tex, 0, 0, 0, 8, 8, 0);
            pools.compute_storage_registry.insert(
                id,
                ResidentStorageImageSlot {
                    slot: null_storage_slot(8, 8),
                    generation: 0,
                    layout: vk::ImageLayout::UNDEFINED,
                    pinned,
                    last_touch_ms: touch,
                },
            );
            pools.compute_storage_order.push_back(id);
        };
        admit(&mut pools, 1, 1_000, false); // aged, evictable
        admit(&mut pools, 2, 1_500, false); // aged, evictable
        admit(&mut pools, 3, 1_500, true); // aged but PINNED — must survive
        admit(&mut pools, 4, 9_000, false); // freshly touched — must survive
        assert_eq!(pools.compute_storage_registry.len(), 4);

        // Idle clock at 10_000 ms → cutoff = 10_000 - IDLE_TARGET_AGE_MS.
        let cutoff = 10_000u64.saturating_sub(IDLE_TARGET_AGE_MS);
        let taken = pools.take_aged_storage_residents(cutoff, IDLE_RECYCLE_TRIM_PER_PASS);

        assert_eq!(
            taken.len(),
            2,
            "only the two aged non-pinned residents are taken"
        );
        assert_eq!(
            pools.compute_storage_registry.len(),
            2,
            "pinned + fresh survive"
        );
        assert_eq!(
            pools.compute_storage_order.len(),
            2,
            "the LRU order drops exactly the reclaimed residents"
        );
        assert!(
            pools
                .compute_storage_registry
                .keys()
                .all(|k| k.texture_ref == 3 || k.texture_ref == 4),
            "the survivors are the pinned (3) and the freshly-touched (4) residents"
        );

        // Per-pass bound: three more aged residents, max=2 → only two taken.
        admit(&mut pools, 5, 0, false);
        admit(&mut pools, 6, 0, false);
        admit(&mut pools, 7, 0, false);
        assert_eq!(pools.take_aged_storage_residents(cutoff, 2).len(), 2);
    }

    /// The recycle diagnostics (`recycle_stats`) count admits vs cap-drops so a
    /// later boot can tell whether the per-key cap or the drain timing is the
    /// lag-tail limiter. `try_recycle_sampled` is the only mutator of the two
    /// recycle counters; the acquire-side hit/alloc counters need a device so
    /// they are exercised on the live path, not here.
    #[test]
    fn recycle_stats_count_admits_and_cap_drops() {
        let mut pools = ResourcePools::new();
        assert_eq!(pools.recycle_stats(), (0, 0, 0, 0));

        // CAP admits, then one over-cap drop, on one geometry.
        for _ in 0..SAMPLED_FREE_CAP_PER_KEY {
            pools.try_recycle_sampled(null_slot(1920, 1080));
        }
        pools.try_recycle_sampled(null_slot(1920, 1080));
        // One admit on an independent geometry.
        pools.try_recycle_sampled(null_slot(64, 64));

        let (free_hits, free_allocs, admits, cap_drops) = pools.recycle_stats();
        assert_eq!(free_hits, 0, "no acquires happened");
        assert_eq!(free_allocs, 0, "no acquires happened");
        assert_eq!(
            admits,
            SAMPLED_FREE_CAP_PER_KEY as u64 + 1,
            "CAP big-geometry admits + 1 small-geometry admit"
        );
        assert_eq!(
            cap_drops, 1,
            "exactly the one over-cap eviction was dropped"
        );
    }

    fn null_target(w: u32, h: u32, format: vk::Format) -> FreeTargetImage {
        FreeTargetImage {
            image: vk::Image::null(),
            memory: vk::DeviceMemory::null(),
            view: vk::ImageView::null(),
            width: w,
            height: h,
            format,
        }
    }

    /// Resident-target images displaced from the identity registry (generation
    /// bump / geometry change / LRU) rejoin `target_free` for reuse up to a
    /// per-(geometry, format) cap; beyond it the caller must destroy so a
    /// one-off geometry cannot pin VRAM for the guest lifetime. Device-free:
    /// exercises only the routing/cap decision (mirrors the sampled recycle).
    #[test]
    fn displaced_targets_recycle_into_free_list_up_to_cap() {
        let mut pools = ResourcePools::new();
        let fmt = translate::pixel::SCANOUT_FORMAT;
        let key = null_target(1920, 1080, fmt).key();

        for i in 0..TARGET_FREE_CAP_PER_KEY {
            assert!(
                pools
                    .try_recycle_target(null_target(1920, 1080, fmt))
                    .is_none(),
                "displacement {i} within cap must recycle"
            );
        }
        assert_eq!(
            pools.target_free.get(&key).map(Vec::len),
            Some(TARGET_FREE_CAP_PER_KEY)
        );

        assert!(
            pools
                .try_recycle_target(null_target(1920, 1080, fmt))
                .is_some(),
            "over-cap displacement must not recycle"
        );
        assert_eq!(
            pools.target_free.get(&key).map(Vec::len),
            Some(TARGET_FREE_CAP_PER_KEY)
        );

        // A different format is an independent bucket (an RGBA image cannot back
        // a BGRA attachment).
        let rgba = null_target(1920, 1080, translate::pixel::RESIDENT_RGBA_FORMAT).key();
        assert!(pools
            .try_recycle_target(null_target(
                1920,
                1080,
                translate::pixel::RESIDENT_RGBA_FORMAT
            ))
            .is_none());
        assert_eq!(pools.target_free.get(&rgba).map(Vec::len), Some(1));
    }

    /// The exact video regression this pool fixes: a per-frame *generation* bump
    /// makes every frame a new `TargetIdentity` (registry miss) — but the
    /// displaced image, recycled by (geometry, format), is popped back on the
    /// next frame's create instead of a fresh `vkCreateImage`/`vkAllocateMemory`.
    /// The reuse split (`target_free_hits` vs `target_free_allocs`) proves it: a
    /// steady-geometry stream after the first fill is all hits, zero allocs.
    #[test]
    fn recycled_target_is_reused_across_generation_bumps() {
        let mut pools = ResourcePools::new();
        let fmt = translate::pixel::SCANOUT_FORMAT;

        // Frame 0: cold — no free image, so it counts as an alloc (miss).
        assert!(pools.take_free_target(1920, 1080, fmt).is_none());
        // Its predecessor image is displaced (a new generation replaced it) and
        // recycled.
        assert!(pools
            .try_recycle_target(null_target(1920, 1080, fmt))
            .is_none());

        // Frames 1..N: each pops the recycled image (hit) and recycles the one
        // it replaces — steady state is hit-per-frame, alloc-once.
        for f in 0..8 {
            assert!(
                pools.take_free_target(1920, 1080, fmt).is_some(),
                "frame {f} must reuse the recycled image"
            );
            assert!(pools
                .try_recycle_target(null_target(1920, 1080, fmt))
                .is_none());
        }

        let (hits, allocs, _admits, _drops) = pools.target_recycle_stats();
        assert_eq!(hits, 8, "8 steady frames each reused a recycled image");
        assert_eq!(allocs, 1, "only the cold first frame allocated");
    }
}

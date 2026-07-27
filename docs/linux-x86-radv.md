# Linux x86 host, RADV — pathway notes

Findings from bringing the x86 pathway (`reims-vgpu-pci`, Vulkan backend) up on
an AMD integrated GPU under Mesa RADV, with a macOS 13.7.8 guest. Everything
here was observed on that pathway; none of it is generalised to arm64, Metal,
or MoltenVK.

Host used for every measurement below: x86_64 KVM, Radeon integrated GPU, Mesa
RADV, 16 GiB RAM with 512 MiB swap, KDE Plasma on X11, ext4.

## `llvm-dis` must be on PATH or every draw fails

`metal2vulkan` shells out to `llvm-dis` to disassemble AIR. When the binary is
absent, translation fails per pipeline with

```
draw_encode_fail reason=draw_vk_nothing_stored class=no_metal
linux_m2v_draw reason=m2v_vertex_translate ... detail=cannot_run_llvm-dis
```

and every draw falls to `draw_fail_clear_fallback`. The guest boots, the
compositor runs, `present` keeps cycling — and the screen stays black, because
each frame is a clear. Nothing about the symptom points at a missing host
tool.

Distributions that ship only versioned binaries (`llvm-dis-18`) need either a
`llvm-dis` symlink on PATH or `METAL2VULKAN_LLVM_DIS` set to the absolute path.
The same applies to `cargo test`: `tests/vk_engine_batch.rs` fails at the first
translation and poisons the shared lock, so the remaining cases in that file
fail as a cascade.

## Host-import budget

See the commit "make the host-import budget bound a working set, not a
lifetime" for what changed. The measurements that drove it, on the same
~158-draw tranche:

| configuration | tranche | `zc_fail_import` | import declines |
|---|---|---|---|
| 1 GiB window, 1-window byte cap | 1.65–3.5 s | 1158 / tranche | constant |
| 1 GiB window, 4-window cap | ~0.45 s | 0 until the cap filled | after the cap filled |
| 256 MiB window, 16-window cap, LRU eviction | best 0.88 ms/draw | 0 | none |

Two things are worth carrying forward.

**The guest's GPU working set is spread across all of its RAM.** A 6 GiB guest
touched 23 distinct 256 MiB windows (~5.75 GiB). Shrinking the window did not
concentrate it: at 1 GiB the same guest needed at least 5 buckets. macOS
allocates GPU-visible buffers wherever it likes, so no windowed budget smaller
than guest RAM avoids re-import churn — with a 4 GiB byte cap against that
guest, 23 unique windows cost 472 imports, the hottest re-imported 50–58 times,
and the tranche cost returned to ~30 ms/draw.

Sizing guest RAM at or under the byte cap removes the churn outright: a 4 GiB
guest against the 4 GiB cap imported 13 windows and never evicted or re-imported
any of them.

**Do not raise the budget to cover a large guest.** Six 1 GiB windows against a
6 GiB guest pins the whole guest allocation. On this host that meant no
reclaimable memory, a 115 s tranche stall, an unresponsive desktop, and a
killed session. It is the whole-VMA pin the window resolver exists to prevent,
reached through the budget instead of the window size.

## Known gaps on this pathway

Observed on a booted desktop; each is visible in `/tmp/reims-vgpu-fail.log`.

- `storage_format_specialize_mismatch` (75 occurrences in one session). The
  shader declares a storage image as `Rgba8Uint` (4 bytes/texel) while the
  guest format is `Rgba32Uint` (16). The dispatch is dropped whole, which is
  what leaves Dock icons as flat colour blocks.
- `compute_stage_tex type11_fail reason=read` (45). Guest pages for a
  1024x1024 sampled texture do not read back.
- `qemu_map_pages_callback_failed rc=-1` (235). `reims_vgpu_pci_map_pages`
  requires the guest pages of a span to be packed-contiguous in host VA. A
  full-screen surface (2040 pages) is scattered across guest physical memory,
  so it falls to the scatter/readback path. Importing maximal contiguous runs
  instead of failing the whole span would keep these on the fast path.
- `TRANSPORT reason=sync_exec_lock_hold` with `finish_us` up to 1.0–1.7 s.
  `engine_us` dominates the tranche once imports are healthy; the exec lock is
  held across a synchronous wait. This is the ceiling on interactive
  responsiveness after the import path is fixed, and it is architectural
  rather than a tuning question.

## Operational notes

- The persistent `VkPipelineCache` lives under `std::env::temp_dir()`
  (`context.rs`, `pipeline_cache_disk_path`). On a host that clears `/tmp` on
  boot the cache is lost every reboot and every pipeline recompiles; RADV
  compiles cost hundreds of milliseconds and show up as multi-second tranches
  on first paint. The cache does work — it grew 352 KB to 535 KB across one
  session — so a location that survives reboot would pay for itself.
- On a filesystem without reflink (ext4 here) each `--snapshot` / `--interactive`
  boot copies the whole guest disk, which is minutes and tens of gigabytes per
  boot. `vm/disks/run/` needs sweeping.
- Kill a stuck VM by PID resolved through `/proc/*/exe`. `pkill -f` on a
  pattern containing `qemu-system-x86_64` also matches the shell that is
  running the pkill.
- A leftover QEMU holds the `hostfwd` port; the next boot dies with
  `Could not set up host forwarding rule 'tcp::2222-:22'`.

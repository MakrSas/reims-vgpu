# reims-vgpu — Linux x86 / AMD RADV fork

[![License: LGPL-3.0-or-later](https://img.shields.io/badge/License-LGPL%203.0%20or%20later-blue.svg)](LICENSE)

Fork of [steelbrain/reims-vgpu](https://github.com/steelbrain/reims-vgpu) used to bring the
**x86 macOS / Linux Vulkan** pathway up on an AMD integrated GPU under Mesa RADV, and to fix what
that bring-up turned up. Upstream's own description follows [below](#upstream-project); this first
section is specific to the fork.

Work here targets that one pathway. arm64, Metal, and MoltenVK are untouched and unmeasured —
nothing claimed here generalises to them.

## The bench

| | |
|---|---|
| CPU | AMD Ryzen 5 5560U (6C/12T) |
| GPU | Radeon Vega, Cezanne/Renoir (`1002:1638`) — Mesa 25.2.8 RADV, Vulkan 1.4 |
| RAM | 16 GiB, **512 MiB swap** |
| OS | Ubuntu 24.04.4, kernel 6.17, KDE Plasma on **X11** |
| Filesystem | ext4 — no reflink, so every snapshot boot copies the whole guest disk |
| Guest | macOS 13.7.8 Ventura, 4 GiB RAM, provisioned via OSX-KVM |

The X11 session and the small swap both matter: two of the findings below exist because of them.

## Status

The guest boots to a working desktop with `Apple Paravirtualized Graphics Device` bound — Dock,
windows, and input all function. It is not smooth yet, and some texture content is wrong. Measured
rather than eyeballed:

- **Works:** boot to desktop, real GPU acceleration, keyboard and mouse, no import declines, no
  present-capture failures.
- **Does not:** Dock icons render as flat colour blocks and the wallpaper stays black
  (`storage_format_specialize_mismatch`, `compute_stage_tex type11_fail`); interactive response is
  uneven, now dominated by command submission rather than by the memory path.

Numbers and remaining leads: [`docs/linux-x86-radv.md`](docs/linux-x86-radv.md).

## What this fork changes

1. **`vm/boot-x86.sh`: default `WAYLAND_DISPLAY` only when a Wayland socket exists.** winit selects
   Wayland whenever that variable is non-empty, without checking that the socket is real, so on an
   X11-only host the launcher's own default forced a backend that could not start and the window
   silently never appeared.

2. **The host-import budget bounds a working set, not a lifetime.**
   `HOST_IMPORT_TOTAL_BYTE_CAP` equalled a single window and `host_imports` never shrank, so the
   first import spent the whole budget permanently and every later bucket fell to the CPU scatter
   path for the rest of the session. The cap now admits several windows and evicts the coldest
   through the existing in-flight-safe deferral. On a ~158-draw tranche: **1.65–3.5 s → ~0.45 s**,
   `zc_fail_import` **1158 per tranche → 0**.

3. **Pathway notes** under `docs/`.

## Running it here

`llvm-dis` must be reachable, or `metal2vulkan` cannot disassemble AIR and *every* draw degrades to
a clear — a black screen with a healthy-looking log and no obvious cause. Ubuntu ships it
versioned:

```bash
sudo apt install llvm-18
```

Then boot with the guest sized to fit the import budget — guest RAM above the byte cap thrashes,
see the notes doc:

```bash
RAM=4G METAL2VULKAN_LLVM_DIS=/usr/bin/llvm-dis-18 sg kvm -c './vm/boot-x86.sh --device reims-vgpu-pci --interactive'
```

Diagnostics land in `/tmp/reims-vgpu-fail.log`. The `drain_tranche_us=` lines are the useful ones:
each breaks a batch of draws down into import, setup, engine, and submit time.

---

# Upstream project

> **Alpha.** This project is early and under active development. The QEMU device ABI, boot scripts,
> crate layout, backend behavior, and supported host/guest pathways may change without a stable
> compatibility guarantee. Treat it as research-quality: useful for experimentation and bring-up,
> not a frozen virtualization product.

reims-vgpu is an experimental virtual GPU for macOS guests. It aims to let macOS running inside a
VM use accelerated graphics instead of a basic framebuffer, while keeping the guest operating system
unchanged.

macOS already includes a paravirtual GPU driver named `AppleParavirtGPU.kext`.
reims-vgpu provides the QEMU device that driver attaches to, then decodes the guest's GPU command
stream on the host and executes it through Metal (TODO) or Vulkan, with Vulkan translation handled
by [`metal2vulkan`](https://github.com/steelbrain/metal2vulkan). There is no custom macOS kext and
no guest driver to install.

Contributions are welcome. I am especially interested in collaborating with developers who want to
work on correctness, visual glitches, synchronization bugs, command-stream decoding, Metal/Vulkan
translation, and making more host/guest combinations reliable.

![reims-vgpu running an arm64 macOS 13 Ventura guest desktop on an Apple Silicon host](assets/readme/reims-vgpu-macos-arm64-desktop.png)

*arm64 macOS 13 Ventura guest on an Apple Silicon host.*

![reims-vgpu running an x86_64 macOS 13 Ventura guest desktop on a Linux host](assets/readme/reims-vgpu-macos-x86-desktop.png)

*x86_64 macOS 13 Ventura guest on a Linux host.*

## Three pathways

`crates/reims-vgpu` targets the following host/guest/backend combinations. Agents pick the pathway
their unit of work is on.

| Pathway | Host | Guest | Device attach | Backend | Boot |
|---|---|---|---|---|---|
| **x86 macOS / Linux Vulkan** | Linux x86_64 (KVM) | x86_64 macOS Metal guest | PCI `reims-vgpu-pci` | host **Vulkan** via `metal2vulkan` | `vm/boot-x86.sh` |
| **arm64 macOS / macOS Metal** | Apple Silicon macOS (HVF) | arm64 macOS Metal guest (`vmapple`) | sysbus MMIO `reims-vgpu-mmio` | host **Metal** | `vm/boot-arm64.sh` |
| **arm64 macOS / macOS Vulkan** | Apple Silicon macOS (HVF) | arm64 macOS Metal guest (`vmapple`) | sysbus MMIO `reims-vgpu-mmio` | host **Vulkan** via `metal2vulkan` through MoltenVK | `vm/boot-arm64.sh` |

- QEMU device shims: `vendor/qemu` tracks
  [`steelbrain/qemu-reims-vgpu@host-reims-vgpu-vmapple`](https://github.com/steelbrain/qemu-reims-vgpu/tree/host-reims-vgpu-vmapple)
  (thin C — QOM/MMIO/IRQ/console/HostOps only)
- Product logic: `crates/reims-vgpu` (decode + device model + Metal/Vulkan backends)
- Vulkan translator dependency: public `steelbrain/metal2vulkan` Git crate. On macOS, the Vulkan
  host backend runs through MoltenVK.
- VM lifecycle: `vm/` (snapshot-revert; arm and x86 guest boot scripts)

## Getting started

This tree ships **boot scripts and the device**, not a ready-made macOS disk image. Guest disks,
firmware vars, and OpenCore blobs are private/gitignored under `vm/`. Pick a pathway, provision a
guest once, freeze a golden snapshot, then use the snapshot-revert boots for day-to-day work.
macOS 13 Ventura is the recommended guest release for bring-up.

### x86_64 guest on Linux (KVM)

1. **Host prep.** You need KVM (`/dev/kvm`), a working NVIDIA (or other) Vulkan stack for the product
   backend, and build deps for the in-tree QEMU (`scripts/qemu-build/qemu-build.sh --target x86_64
   --backend vulkan`). KVM must ignore unhandled MSRs or macOS will not boot — e.g. a modprobe conf
   with `options kvm ignore_msrs=1` (reboot or reload the module after).

2. **Generate OpenCore, OVMF, and a guest disk with [OSX-KVM](https://github.com/kholia/OSX-KVM).**
   **macOS 13 is recommended**.Follow that project’s docs to fetch recovery media, build OpenCore,
   and install macOS under QEMU+KVM. The point of this step is only to produce a
   **working, post-Setup-Assistant guest** plus the usual OpenCore/OVMF pieces — not to stay on
   OSX-KVM’s long-term launcher.

3. **Drop the artifacts where this repo expects them** (paths are the defaults in `vm/boot-x86.sh`;
   override with env if you prefer):

   | Artifact | Default location |
   |---|---|
   | Guest system disk | `vm/disks/macos.img` |
   | OpenCore boot disk | `vm/disks/OpenCore.qcow2` |
   | OVMF code | `vm/ovmf/OVMF_CODE_4M.fd` |
   | OVMF vars template | `vm/ovmf/OVMF_VARS-1920x1080.fd` |

   Finish install in the guest: enable Remote Login, install your SSH key, turn off sleep/screensaver
   as you like. Host SSH is typically `localhost:2222` → guest `:22` (see `vm/boot-x86.sh`).

4. **Capture the first immutable snapshot.** From a clean guest state (logged in, network/SSH
   known-good), shut down cleanly while booting in snapshot-capture mode:

   ```bash
   vm/boot-x86.sh --snapshot --device vmware-svga
   # clean shutdown from inside the guest → new label under vm/disks/snapshots/
   # and snapshots/current points at it
   ```

   Every later boot clones `snapshots/current` (COW when possible) and **throws the clone away** on
   exit, so wedges and hard kills never poison the golden image.

5. **Day-to-day boots.**

   ```bash
   # Console only (mainstream OSX-KVM-style VGA) while you debug the host stack
   vm/boot-x86.sh --testing --device vmware-svga

   # Product Reims VGPU device (needs in-tree QEMU + reims-vgpu Vulkan)
   REIMS_VGPU_BACKEND=vulkan scripts/qemu-build/qemu-build.sh --target x86_64
   vm/boot-x86.sh --testing --device reims-vgpu-pci

   # Host-window screenshot on the Linux/Plasma host
   scripts/screenshot-when-kde-plasma-host/screenshot-when-kde-plasma-host.sh -o /tmp/screen.png
   ```

### arm64 guest on Apple Silicon (HVF / vmapple)

Arm bring-up is **in-tree**: Virtualization.framework via Homebrew **`macosvm`**, then QEMU’s
`vmapple` machine under HVF. There is no OSX-KVM step.

1. Install **`macosvm`**, and build the vendored QEMU:

   ```bash
   scripts/qemu-build/qemu-build.sh --target aarch64 --backend metal
   ```

2. Provision a guest from a UniversalMac IPSW with the project helpers in
   `scripts/vmapple-provision/`. The live bundle lives under `vm/guest/` (disk, aux, `vm.json` /
   ECID).

3. Configure the guest once: enable Remote Login, run `scripts/vmapple-guest-config/` for no-sleep
   settings, and optionally enable auto-login by hand in System Settings. Capture a golden under
   `vm/guest/snapshots/` with the snapshot helpers (`scripts/vmapple-snapshot/`, or
   `vm/boot-arm64.sh --snapshot` once the disk is ready).

4. Boot:

   ```bash
   vm/boot-arm64.sh --testing --device reims-vgpu-mmio    # product
   vm/boot-arm64.sh --testing --device apple-gfx-mmio   # Apple ParavirtualizedGraphics A/B
   scripts/screenshot-when-macos-host/screenshot-when-macos-host.sh /tmp/screen.png
   ```

   Optional **performance ceiling** reference: the same guest under native VZ via `macosvm --gui`.

### After the first snapshot

- Prefer **`--testing`** for agent/measurement boots (time-bounded, always reverts).
- Use **`--interactive`** when you need an open-ended GUI session (still reverts unless you are in
  `--snapshot` capture mode).
- Never commit disks, IPSWs, or OpenCore/OVMF runtime under `vm/`.
- Device/backend work lives in `crates/reims-vgpu` + the thin shims in `vendor/qemu`; rebuild QEMU after
  product changes before claiming a live boot result.

## Repo layout

```text
AGENTS.md           - repo operating guide for agents
crates/             - Rust crates (`reims-vgpu`, `reims-vgpu-efi`)
scripts/            - host setup, VM lifecycle, screenshot, and diagnostic helpers
vendor/             - vendored QEMU submodule and patch record
vm/                 - VM launch/configuration glue; images are private/untracked
```

## License

Licensed under the [GNU Lesser General Public License v3.0 or later](LICENSE)
(`LGPL-3.0-or-later`).

Metal, macOS are trademarks of Apple Inc. reims-vgpu is an independent project and is not affiliated
with, sponsored by, or endorsed by Apple Inc.

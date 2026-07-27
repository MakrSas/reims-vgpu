//! Host-owned window presentation — a self-contained `winit` window with its
//! own `VkSurfaceKHR`/swapchain that presents the guest frame, replacing QEMU's
//! UI ([[host-window]]).
//!
//! v1 scope (this file): open the window, build a swapchain, and drive an
//! acquire → clear/blit → present loop; translate window input via
//! [`super::input_map`] and hand each [`HostAction`] to the [`InputSink`] (the
//! device wires that to the prompt action queue). The frame source is a
//! CPU-BGRA [`FrameSlot`] the device fills from its present capture — a working
//! window first; sharing the engine's `VkDevice` to present its resident image
//! with zero copy is the direct-present increment (see the plan).
//!
//! Linux owns the event loop on a dedicated thread. macOS requires AppKit work
//! on the process main thread, so QEMU creates it through
//! [`start_main_thread`] during device realize and then makes
//! [`run_main_thread`] its process-main UI loop.
#![allow(dead_code)]
// ash-heavy module: inner unsafe blocks add noise per ash call (matches the
// engine modules' convention).
#![allow(unsafe_op_in_unsafe_fn)]

#[cfg(target_os = "macos")]
use std::cell::RefCell;
use std::collections::HashSet;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use ash::vk;
use raw_window_handle::{HasDisplayHandle, HasWindowHandle};
use winit::application::ApplicationHandler;
use winit::event::{ElementState, WindowEvent};
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::keyboard::PhysicalKey;
use winit::window::{Window, WindowId};

use super::input_map;
use crate::backend::vulkan::caps;
use crate::backend::vulkan::engine::dmabuf_export::{ImportedDmabufImage, SCANOUT_EXPORT_RING};
use crate::backend::vulkan::engine::DrawError;
use crate::runtime::host::HostAction;

/// Consecutive CPU-staging presents (after direct present was established) that
/// trip the silent-revert alarm. ~240 presents is a couple seconds even at
/// 120 Hz — far longer than the zero-to-one staging frame a geometry change can
/// cause (verified: staging stays frozen across fullscreen/video transitions),
/// so a trip means a real regression, not a transient.
const REVERT_ALARM_RUN: u32 = 240;
const ENGINE_WINDOW_REDRAW_POLL: std::time::Duration = std::time::Duration::from_millis(2);
/// How long a guest-driven native resize request may stay unmatched by a
/// winit `Resized` event before the always-on alarm names it. Live requests
/// apply within single-digit milliseconds; one second means AppKit refused or
/// clamped the size and the window is presenting letterboxed instead.
#[cfg(target_os = "macos")]
const GUEST_RESIZE_WARN_AFTER: std::time::Duration = std::time::Duration::from_secs(1);

/// Window creation parameters.
#[derive(Clone, Debug)]
pub struct WindowConfig {
    pub title: String,
    pub width: u32,
    pub height: u32,
}

impl Default for WindowConfig {
    fn default() -> Self {
        Self {
            title: "Reims vGPU".to_string(),
            width: 1280,
            height: 800,
        }
    }
}

/// Called on the window thread for each input [`HostAction`] the window
/// produces. The implementation pushes onto the device prompt queue and wakes
/// the delivery BH (both thread-safe), so guest input flows without the device
/// lock.
pub type InputSink = Arc<dyn Fn(HostAction) + Send + Sync>;

/// A finished frame's zero-copy dmabuf source (direct-present route B). The
/// engine exported one of its ring slots into this dmabuf; the window imports it
/// once per `ring_idx` and re-blits the cached import while the engine writes
/// fresh content into that slot's shared memory. `fd=None` means the window has
/// already acknowledged importing this slot for the current geometry. When an fd
/// is present, the window `take`s it on first sight of this frame (importing
/// transfers fd ownership to Vulkan), and `Drop` closes it if the frame is
/// superseded before the window ever reads it.
pub struct FrameDmabuf {
    /// Exported dmabuf fd for this frame's ring slot; `None` once consumed or
    /// when the slot was already imported for this geometry.
    pub fd: Mutex<Option<std::os::fd::OwnedFd>>,
    /// LINEAR row pitch of the exported image (currently informational — the
    /// import re-derives its own subresource layout).
    pub pitch: u64,
    /// Which engine ring slot this fd backs, so the window caches one import per
    /// slot and re-blits the right one.
    pub ring_idx: usize,
    /// Shared import-success latch for this host window.
    pub import_ack: Arc<DirectPresentImportAck>,
}

/// The latest guest frame to present (BGRA8, tightly packed `width*height*4`).
/// `None` until the first present capture; the window clears to a flat color
/// until then. Not `Clone`: it may own a single-consume dmabuf fd, and it is
/// shared via `Arc` (never deep-cloned).
pub struct Frame {
    /// Monotonic publish sequence (assigned by the device when it writes a new
    /// frame). A static desktop publishes a new frame only when content changes.
    /// Linux re-blits its prepared dmabuf/staging source each vblank but prepares
    /// it only when `seq` advances. macOS submits the engine resident only when
    /// `seq` advances or the window resizes, so an unchanged desktop does not
    /// contend with guest render work for the engine queue.
    /// Wrap-around is harmless: a collision at most skips one prepare (the source
    /// still holds valid content).
    pub seq: u64,
    pub width: u32,
    pub height: u32,
    pub bgra: Vec<u8>,
    /// Zero-copy source, when the engine exported this frame's resident target.
    /// The window prefers it (no CPU upload) and falls back to `bgra` when absent
    /// or when its device cannot import dmabufs.
    pub dmabuf: Option<FrameDmabuf>,
    /// Engine-resident source for same-device MoltenVK presentation.
    pub resident: Option<crate::backend::vulkan::engine::WindowPresentSource>,
}

/// Shared slot the device writes and the window reads (latest-wins). The frame
/// is `Arc`-wrapped so the window's per-vblank read is a refcount bump, not an
/// 8 MiB deep copy of an unchanged frame.
pub type FrameSlot = Arc<Mutex<Option<Arc<Frame>>>>;

/// Whether a present must upload `incoming`'s frame into the staging image.
/// True unless the staging image already holds that exact `seq` — the seq-gated
/// fast path that elides the per-vblank full-frame re-upload of unchanged
/// content. `staged` is `None` before the first upload or after a staging
/// recreate, which always forces an upload.
fn needs_staging_upload(staged: Option<u64>, incoming: u64) -> bool {
    staged != Some(incoming)
}

#[cfg(target_os = "macos")]
fn needs_engine_present(
    presented: Option<u64>,
    redraw_required: bool,
    incoming: Option<u64>,
) -> bool {
    redraw_required || presented != incoming
}

/// Whether a newly observed guest frame geometry should request a native
/// content resize: only on a geometry change, and only when the window does
/// not already match. User-driven host resizing stays untouched until the
/// guest picks another mode.
#[cfg(target_os = "macos")]
fn guest_resize_request(
    observed: Option<(u32, u32)>,
    incoming: (u32, u32),
    window: (u32, u32),
) -> bool {
    observed != Some(incoming) && incoming != window
}

/// A guest-driven native resize not yet confirmed by a matching `Resized`
/// event. While it is outstanding the window holds its previous drawable
/// (guest boot presents are seconds apart, so a mismatched interim present
/// would stay on screen that long); the hold is bounded by
/// [`GUEST_RESIZE_WARN_AFTER`], after which the request is dropped with a
/// fail-visible line and presentation resumes letterboxed.
#[cfg(target_os = "macos")]
struct PendingGuestResize {
    target: (u32, u32),
    requested_at: std::time::Instant,
}

fn direct_present_degrade_line(reason: &'static str, detail: String) -> String {
    format!("direct_present_degrade reason={reason} {detail}")
}

fn direct_present_source_line(
    presents: u64,
    uploads: u64,
    dmabuf_blits: u64,
    staging_blits: u64,
    fresh_imports: u64,
    redundant_fds: u64,
) -> String {
    format!(
        "direct_present_source presents={presents} uploads={uploads} dmabuf_blits={dmabuf_blits} \
         staging_blits={staging_blits} fresh_imports={fresh_imports} \
         redundant_fds={redundant_fds}"
    )
}

/// What the present should blit into the acquired swapchain image this frame.
#[derive(Clone, Copy)]
enum BlitSource {
    /// No usable frame yet — clear to the slate color.
    Slate,
    /// Blit the CPU staging image (the guest frame was uploaded into it).
    Staging { dmabuf_fallback: bool },
    /// Blit the imported engine dmabuf at this ring slot (zero-copy path).
    Imported(usize),
}

fn counts_direct_present_revert_alarm(
    dmabuf_blits: u64,
    revert_logged: bool,
    src: BlitSource,
) -> bool {
    dmabuf_blits > 0
        && !revert_logged
        && matches!(
            src,
            BlitSource::Staging {
                dmabuf_fallback: true
            }
        )
}

fn next_direct_present_revert_run(
    staging_run: u32,
    dmabuf_blits: u64,
    revert_logged: bool,
    src: BlitSource,
) -> u32 {
    if counts_direct_present_revert_alarm(dmabuf_blits, revert_logged, src) {
        staging_run.saturating_add(1)
    } else {
        0
    }
}

/// Shared flag the device sets to ask the window to close (VM teardown). The
/// event loop polls it in `about_to_wait` and exits promptly. Distinct from a
/// UI close (which the window originates); either way the thread ends and its
/// Vulkan objects tear down before the join returns.
pub type StopFlag = Arc<AtomicBool>;

/// Set by the window thread once its Vulkan device is up, to `true` iff that
/// device enabled the dmabuf-import extensions. The device reads it to decide
/// whether the zero-copy resident export is worth doing: on a window whose
/// device cannot import (e.g. an iGPU lacking the exts), exporting would be a
/// wasted GPU blit per present, so the device skips it and publishes only the
/// CPU frame. Starts `false` (unknown → CPU path) and latches once at startup.
pub type ImportCapableFlag = Arc<AtomicBool>;

/// Set after the native window and all of its Vulkan objects have torn down.
/// QEMU's backend teardown waits for this before destroying shared GPU state.
pub type ExitedFlag = Arc<AtomicBool>;

/// Window-side acknowledgement that a direct-present dmabuf ring slot was
/// successfully imported for a specific guest geometry.
///
/// The producer uses this only to skip duplicating an fd for a slot that the
/// window has already imported. A false negative is harmless (one redundant fd);
/// a true value is published only after `import_bgra_dmabuf_image` succeeds.
#[derive(Debug, Default)]
pub struct DirectPresentImportAck {
    geom: AtomicU64,
    slots: AtomicU64,
}

impl DirectPresentImportAck {
    fn pack_geom(width: u32, height: u32) -> u64 {
        ((width as u64) << 32) | height as u64
    }

    pub fn is_imported(&self, width: u32, height: u32, ring_idx: usize) -> bool {
        if ring_idx >= 64 {
            return false;
        }
        let geom = Self::pack_geom(width, height);
        self.geom.load(Ordering::Acquire) == geom
            && (self.slots.load(Ordering::Acquire) & (1u64 << ring_idx)) != 0
    }

    pub fn mark_imported(&self, width: u32, height: u32, ring_idx: usize) {
        if ring_idx >= 64 {
            return;
        }
        let geom = Self::pack_geom(width, height);
        if self.geom.load(Ordering::Acquire) != geom {
            self.slots.store(0, Ordering::Release);
            self.geom.store(geom, Ordering::Release);
        }
        self.slots.fetch_or(1u64 << ring_idx, Ordering::AcqRel);
    }
}

/// Errors from bringing up or running the window.
///
/// One variant per distinct check, so each names itself in `/tmp/reims-vgpu-fail.log`
/// through [`crate::observe::Decline`]. The old three String variants
/// (`EventLoop`/`Vulkan`/`Handle`) collapsed twenty-six checks into three grep
/// prefixes: the Linux `VkState::new` bring-up alone hid seventeen ash calls
/// behind `Vulkan(String)`, and its two `semaphore: {e}` sites were the same
/// prose for two different objects. The specific check now lives in the slug;
/// the raw driver/winit string rides along as a whitespace-safe `detail=` field.
///
/// `#[allow(dead_code)]` because the enum is shared across two mutually
/// exclusive platform paths — the `Attach*` variants are macOS-only (the engine
/// swapchain attach) and the `Vk*` variants build the window's own ash device on
/// Linux — so on either target roughly half are unconstructed.
#[allow(dead_code)]
#[derive(Debug)]
pub enum WindowError {
    /// `EventLoop::build()` failed (winit).
    EventLoopBuild(String),
    /// `run_app` returned an error on the off-main-thread `run()` path.
    RunApp(String),
    /// `run_app` returned an error on the macOS main-thread loop.
    MainLoopRun(String),
    /// A second device tried to claim the single process window.
    AlreadyOwned { owner: u64 },
    /// `run_main_thread` found no registered window for the device.
    NoRegisteredWindow { id: u64 },
    /// `run_main_thread` was asked to run a window owned by another device.
    WrongOwner { owner: u64, requested: u64 },
    /// `resumed`: winit could not create the native window (shared step, both
    /// platforms) — the bring-up cannot proceed past this.
    CreateNativeWindow(String),
    /// macOS engine-attach: the window's display handle was unavailable.
    AttachDisplayHandle(String),
    /// macOS engine-attach: the window's window handle was unavailable.
    AttachWindowHandle(String),
    /// macOS engine-attach: `window_present_attach` (engine swapchain) failed.
    AttachEngine(String),
    /// Linux bring-up: the Vulkan loader failed to load.
    VkLoadLoader(String),
    /// Linux bring-up: the window's display handle was unavailable.
    VkDisplayHandle(String),
    /// Linux bring-up: the window's window handle was unavailable.
    VkWindowHandle(String),
    /// Linux bring-up: the required surface extensions could not be enumerated.
    VkRequiredExts(String),
    /// Linux bring-up: enumerating instance extensions failed.
    VkEnumerateInstanceExts(String),
    /// Linux bring-up: `create_instance` failed.
    VkCreateInstance(String),
    /// Linux bring-up: `create_surface` failed.
    VkCreateSurface(String),
    /// Linux bring-up: enumerating physical devices failed.
    VkEnumeratePhysicalDevices(String),
    /// Linux bring-up: no present-capable device met the Vulkan floor.
    VkNoUsableDevice(String),
    /// Linux bring-up: enumerating device extensions failed.
    VkEnumerateDeviceExts(String),
    /// Linux bring-up: the chosen device does not advertise `VK_KHR_swapchain`.
    VkNoSwapchainExtension,
    /// Linux bring-up: `create_device` failed.
    VkCreateDevice(String),
    /// Linux bring-up: `create_command_pool` failed.
    VkCommandPool(String),
    /// Linux bring-up: `allocate_command_buffers` failed.
    VkAllocCmd(String),
    /// Linux bring-up: the image-available semaphore could not be created.
    VkSemaphoreImageAvailable(String),
    /// Linux bring-up: the render-finished semaphore could not be created.
    VkSemaphoreRenderFinished(String),
    /// Linux bring-up: the in-flight fence could not be created.
    VkFence(String),
    /// Linux present loop: acquiring the next swapchain image failed.
    PresentAcquire(vk::Result),
    /// Linux present loop: resetting the in-flight fence failed.
    PresentResetFence(vk::Result),
    /// Linux present loop: resetting its command buffer failed.
    PresentResetCommandBuffer(vk::Result),
    /// Linux present loop: beginning its command buffer failed.
    PresentBeginCommandBuffer(vk::Result),
    /// Linux present loop: ending its command buffer failed.
    PresentEndCommandBuffer(vk::Result),
    /// Linux present loop: submitting its blit failed.
    PresentQueueSubmit(vk::Result),
    /// Linux present loop: presenting the acquired image failed.
    PresentQueue(vk::Result),
    /// CPU fallback: creating the persistently mapped staging image failed.
    StagingCreateImage(vk::Result),
    /// CPU fallback: no upload-compatible memory type exists.
    StagingMemoryTypeUnavailable { type_bits: u32 },
    /// CPU fallback: allocating staging-image memory failed.
    StagingAllocateMemory { bytes: u64, result: vk::Result },
    /// CPU fallback: binding staging-image memory failed.
    StagingBindMemory(vk::Result),
    /// CPU fallback: mapping staging-image memory failed.
    StagingMapMemory { bytes: u64, result: vk::Result },
    /// Direct present: the window device did not enable dmabuf import.
    DmabufImportExtensionsMissing,
    /// Direct present: the producer named a slot outside the fixed import ring.
    DmabufRingIndexOutOfRange { ring_idx: usize, ring_len: usize },
    /// Direct present: the engine's typed dmabuf import declined.
    DmabufImport(DrawError),
}

impl WindowError {
    /// The raw driver/winit detail this error carries, if any — for `Display`
    /// and the diagnostic `eprintln!`, which want the string verbatim rather
    /// than the whitespace-collapsed form the log field uses.
    fn detail(&self) -> Option<&str> {
        match self {
            Self::EventLoopBuild(d)
            | Self::RunApp(d)
            | Self::MainLoopRun(d)
            | Self::CreateNativeWindow(d)
            | Self::AttachDisplayHandle(d)
            | Self::AttachWindowHandle(d)
            | Self::AttachEngine(d)
            | Self::VkLoadLoader(d)
            | Self::VkDisplayHandle(d)
            | Self::VkWindowHandle(d)
            | Self::VkRequiredExts(d)
            | Self::VkEnumerateInstanceExts(d)
            | Self::VkCreateInstance(d)
            | Self::VkCreateSurface(d)
            | Self::VkEnumeratePhysicalDevices(d)
            | Self::VkNoUsableDevice(d)
            | Self::VkEnumerateDeviceExts(d)
            | Self::VkCreateDevice(d)
            | Self::VkCommandPool(d)
            | Self::VkAllocCmd(d)
            | Self::VkSemaphoreImageAvailable(d)
            | Self::VkSemaphoreRenderFinished(d)
            | Self::VkFence(d) => Some(d),
            Self::AlreadyOwned { .. }
            | Self::NoRegisteredWindow { .. }
            | Self::WrongOwner { .. }
            | Self::VkNoSwapchainExtension
            | Self::PresentAcquire(_)
            | Self::PresentResetFence(_)
            | Self::PresentResetCommandBuffer(_)
            | Self::PresentBeginCommandBuffer(_)
            | Self::PresentEndCommandBuffer(_)
            | Self::PresentQueueSubmit(_)
            | Self::PresentQueue(_)
            | Self::StagingCreateImage(_)
            | Self::StagingMemoryTypeUnavailable { .. }
            | Self::StagingAllocateMemory { .. }
            | Self::StagingBindMemory(_)
            | Self::StagingMapMemory { .. }
            | Self::DmabufImportExtensionsMissing
            | Self::DmabufRingIndexOutOfRange { .. }
            | Self::DmabufImport(_) => None,
        }
    }
}

/// Collapse whitespace runs to single `_` so a driver/winit string is safe as a
/// log field value ([`crate::observe::Emit`] splits the line on spaces).
fn detail_field(s: &str) -> String {
    s.split_whitespace().collect::<Vec<_>>().join("_")
}

impl crate::observe::Decline for WindowError {
    fn slug(&self) -> &'static str {
        match self {
            Self::EventLoopBuild(_) => "window_event_loop_build",
            Self::RunApp(_) => "window_run_app",
            Self::MainLoopRun(_) => "window_main_loop_run",
            Self::AlreadyOwned { .. } => "window_already_owned",
            Self::NoRegisteredWindow { .. } => "window_no_registered_window",
            Self::WrongOwner { .. } => "window_wrong_owner",
            Self::CreateNativeWindow(_) => "window_create_native_window",
            Self::AttachDisplayHandle(_) => "window_attach_display_handle",
            Self::AttachWindowHandle(_) => "window_attach_window_handle",
            Self::AttachEngine(_) => "window_attach_engine",
            Self::VkLoadLoader(_) => "window_vk_load_loader",
            Self::VkDisplayHandle(_) => "window_vk_display_handle",
            Self::VkWindowHandle(_) => "window_vk_window_handle",
            Self::VkRequiredExts(_) => "window_vk_required_exts",
            Self::VkEnumerateInstanceExts(_) => "window_vk_enumerate_instance_exts",
            Self::VkCreateInstance(_) => "window_vk_create_instance",
            Self::VkCreateSurface(_) => "window_vk_create_surface",
            Self::VkEnumeratePhysicalDevices(_) => "window_vk_enumerate_physical_devices",
            Self::VkNoUsableDevice(_) => "window_vk_no_usable_device",
            Self::VkEnumerateDeviceExts(_) => "window_vk_enumerate_device_exts",
            Self::VkNoSwapchainExtension => "window_vk_no_swapchain_extension",
            Self::VkCreateDevice(_) => "window_vk_create_device",
            Self::VkCommandPool(_) => "window_vk_command_pool",
            Self::VkAllocCmd(_) => "window_vk_alloc_cmd",
            Self::VkSemaphoreImageAvailable(_) => "window_vk_semaphore_image_available",
            Self::VkSemaphoreRenderFinished(_) => "window_vk_semaphore_render_finished",
            Self::VkFence(_) => "window_vk_fence",
            Self::PresentAcquire(_) => "window_present_acquire",
            Self::PresentResetFence(_) => "window_present_reset_fence",
            Self::PresentResetCommandBuffer(_) => "window_present_reset_command_buffer",
            Self::PresentBeginCommandBuffer(_) => "window_present_begin_command_buffer",
            Self::PresentEndCommandBuffer(_) => "window_present_end_command_buffer",
            Self::PresentQueueSubmit(_) => "window_present_queue_submit",
            Self::PresentQueue(_) => "window_present_queue",
            Self::StagingCreateImage(_) => "window_staging_create_image",
            Self::StagingMemoryTypeUnavailable { .. } => "window_staging_memory_type_unavailable",
            Self::StagingAllocateMemory { .. } => "window_staging_allocate_memory",
            Self::StagingBindMemory(_) => "window_staging_bind_memory",
            Self::StagingMapMemory { .. } => "window_staging_map_memory",
            Self::DmabufImportExtensionsMissing => "window_dmabuf_import_extensions_missing",
            Self::DmabufRingIndexOutOfRange { .. } => "window_dmabuf_ring_index_out_of_range",
            Self::DmabufImport(reason) => reason.slug(),
        }
    }

    fn fields(&self) -> Vec<(&'static str, String)> {
        match self {
            Self::AlreadyOwned { owner } => vec![("owner", owner.to_string())],
            Self::NoRegisteredWindow { id } => vec![("id", id.to_string())],
            Self::WrongOwner { owner, requested } => vec![
                ("owner", owner.to_string()),
                ("requested", requested.to_string()),
            ],
            Self::PresentAcquire(result)
            | Self::PresentResetFence(result)
            | Self::PresentResetCommandBuffer(result)
            | Self::PresentBeginCommandBuffer(result)
            | Self::PresentEndCommandBuffer(result)
            | Self::PresentQueueSubmit(result)
            | Self::PresentQueue(result)
            | Self::StagingCreateImage(result)
            | Self::StagingBindMemory(result) => {
                vec![("vk_result", result.as_raw().to_string())]
            }
            Self::StagingMemoryTypeUnavailable { type_bits } => {
                vec![("type_bits", format!("{type_bits:#x}"))]
            }
            Self::StagingAllocateMemory { bytes, result }
            | Self::StagingMapMemory { bytes, result } => vec![
                ("bytes", bytes.to_string()),
                ("vk_result", result.as_raw().to_string()),
            ],
            Self::DmabufRingIndexOutOfRange { ring_idx, ring_len } => vec![
                ("ring_idx", ring_idx.to_string()),
                ("ring_len", ring_len.to_string()),
            ],
            Self::DmabufImport(reason) => reason.fields(),
            other => match other.detail() {
                Some(d) => vec![("detail", detail_field(d))],
                None => Vec::new(),
            },
        }
    }
}

impl std::fmt::Display for WindowError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        use crate::observe::Decline as _;
        match self.detail() {
            Some(d) => write!(f, "{}: {d}", self.slug()),
            None => write!(f, "{}", self.slug()),
        }
    }
}

impl std::error::Error for WindowError {}

/// Spawn the window on a dedicated thread and return its join handle. The thread
/// owns the winit event loop for its lifetime; it exits when the window closes.
pub fn spawn(
    config: WindowConfig,
    on_input: InputSink,
    frames: FrameSlot,
    stop: StopFlag,
    import_capable: ImportCapableFlag,
) -> std::thread::JoinHandle<Result<(), WindowError>> {
    std::thread::Builder::new()
        .name("reims-vgpu-window".to_string())
        .spawn(move || run(config, on_input, frames, stop, import_capable))
        .expect("spawn reims-vgpu-window thread")
}

/// Run the window event loop on the calling thread (blocks until the window
/// closes). Prefer [`spawn`]; call this directly only if you already own a
/// suitable thread.
pub fn run(
    config: WindowConfig,
    on_input: InputSink,
    frames: FrameSlot,
    stop: StopFlag,
    import_capable: ImportCapableFlag,
) -> Result<(), WindowError> {
    let event_loop = build_event_loop()?;
    let mut app = App {
        config,
        on_input,
        frames,
        stop,
        import_capable,
        closed_sent: false,
        window: None,
        vk: None,
        cursor: (0, 0),
        #[cfg(target_os = "macos")]
        first_engine_present_logged: false,
        #[cfg(target_os = "macos")]
        first_engine_guest_logged: false,
        #[cfg(target_os = "macos")]
        engine_error_logged: false,
        next_engine_redraw: std::time::Instant::now(),
        #[cfg(target_os = "macos")]
        last_engine_seq: None,
        #[cfg(target_os = "macos")]
        engine_redraw_required: true,
        #[cfg(target_os = "macos")]
        guest_extent: None,
        #[cfg(target_os = "macos")]
        pending_guest_resize: None,
    };
    event_loop
        .run_app(&mut app)
        .map_err(|e| WindowError::RunApp(e.to_string()))
}

#[cfg(target_os = "macos")]
struct MainThreadWindow {
    id: u64,
    event_loop: EventLoop<()>,
    app: App,
    exited: ExitedFlag,
}

#[cfg(target_os = "macos")]
thread_local! {
    static MAIN_THREAD_WINDOW: RefCell<Option<MainThreadWindow>> = const { RefCell::new(None) };
}

/// Create the macOS host window on the process main thread.
///
/// AppKit requires event-loop creation and dispatch on the process main thread.
/// QEMU owns that thread, so the thin shim calls this at device realize and
/// later makes [`run_main_thread`] its blocking UI entry. Only one display
/// window may exist in a process; repeated starts for the same device are
/// idempotent.
#[cfg(target_os = "macos")]
pub fn start_main_thread(
    id: u64,
    config: WindowConfig,
    on_input: InputSink,
    frames: FrameSlot,
    stop: StopFlag,
    import_capable: ImportCapableFlag,
    exited: ExitedFlag,
) -> Result<(), WindowError> {
    MAIN_THREAD_WINDOW.with(|cell| {
        let mut slot = cell.borrow_mut();
        if let Some(existing) = slot.as_ref() {
            return if existing.id == id {
                Ok(())
            } else {
                Err(WindowError::AlreadyOwned { owner: existing.id })
            };
        }
        let event_loop = build_event_loop()?;
        let app = App {
            config,
            on_input,
            frames,
            stop,
            import_capable,
            closed_sent: false,
            window: None,
            vk: None,
            cursor: (0, 0),
            first_engine_present_logged: false,
            first_engine_guest_logged: false,
            engine_error_logged: false,
            next_engine_redraw: std::time::Instant::now(),
            last_engine_seq: None,
            engine_redraw_required: true,
            guest_extent: None,
            pending_guest_resize: None,
        };
        *slot = Some(MainThreadWindow {
            id,
            event_loop,
            app,
            exited,
        });
        Ok(())
    })
}

/// Run the registered macOS window as QEMU's process-main UI loop.
///
/// QEMU runs emulation on its `qemu_main` thread on Darwin, leaving the process
/// main thread to this blocking AppKit loop. The exit flag is published only
/// after `run_app` returns and destroys the app's native Vulkan state.
#[cfg(target_os = "macos")]
pub fn run_main_thread(id: u64) -> Result<(), WindowError> {
    MAIN_THREAD_WINDOW.with(|cell| {
        let Some(mut window) = cell.borrow_mut().take() else {
            return Err(WindowError::NoRegisteredWindow { id });
        };
        if window.id != id {
            let owner = window.id;
            *cell.borrow_mut() = Some(window);
            return Err(WindowError::WrongOwner {
                owner,
                requested: id,
            });
        }
        let result = window
            .event_loop
            .run_app(&mut window.app)
            .map_err(|error| WindowError::MainLoopRun(error.to_string()));
        window.exited.store(true, Ordering::Release);
        result
    })
}

/// Build an event loop that may run off the main thread (QEMU owns the main
/// thread). X11 and Wayland both allow it via their platform extension.
fn build_event_loop() -> Result<EventLoop<()>, WindowError> {
    let mut builder = EventLoop::builder();
    #[cfg(all(unix, not(target_os = "macos")))]
    {
        use winit::platform::wayland::EventLoopBuilderExtWayland;
        use winit::platform::x11::EventLoopBuilderExtX11;
        // Fully-qualified so the two identically-named ext methods don't clash;
        // each sets its own backend's any-thread flag (only the active one runs).
        EventLoopBuilderExtX11::with_any_thread(&mut builder, true);
        EventLoopBuilderExtWayland::with_any_thread(&mut builder, true);
    }
    builder
        .build()
        .map_err(|e| WindowError::EventLoopBuild(e.to_string()))
}

struct App {
    config: WindowConfig,
    on_input: InputSink,
    frames: FrameSlot,
    /// Set by the device to request teardown; polled in `about_to_wait`.
    stop: StopFlag,
    /// Latched `true` once the Vulkan device is up iff it can import dmabufs, so
    /// the device only exports resident frames a window that can consume them.
    import_capable: ImportCapableFlag,
    /// True once a `WindowClosed` action has been emitted (UI close), so the
    /// shutdown request is sent exactly once.
    closed_sent: bool,
    /// MUST be declared before `window`: Rust drops fields in declaration order,
    /// and `VkState::drop` destroys the swapchain + `VkSurfaceKHR`, which the
    /// driver services through the native (Wayland/X) surface owned by `window`.
    /// Dropping the window first leaves those calls marshalling to a dead
    /// `wl_proxy` → SIGSEGV inside the driver's WSI. `exiting()` also tears them
    /// down explicitly in this order; this ordering is the backstop.
    vk: Option<VkState>,
    window: Option<Arc<Window>>,
    /// Last cursor position in window pixels (for absolute pointer moves).
    cursor: (u32, u32),
    #[cfg(target_os = "macos")]
    first_engine_present_logged: bool,
    #[cfg(target_os = "macos")]
    first_engine_guest_logged: bool,
    #[cfg(target_os = "macos")]
    engine_error_logged: bool,
    /// Deadline for the next periodic `request_redraw()` in `about_to_wait`.
    /// macOS uses it alongside the native-resize handshake below; non-macOS
    /// uses it alone as the redraw pump, since there `RedrawRequested`'s
    /// self-chain (see the handler) can stall until an externally generated
    /// X11 event unsticks it.
    next_engine_redraw: std::time::Instant,
    #[cfg(target_os = "macos")]
    last_engine_seq: Option<u64>,
    #[cfg(target_os = "macos")]
    engine_redraw_required: bool,
    /// Last guest DisplaySwap geometry the window observed. Drives the
    /// once-per-mode-change native resize request and the pointer-to-guest
    /// viewport transform ([`super::viewport`]).
    #[cfg(target_os = "macos")]
    guest_extent: Option<(u32, u32)>,
    /// Outstanding guest-driven native resize, kept only for the fail-visible
    /// `native_resize_not_applied` alarm — never a presentation gate.
    #[cfg(target_os = "macos")]
    pending_guest_resize: Option<PendingGuestResize>,
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }
        let attrs = Window::default_attributes()
            .with_title(self.config.title.clone())
            .with_inner_size(winit::dpi::PhysicalSize::new(
                self.config.width,
                self.config.height,
            ));
        let window = match event_loop.create_window(attrs) {
            Ok(w) => Arc::new(w),
            Err(e) => {
                crate::observe::Emit::decline(
                    "host_window_init",
                    &WindowError::CreateNativeWindow(e.to_string()),
                )
                .fail();
                eprintln!("reims-vgpu-window: create_window failed: {e}");
                self.request_shutdown();
                event_loop.exit();
                return;
            }
        };
        #[cfg(target_os = "macos")]
        {
            let attach = window
                .display_handle()
                .map_err(|error| WindowError::AttachDisplayHandle(error.to_string()))
                .and_then(|display| {
                    window
                        .window_handle()
                        .map_err(|error| WindowError::AttachWindowHandle(error.to_string()))
                        .map(|handle| (display.as_raw(), handle.as_raw()))
                })
                .and_then(|(display, handle)| {
                    let size = window.inner_size();
                    crate::backend::vulkan::engine::window_present_attach(
                        display,
                        handle,
                        size.width.max(1),
                        size.height.max(1),
                    )
                    .map_err(|error| WindowError::AttachEngine(error.to_string()))
                });
            match attach {
                Ok(()) => {
                    self.import_capable.store(true, Ordering::Release);
                    window.request_redraw();
                    self.window = Some(window);
                }
                Err(error) => {
                    crate::observe::Emit::decline("host_window_init", &error).fail();
                    eprintln!("reims-vgpu-window: engine swapchain init failed: {error}");
                    self.request_shutdown();
                    event_loop.exit();
                }
            }
        }
        #[cfg(not(target_os = "macos"))]
        match unsafe { VkState::new(&window) } {
            Ok(vk) => {
                // Tell the device whether this window can consume exported
                // dmabufs, so it only pays the export blit when we'll use it.
                self.import_capable
                    .store(vk.import_fd_loader.is_some(), Ordering::Relaxed);
                self.vk = Some(vk);
                // Kick the first frame; RedrawRequested re-arms each subsequent
                // one, so without this the window would never draw.
                window.request_redraw();
                self.window = Some(window);
            }
            Err(e) => {
                crate::observe::Emit::decline("host_window_init", &e).fail();
                eprintln!("reims-vgpu-window: Vulkan init failed: {e}");
                self.request_shutdown();
                event_loop.exit();
            }
        }
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => {
                // The window IS the VM's display, so a UI close means "shut the
                // VM down". Emit WindowClosed once (the shim turns it into a
                // shutdown request) before tearing the window down. The device
                // will also set `stop` from its exit path; either order is fine.
                self.request_shutdown();
                event_loop.exit();
            }
            WindowEvent::Resized(size) => {
                #[cfg(target_os = "macos")]
                {
                    let applied = (size.width.max(1), size.height.max(1));
                    crate::backend::vulkan::engine::window_present_resize(applied.0, applied.1);
                    self.engine_redraw_required = true;
                    self.note_guest_resize_applied(applied);
                }
                #[cfg(not(target_os = "macos"))]
                if let Some(vk) = self.vk.as_mut() {
                    unsafe { vk.recreate_swapchain(size.width.max(1), size.height.max(1)) };
                }
            }
            WindowEvent::KeyboardInput { event, .. } => {
                if let PhysicalKey::Code(code) = event.physical_key {
                    if let Some(evdev) = input_map::keycode_to_evdev(code) {
                        let down = event.state == ElementState::Pressed;
                        (self.on_input)(HostAction::input_key(evdev, down));
                    }
                }
            }
            WindowEvent::CursorMoved { position, .. } => {
                self.pointer_move((position.x, position.y));
            }
            WindowEvent::MouseInput { state, button, .. } => {
                if let Some(btn) = input_map::mouse_button(button) {
                    (self.on_input)(HostAction::input_pointer_button(
                        btn,
                        state == ElementState::Pressed,
                    ));
                }
            }
            WindowEvent::MouseWheel { delta, .. } => {
                for action in input_map::scroll_actions(delta) {
                    (self.on_input)(action);
                }
            }
            WindowEvent::RedrawRequested => {
                self.draw();
                #[cfg(not(target_os = "macos"))]
                if let Some(w) = self.window.as_ref() {
                    w.request_redraw();
                }
            }
            _ => {}
        }
    }

    fn exiting(&mut self, _event_loop: &ActiveEventLoop) {
        // Tear Vulkan down while the native window is still alive. VkState::drop
        // destroys the swapchain and VkSurfaceKHR, and the driver services those
        // through the Wayland/X surface owned by `window`; releasing the window
        // first makes the driver marshal to a freed wl_proxy and crash. winit
        // calls this before the loop ends, so the ordering is explicit here
        // rather than relying on struct field order alone.
        #[cfg(target_os = "macos")]
        crate::backend::vulkan::engine::window_present_detach();
        #[cfg(not(target_os = "macos"))]
        {
            self.vk = None;
        }
        self.window = None;
    }

    fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
        // The device sets `stop` on VM teardown. The window redraws every frame
        // (throttled by the FIFO swapchain to vblank), so this runs ~per frame
        // and picks the request up within one frame — then the loop exits and
        // `VkState::drop` tears the window's Vulkan objects down on this thread
        // before the device's join returns.
        if self.stop.load(Ordering::Relaxed) {
            event_loop.exit();
            return;
        }
        #[cfg(target_os = "macos")]
        if let Some(window) = self.window.as_ref() {
            if let Some(pending) = self.pending_guest_resize.as_ref() {
                if pending.requested_at.elapsed() >= GUEST_RESIZE_WARN_AFTER {
                    let actual = window.inner_size();
                    crate::observe::fail(format!(
                        "host_window_guest_resize FAIL reason=native_resize_not_applied \
                         requested={}x{} actual={}x{}",
                        pending.target.0, pending.target.1, actual.width, actual.height
                    ));
                    // Drop the request so presentation resumes (letterboxed
                    // into whatever drawable exists) instead of holding.
                    self.pending_guest_resize = None;
                    self.engine_redraw_required = true;
                }
            }
            let now = std::time::Instant::now();
            if now >= self.next_engine_redraw {
                window.request_redraw();
                self.next_engine_redraw = now + ENGINE_WINDOW_REDRAW_POLL;
            }
            event_loop.set_control_flow(winit::event_loop::ControlFlow::WaitUntil(
                self.next_engine_redraw,
            ));
        }
        // Non-macOS presentation has no native-resize handshake to drive, but
        // it still needs a redraw pump: `WindowEvent::RedrawRequested` below
        // re-requests itself each frame, a self-chain that depends on the
        // windowing system continuing to deliver events. Under X11 that chain
        // can stall — observed as a live guest whose window only repaints
        // after a click, resize, or other externally generated event unsticks
        // it. Poll on the same cadence macOS already uses so presentation
        // does not depend on X11 event delivery being reliable.
        #[cfg(not(target_os = "macos"))]
        if let Some(window) = self.window.as_ref() {
            let now = std::time::Instant::now();
            if now >= self.next_engine_redraw {
                window.request_redraw();
                self.next_engine_redraw = now + ENGINE_WINDOW_REDRAW_POLL;
            }
            event_loop.set_control_flow(winit::event_loop::ControlFlow::WaitUntil(
                self.next_engine_redraw,
            ));
        }
    }
}

impl App {
    fn request_shutdown(&mut self) {
        if !self.closed_sent {
            (self.on_input)(HostAction::window_closed());
            self.closed_sent = true;
        }
    }

    fn surface_dims(&self) -> (u32, u32) {
        #[cfg(target_os = "macos")]
        if let Some(window) = self.window.as_ref() {
            let size = window.inner_size();
            return (size.width.max(1), size.height.max(1));
        }
        self.vk
            .as_ref()
            .map(|v| (v.extent.width, v.extent.height))
            .unwrap_or((self.config.width, self.config.height))
    }

    /// Emit an absolute pointer move. On macOS with a known guest geometry the
    /// position maps through the presenter's aspect-fit viewport into guest
    /// pixels, so display placement and pointer translation move as one unit;
    /// otherwise the full-window coordinate space is forwarded unchanged.
    fn pointer_move(&mut self, position: (f64, f64)) {
        let (w, h) = self.surface_dims();
        #[cfg(target_os = "macos")]
        if let Some(guest) = self.guest_extent {
            let mapped = super::viewport::pointer_to_guest(position, (w, h), guest);
            self.cursor = mapped;
            (self.on_input)(HostAction::input_pointer_move(
                mapped.0, mapped.1, guest.0, guest.1,
            ));
            return;
        }
        self.cursor = (position.0.max(0.0) as u32, position.1.max(0.0) as u32);
        (self.on_input)(HostAction::input_pointer_move(
            self.cursor.0,
            self.cursor.1,
            w,
            h,
        ));
    }

    fn draw(&mut self) {
        #[cfg(target_os = "macos")]
        {
            self.draw_engine_window();
        }
        #[cfg(not(target_os = "macos"))]
        let Some(vk) = self.vk.as_mut() else {
            return;
        };
        #[cfg(not(target_os = "macos"))]
        let frame = self.frames.lock().ok().and_then(|g| g.clone());
        #[cfg(not(target_os = "macos"))]
        if let Err(e) = unsafe { vk.present(frame.as_deref()) } {
            crate::observe::Emit::decline("host_window_present", &e).fail_once(0);
            eprintln!("reims-vgpu-window: present failed: {e}");
        }
    }

    #[cfg(target_os = "macos")]
    fn draw_engine_window(&mut self) {
        let frame = self.frames.lock().ok().and_then(|guard| guard.clone());
        self.request_guest_geometry(frame.as_deref());
        if self.pending_guest_resize.is_some() {
            // A guest mode change is being applied to the native window
            // (normally single-digit milliseconds; bounded by the 1 s alarm).
            // Hold the previous drawable rather than letterboxing the
            // new-geometry frame into the outgoing swapchain — at boot the
            // next guest present can be seconds away, which would pin that
            // interim frame on screen.
            return;
        }
        let incoming_seq = frame.as_ref().map(|frame| frame.seq);
        if !needs_engine_present(
            self.last_engine_seq,
            self.engine_redraw_required,
            incoming_seq,
        ) {
            return;
        }
        let result = crate::backend::vulkan::engine::window_present_frame(
            frame.as_ref().and_then(|frame| frame.resident.as_ref()),
        );
        match result {
            Ok(crate::backend::vulkan::engine::WindowPresentOutcome::Busy) => {}
            Ok(crate::backend::vulkan::engine::WindowPresentOutcome::Presented {
                direct,
                width,
                height,
                swapchain_images,
                suboptimal,
            }) => {
                self.engine_error_logged = false;
                self.last_engine_seq = incoming_seq;
                // A suboptimal present armed a swapchain recreation; redraw
                // promptly so the corrected drawable replaces this one even if
                // no new guest frame arrives for seconds.
                self.engine_redraw_required = suboptimal;
                self.import_capable.store(true, Ordering::Release);
                if !self.first_engine_present_logged {
                    eprintln!(
                        "reims-vgpu-window: first frame presented \
                         ({width}x{height}, {swapchain_images} swapchain images)"
                    );
                    self.first_engine_present_logged = true;
                }
                if direct && frame.is_some() && !self.first_engine_guest_logged {
                    eprintln!(
                        "reims-vgpu-window: first guest frame presented via engine resident \
                         (same-device zero-copy)"
                    );
                    crate::observe::off(
                        "host_window_direct_present path=engine_resident status=live",
                    );
                    self.first_engine_guest_logged = true;
                }
            }
            Err(error) => {
                self.import_capable.store(false, Ordering::Release);
                if !self.engine_error_logged {
                    // The engine present rail's `DrawError` names its own reason
                    // — a `VkCall`'s `vk_window_*` slug, a `DrawReason` refusal,
                    // or `vk_engine_*_untyped` for the not-yet-typed variants.
                    // Emitting it typed keeps that slug the primary `reason=`
                    // rather than nesting it inside a coarse
                    // `reason=engine_resident_present error=...` double-reason.
                    crate::observe::Emit::decline("host_window_present", &error).fail();
                    eprintln!("reims-vgpu-window: engine resident present failed: {error}");
                    self.engine_error_logged = true;
                }
            }
        }
    }

    /// Track the accepted guest frame geometry and ask the native window to
    /// match a newly selected guest mode, once per change. The frame
    /// dimensions are protocol state — the same width/height that select the
    /// compositor resident — never a content heuristic. Presentation does not
    /// wait: until the resize applies, frames letterbox into the current
    /// drawable and pointer input maps through the same viewport.
    #[cfg(target_os = "macos")]
    fn request_guest_geometry(&mut self, frame: Option<&Frame>) {
        let Some(frame) = frame else { return };
        let incoming = (frame.width.max(1), frame.height.max(1));
        if self.guest_extent == Some(incoming) {
            return;
        }
        let Some(window) = self.window.as_ref() else {
            return;
        };
        let size = window.inner_size();
        let actual = (size.width.max(1), size.height.max(1));
        let request = guest_resize_request(self.guest_extent, incoming, actual);
        self.guest_extent = Some(incoming);
        if !request {
            return;
        }
        crate::observe::off(format!(
            "host_window_guest_resize status=requested from={}x{} to={}x{}",
            actual.0, actual.1, incoming.0, incoming.1
        ));
        self.pending_guest_resize = Some(PendingGuestResize {
            target: incoming,
            requested_at: std::time::Instant::now(),
        });
        let immediate =
            window.request_inner_size(winit::dpi::PhysicalSize::new(incoming.0, incoming.1));
        if let Some(applied) = immediate {
            // Applied synchronously — winit emits no later `Resized` for it.
            let applied = (applied.width.max(1), applied.height.max(1));
            crate::backend::vulkan::engine::window_present_resize(applied.0, applied.1);
            self.engine_redraw_required = true;
            self.note_guest_resize_applied(applied);
        }
    }

    /// Clear the outstanding guest resize once the window system confirms the
    /// exact target geometry (via `Resized` or a synchronous apply).
    #[cfg(target_os = "macos")]
    fn note_guest_resize_applied(&mut self, applied: (u32, u32)) {
        if self
            .pending_guest_resize
            .as_ref()
            .is_some_and(|pending| pending.target == applied)
        {
            crate::observe::off(format!(
                "host_window_guest_resize status=applied width={} height={}",
                applied.0, applied.1
            ));
            self.pending_guest_resize = None;
        }
    }
}

/// The Vulkan swapchain + per-frame objects for the window. Self-contained
/// instance/device (see module docs).
struct VkState {
    _entry: ash::Entry,
    instance: ash::Instance,
    surface_loader: ash::khr::surface::Instance,
    surface: vk::SurfaceKHR,
    pd: vk::PhysicalDevice,
    device: ash::Device,
    queue: vk::Queue,
    qfamily: u32,
    swapchain_loader: ash::khr::swapchain::Device,
    swapchain: vk::SwapchainKHR,
    images: Vec<vk::Image>,
    format: vk::Format,
    extent: vk::Extent2D,
    cmd_pool: vk::CommandPool,
    cmd: vk::CommandBuffer,
    image_available: vk::Semaphore,
    render_finished: vk::Semaphore,
    in_flight: vk::Fence,
    /// Latches after the first successful present so bring-up logs the live
    /// window exactly once (not per frame).
    first_present_logged: bool,
    /// Latches after the first present that carried a real guest [`Frame`] (not
    /// the pre-frame slate). Distinguishes "window is up" from "guest content is
    /// flowing end to end" in the boot log — diagnostic only.
    first_guest_frame_logged: bool,
    /// Latches after the first present that blitted the zero-copy engine dmabuf
    /// (route B). The `first guest frame` line above latches on whatever the
    /// FIRST frame used (often CPU staging during warmup before the resident is
    /// ready), so it cannot confirm steady-state direct present; this line does.
    first_dmabuf_logged: bool,
    /// Latched direct-present degradation reasons. These are always-on fail-log
    /// lines, not per-frame stderr, because a broken import with an intentionally
    /// elided CPU buffer otherwise looks like a frozen/blank window with no
    /// reason in `/tmp/reims-vgpu-fail.log`.
    direct_degrade_logged: HashSet<&'static str>,
    mem_props: vk::PhysicalDeviceMemoryProperties,
    /// Support-matrix classification of the WINDOW device. It may be a
    /// different physical device than the engine's (hybrid laptops), so it gets
    /// its own topology and handoff answers rather than borrowing the engine's.
    caps: caps::HostGpuCaps,
    /// Host-visible LINEAR staging image the guest BGRA frame uploads into and
    /// the swapchain image blits from; recreated when the frame geometry changes.
    staging: Option<StagingFrame>,
    /// True once the current staging image has been transitioned out of
    /// `PREINITIALIZED` (its first blit). Reset when staging is recreated.
    staging_general: bool,
    /// `Frame::seq` currently resident in the staging image, or `None` before
    /// the first upload / after a staging recreate. A present whose frame seq
    /// matches this skips the full-frame CPU upload and only re-blits — the
    /// idle/steady-desktop fast path (the swapchain still redraws per vblank).
    staged_seq: Option<u64>,
    /// Skip-ratio proxy (always-on, throttled to one log line per 1024
    /// presents so it never floods). `presents` counts every `queue_present`;
    /// `uploads` counts the full-frame staging copies actually performed. A
    /// presents≫uploads ratio confirms the seq-gated upload elision is live on
    /// a static/steady desktop; a ~1:1 ratio would mean every vblank still
    /// re-uploads (a regression).
    presents: u64,
    uploads: u64,
    /// Present-source census (always-on, folded into the throttled skip-ratio
    /// line). `dmabuf_blits` counts presents served by the zero-copy engine
    /// import (route B); `staging_blits` counts presents served by the CPU
    /// staging path. On an import-capable window a healthy direct-present boot is
    /// dmabuf≫staging in steady state; a `staging_blits` that keeps climbing
    /// after warmup means the export silently reverted to CPU (e.g. the member-
    /// resident fallback regressed) — a regression this line NAMES without a
    /// human watching the window.
    dmabuf_blits: u64,
    staging_blits: u64,
    /// Direct-present import churn proxy. `fresh_imports` counts the one-time
    /// Vulkan imports that populate the ring cache; `redundant_fds` counts fresh
    /// fds the producer still supplied for ring slots the window had already
    /// imported. A healthy route-B window can blit from the cached slots, so a
    /// growing `redundant_fds` value names the remaining fd-dup/close overhead.
    fresh_imports: u64,
    redundant_fds: u64,
    /// Silent-revert guard. Once direct present is established (`dmabuf_blits`
    /// has been nonzero), a sustained run of CPU-staging presents means the
    /// engine export regressed (e.g. the member-resident fallback broke). Count
    /// CONSECUTIVE staging presents after establishment; a run past
    /// `REVERT_ALARM_RUN` emits ONE always-on fail line — a single transient
    /// staging frame across a geometry change never trips it. `revert_logged`
    /// latches the one-shot.
    staging_run: u32,
    revert_logged: bool,
    /// `VK_KHR_external_memory_fd` loader when the window device enables the
    /// dmabuf-import extensions (route B direct-present) — `None` on a device
    /// that lacks them, in which case the window stays on the CPU staging path.
    import_fd_loader: Option<ash::khr::external_memory_fd::Device>,
    /// One imported dmabuf per engine ring slot (direct-present route B). The
    /// engine hands a different ring slot each present and re-writes each slot's
    /// shared memory in place, so the window imports each slot's fd ONCE and
    /// re-blits the cached import — a steady-state present neither uploads nor
    /// re-imports, just scale-blits. Sized [`SCANOUT_EXPORT_RING`]; all `None`
    /// until the first dmabuf frame.
    import_ring: Vec<Option<ImportedDmabufImage>>,
    /// Geometry the `import_ring` entries are valid for; on a geometry change
    /// every import is dropped and re-imported at the new size.
    import_geom: (u32, u32),
    /// The ring slot to blit this present (and subsequent same-`seq` presents),
    /// or `None` when the CPU staging path is active. Set when a new dmabuf frame
    /// is prepared; `staged_seq` gates re-preparing.
    active_import: Option<usize>,
    /// OPTIMAL-tiled device-local scratch the imported LINEAR dmabuf is copied
    /// into before the scaling blit to the swapchain (route B). A scaling
    /// `vkCmdBlitImage` with `VK_FILTER_LINEAR` requires the source to support
    /// `SAMPLED_IMAGE_FILTER_LINEAR` (and `BLIT_SRC`) for its tiling — LINEAR
    /// tiling guarantees neither, so a LINEAR imported dmabuf is not a portable
    /// scaling-blit source (the 4K→window downscale resamples; the ≈1:1 1080p
    /// case does not). A raw `vkCmdCopyImage` LINEAR→OPTIMAL preserves the exact
    /// bytes and the scaling blit then reads a well-defined OPTIMAL source. Sized
    /// to the current import geometry; recreated on a geometry change.
    ///
    /// NOTE: this scratch does NOT affect channel order — a whole-desktop R/B
    /// swap once attributed to a "LINEAR-source scaling blit" was root-caused to
    /// the compute BGRA storage-image composite instead (fixed there via a
    /// format-less B8G8R8A8_UNORM storage view); window-side blit changes were
    /// proven byte-identical. Kept purely for the scaling-blit portability above.
    blit_scratch: Option<ScratchImage>,
}

/// An OPTIMAL-tiled device-local BGRA8 image the imported LINEAR dmabuf is copied
/// into (raw, same size) so the scaling swapchain blit reads a portable source.
struct ScratchImage {
    image: vk::Image,
    memory: vk::DeviceMemory,
    width: u32,
    height: u32,
    /// `false` until the first copy transitions it out of `UNDEFINED`.
    initialized: bool,
}

impl ScratchImage {
    unsafe fn destroy(self, device: &ash::Device) {
        device.destroy_image(self.image, None);
        device.free_memory(self.memory, None);
    }
}

/// A host-visible LINEAR BGRA8 image the guest frame is uploaded into (mapped,
/// coherent) and blitted from into the swapchain image. Kept in `GENERAL` layout
/// so both host writes and the transfer read are valid without a layout change
/// per frame.
struct StagingFrame {
    image: vk::Image,
    memory: vk::DeviceMemory,
    /// Persistent HOST_COHERENT mapping of `memory` (write the frame bytes here).
    mapped: *mut u8,
    width: u32,
    height: u32,
    /// Bytes per row of the LINEAR image (from the subresource layout — may
    /// exceed `width*4` due to driver alignment).
    row_pitch: u64,
    /// Byte offset of the (single) subresource within `memory`.
    offset: u64,
}

impl StagingFrame {
    /// Unmap + destroy. Caller ensures no in-flight command references it.
    unsafe fn destroy(self, device: &ash::Device) {
        device.unmap_memory(self.memory);
        device.destroy_image(self.image, None);
        device.free_memory(self.memory, None);
    }
}

impl VkState {
    unsafe fn new(window: &Window) -> Result<Self, WindowError> {
        let entry = ash::Entry::load().map_err(|e| WindowError::VkLoadLoader(e.to_string()))?;
        let display = window
            .display_handle()
            .map_err(|e| WindowError::VkDisplayHandle(e.to_string()))?
            .as_raw();
        let win = window
            .window_handle()
            .map_err(|e| WindowError::VkWindowHandle(e.to_string()))?
            .as_raw();

        let surface_exts = ash_window::enumerate_required_extensions(display)
            .map_err(|e| WindowError::VkRequiredExts(e.to_string()))?;
        let portability_enumeration = entry
            .enumerate_instance_extension_properties(None)
            .map_err(|e| WindowError::VkEnumerateInstanceExts(e.to_string()))?
            .iter()
            .any(|extension| {
                std::ffi::CStr::from_ptr(extension.extension_name.as_ptr())
                    == vk::KHR_PORTABILITY_ENUMERATION_NAME
            });
        let mut instance_exts = surface_exts.to_vec();
        if portability_enumeration {
            instance_exts.push(vk::KHR_PORTABILITY_ENUMERATION_NAME.as_ptr());
        }
        // Same negotiation as the engine device: ask for what the loader can
        // give, capped at the highest version we know how to use. Hardcoding
        // 1.3 is VK_ERROR_INCOMPATIBLE_DRIVER on a Vulkan 1.0 loader.
        let loader_version = entry
            .try_enumerate_instance_version()
            .ok()
            .flatten()
            .unwrap_or(vk::API_VERSION_1_0);
        let app = vk::ApplicationInfo::default()
            .api_version(caps::api_floor::instance_api_version(loader_version));
        let mut ici = vk::InstanceCreateInfo::default()
            .application_info(&app)
            .enabled_extension_names(&instance_exts);
        if portability_enumeration {
            ici = ici.flags(vk::InstanceCreateFlags::ENUMERATE_PORTABILITY_KHR);
        }
        let instance = entry
            .create_instance(&ici, None)
            .map_err(|e| WindowError::VkCreateInstance(e.to_string()))?;

        let surface = ash_window::create_surface(&entry, &instance, display, win, None)
            .map_err(|e| WindowError::VkCreateSurface(e.to_string()))?;
        let surface_loader = ash::khr::surface::Instance::new(&entry, &instance);

        // Pick a device + queue family that can present to this surface, preferring
        // a discrete GPU (falls back to any device — iGPU hosts).
        let pds = instance
            .enumerate_physical_devices()
            .map_err(|e| WindowError::VkEnumeratePhysicalDevices(e.to_string()))?;
        // Candidates are (api_version, device_type, (pd, queue_family)) for
        // every present-capable graphics family. Ranking and the Vulkan 1.2
        // floor come from the SHARED selector the engine device uses, so the
        // two devices can no longer disagree about which GPU is best — the old
        // local 3/2/1/0 scale did not demote a CPU software rasterizer below an
        // unclassified device, and applied no API floor at all.
        let mut candidates: Vec<(u32, vk::PhysicalDeviceType, (vk::PhysicalDevice, u32))> =
            Vec::new();
        for pd in pds {
            let props = instance.get_physical_device_properties(pd);
            let qfs = instance.get_physical_device_queue_family_properties(pd);
            for (i, qf) in qfs.iter().enumerate() {
                if !qf.queue_flags.contains(vk::QueueFlags::GRAPHICS) {
                    continue;
                }
                let present = surface_loader
                    .get_physical_device_surface_support(pd, i as u32, surface)
                    .unwrap_or(false);
                if !present {
                    continue;
                }
                candidates.push((props.api_version, props.device_type, (pd, i as u32)));
            }
        }
        let ((pd, qfamily), _chosen_api_version) =
            caps::device_select::select_physical_device(&candidates).map_err(|below| {
                let msg = if below.is_empty() {
                    "no present-capable device".to_string()
                } else {
                    format!(
                        "no present-capable device meets the Vulkan {} floor",
                        caps::api_floor::version_str(caps::api_floor::MIN_SUPPORTED_API),
                    )
                };
                // Emitted by the caller through `Emit::decline("host_window_init",
                // …)` when this reaches `resumed`, so the reason is the registered
                // `window_vk_no_usable_device` rather than a bare free-text line.
                WindowError::VkNoUsableDevice(msg)
            })?;

        let prio = [1.0f32];
        let qci = [vk::DeviceQueueCreateInfo::default()
            .queue_family_index(qfamily)
            .queue_priorities(&prio)];
        // Enable the dmabuf-import extensions (route B direct-present) only when
        // the device advertises all three — on a host that lacks them the window
        // stays on the CPU staging path (import_fd_loader stays None). Same
        // portability discipline as the engine device (AGENTS.md).
        let want_import = [
            vk::KHR_EXTERNAL_MEMORY_NAME,
            vk::KHR_EXTERNAL_MEMORY_FD_NAME,
            vk::EXT_EXTERNAL_MEMORY_DMA_BUF_NAME,
        ];
        let device_extensions = instance
            .enumerate_device_extension_properties(pd)
            .map_err(|e| WindowError::VkEnumerateDeviceExts(e.to_string()))?;
        let has_device_extension = |want: &std::ffi::CStr| {
            device_extensions
                .iter()
                .any(|property| std::ffi::CStr::from_ptr(property.extension_name.as_ptr()) == want)
        };
        let have_import = want_import.iter().all(|want| {
            device_extensions
                .iter()
                .any(|property| std::ffi::CStr::from_ptr(property.extension_name.as_ptr()) == *want)
        });
        // VK_KHR_swapchain was pushed unconditionally, so a device that does
        // not advertise it failed at create_device with a bare
        // ERROR_EXTENSION_NOT_PRESENT and no reason. A window without a
        // swapchain cannot work, so decline here by name instead — the caller
        // emits `window_vk_no_swapchain_extension` when this reaches `resumed`.
        if !has_device_extension(ash::khr::swapchain::NAME) {
            return Err(WindowError::VkNoSwapchainExtension);
        }
        let mut dev_exts = vec![ash::khr::swapchain::NAME.as_ptr()];
        if has_device_extension(vk::KHR_PORTABILITY_SUBSET_NAME) {
            dev_exts.push(vk::KHR_PORTABILITY_SUBSET_NAME.as_ptr());
        }
        if have_import {
            dev_exts.extend(want_import.iter().map(|n| n.as_ptr()));
        }
        let dci = vk::DeviceCreateInfo::default()
            .queue_create_infos(&qci)
            .enabled_extension_names(&dev_exts);
        let device = instance
            .create_device(pd, &dci, None)
            .map_err(|e| WindowError::VkCreateDevice(e.to_string()))?;
        let queue = device.get_device_queue(qfamily, 0);
        let swapchain_loader = ash::khr::swapchain::Device::new(&instance, &device);
        let import_fd_loader =
            have_import.then(|| ash::khr::external_memory_fd::Device::new(&instance, &device));

        let cmd_pool = device
            .create_command_pool(
                &vk::CommandPoolCreateInfo::default()
                    .queue_family_index(qfamily)
                    .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER),
                None,
            )
            .map_err(|e| WindowError::VkCommandPool(e.to_string()))?;
        let cmd = device
            .allocate_command_buffers(
                &vk::CommandBufferAllocateInfo::default()
                    .command_pool(cmd_pool)
                    .level(vk::CommandBufferLevel::PRIMARY)
                    .command_buffer_count(1),
            )
            .map_err(|e| WindowError::VkAllocCmd(e.to_string()))?[0];
        let sem = |device: &ash::Device| {
            device.create_semaphore(&vk::SemaphoreCreateInfo::default(), None)
        };
        let image_available =
            sem(&device).map_err(|e| WindowError::VkSemaphoreImageAvailable(e.to_string()))?;
        let render_finished =
            sem(&device).map_err(|e| WindowError::VkSemaphoreRenderFinished(e.to_string()))?;
        let in_flight = device
            .create_fence(
                &vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED),
                None,
            )
            .map_err(|e| WindowError::VkFence(e.to_string()))?;

        let mem_props = instance.get_physical_device_memory_properties(pd);
        let device_props = instance.get_physical_device_properties(pd);
        let window_caps = caps::HostGpuCaps {
            memory: caps::memory_topology::classify_memory(&mem_props),
            // The window's zero-copy story is the import side of the display
            // rail: it never exports a dmabuf and never imports guest pages
            // (both memory rails belong to the engine device), so the only
            // mechanism that means anything here is dmabuf import.
            zero_copy: caps::ZeroCopyProfile::display_only(have_import),
            // Always a swapchain — reaching here proved VK_KHR_swapchain is
            // advertised. Never an export: this device consumes frames.
            handoff: caps::HandoffLadder::resolve(&caps::frame_interop::HandoffInputs {
                dmabuf_export: false,
                engine_swapchain: true,
            }),
            quirks: caps::DriverQuirk::for_portability_subset(has_device_extension(
                vk::KHR_PORTABILITY_SUBSET_NAME,
            )),
            portability_subset: has_device_extension(vk::KHR_PORTABILITY_SUBSET_NAME),
            device_api_version: device_props.api_version,
            device_type: device_props.device_type,
        };
        crate::observe::off(format!(
            "host_window_{}",
            window_caps.consumer_line(
                &std::ffi::CStr::from_ptr(device_props.device_name.as_ptr()).to_string_lossy()
            )
        ));
        let mut s = VkState {
            _entry: entry,
            instance,
            surface_loader,
            surface,
            pd,
            device,
            queue,
            qfamily,
            swapchain_loader,
            swapchain: vk::SwapchainKHR::null(),
            images: Vec::new(),
            format: crate::backend::vulkan::translate::pixel::SCANOUT_FORMAT,
            extent: vk::Extent2D {
                width: 0,
                height: 0,
            },
            cmd_pool,
            cmd,
            image_available,
            render_finished,
            in_flight,
            first_present_logged: false,
            first_guest_frame_logged: false,
            first_dmabuf_logged: false,
            direct_degrade_logged: HashSet::new(),
            mem_props,
            caps: window_caps,
            staging: None,
            staging_general: false,
            staged_seq: None,
            presents: 0,
            uploads: 0,
            dmabuf_blits: 0,
            staging_blits: 0,
            fresh_imports: 0,
            redundant_fds: 0,
            staging_run: 0,
            revert_logged: false,
            import_fd_loader,
            import_ring: (0..SCANOUT_EXPORT_RING).map(|_| None).collect(),
            import_geom: (0, 0),
            active_import: None,
            blit_scratch: None,
        };
        let size = window.inner_size();
        s.recreate_swapchain(size.width.max(1), size.height.max(1));
        Ok(s)
    }

    /// (Re)build the swapchain at `width`x`height`. Idempotent on failure: leaves
    /// `swapchain` null so `present` skips until the next resize.
    unsafe fn recreate_swapchain(&mut self, width: u32, height: u32) {
        let _ = self.device.device_wait_idle();
        // A failure here leaves `swapchain` null, so `present()` returns Ok
        // and draws nothing — a permanently blank window. It must be
        // fail-visible, not an eprintln nobody reads. (Local renamed from
        // `caps` so it no longer shadows the capability module.)
        let surface_caps = match self
            .surface_loader
            .get_physical_device_surface_capabilities(self.pd, self.surface)
        {
            Ok(c) => c,
            Err(error) => {
                self.log_direct_present_degrade(
                    "swapchain_surface_caps",
                    format!("{width}x{height} err={error}"),
                );
                return;
            }
        };
        let formats = self
            .surface_loader
            .get_physical_device_surface_formats(self.pd, self.surface)
            .unwrap_or_default();
        // Prefer BGRA8 UNORM sRGB-nonlinear; else take the first offered.
        let sfmt = formats
            .iter()
            .find(|f| {
                f.format == crate::backend::vulkan::translate::pixel::SCANOUT_FORMAT
                    && f.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR
            })
            .or_else(|| formats.first())
            .copied()
            .unwrap_or(vk::SurfaceFormatKHR {
                format: crate::backend::vulkan::translate::pixel::SCANOUT_FORMAT,
                color_space: vk::ColorSpaceKHR::SRGB_NONLINEAR,
            });
        let extent = if surface_caps.current_extent.width != u32::MAX {
            surface_caps.current_extent
        } else {
            vk::Extent2D {
                width: width.clamp(
                    surface_caps.min_image_extent.width,
                    surface_caps.max_image_extent.width,
                ),
                height: height.clamp(
                    surface_caps.min_image_extent.height,
                    surface_caps.max_image_extent.height,
                ),
            }
        };
        if extent.width == 0 || extent.height == 0 {
            return;
        }
        let mut min_images = surface_caps.min_image_count + 1;
        if surface_caps.max_image_count > 0 {
            min_images = min_images.min(surface_caps.max_image_count);
        }
        let old = self.swapchain;
        let ci = vk::SwapchainCreateInfoKHR::default()
            .surface(self.surface)
            .min_image_count(min_images)
            .image_format(sfmt.format)
            .image_color_space(sfmt.color_space)
            .image_extent(extent)
            .image_array_layers(1)
            .image_usage(vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::COLOR_ATTACHMENT)
            .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
            .pre_transform(surface_caps.current_transform)
            .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
            .present_mode(vk::PresentModeKHR::FIFO)
            .clipped(true)
            .old_swapchain(old);
        let swapchain = match self.swapchain_loader.create_swapchain(&ci, None) {
            Ok(sc) => sc,
            Err(e) => {
                eprintln!("reims-vgpu-window: create_swapchain: {e}");
                return;
            }
        };
        if old != vk::SwapchainKHR::null() {
            self.swapchain_loader.destroy_swapchain(old, None);
        }
        self.swapchain = swapchain;
        self.images = self
            .swapchain_loader
            .get_swapchain_images(swapchain)
            .unwrap_or_default();
        self.format = sfmt.format;
        self.extent = extent;
    }

    /// Acquire → clear (or blit the frame) → present one image. Recreates the
    /// swapchain on OUT_OF_DATE/SUBOPTIMAL.
    unsafe fn present(&mut self, frame: Option<&Frame>) -> Result<(), WindowError> {
        if self.swapchain == vk::SwapchainKHR::null() {
            return Ok(());
        }
        let fences = [self.in_flight];
        let _ = self.device.wait_for_fences(&fences, true, u64::MAX);

        // Prepare this frame's blit source (dmabuf import or CPU staging upload),
        // seq-gated so an unchanged frame is neither re-uploaded nor re-imported.
        let src = self.prepare_frame(frame);

        let acquire = self.swapchain_loader.acquire_next_image(
            self.swapchain,
            u64::MAX,
            self.image_available,
            vk::Fence::null(),
        );
        let index = match acquire {
            Ok((i, _suboptimal)) => i,
            Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                self.recreate_swapchain(self.extent.width, self.extent.height);
                return Ok(());
            }
            Err(e) => return Err(WindowError::PresentAcquire(e)),
        };
        self.device
            .reset_fences(&fences)
            .map_err(WindowError::PresentResetFence)?;
        let image = self.images[index as usize];

        self.device
            .reset_command_buffer(self.cmd, vk::CommandBufferResetFlags::empty())
            .map_err(WindowError::PresentResetCommandBuffer)?;
        self.device
            .begin_command_buffer(
                self.cmd,
                &vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )
            .map_err(WindowError::PresentBeginCommandBuffer)?;

        let sub = vk::ImageSubresourceRange::default()
            .aspect_mask(vk::ImageAspectFlags::COLOR)
            .level_count(1)
            .layer_count(1);
        self.barrier(
            image,
            sub,
            vk::ImageLayout::UNDEFINED,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            vk::AccessFlags::empty(),
            vk::AccessFlags::TRANSFER_WRITE,
            vk::PipelineStageFlags::TOP_OF_PIPE,
            vk::PipelineStageFlags::TRANSFER,
        );
        match src {
            // Scale-blit the guest frame into the swapchain image (LINEAR filter
            // covers guest-res != window-res).
            BlitSource::Imported(idx) => {
                self.dmabuf_blits = self.dmabuf_blits.wrapping_add(1);
                self.staging_run = 0;
                self.blit_ring_slot(image, idx)
            }
            BlitSource::Staging { .. } => {
                self.staging_blits = self.staging_blits.wrapping_add(1);
                // Only meaningful when a dmabuf-carried frame fell through to
                // staging after direct present was established. Plain CPU frames
                // with no dmabuf source are counted in the source census, but
                // are not a window import regression.
                self.staging_run = next_direct_present_revert_run(
                    self.staging_run,
                    self.dmabuf_blits,
                    self.revert_logged,
                    src,
                );
                if self.staging_run >= REVERT_ALARM_RUN {
                    self.revert_logged = true;
                    crate::observe::off(format!(
                        "direct_present_reverted_to_staging run={} dmabuf_blits={} \
                             staging_blits={} (engine export regressed — window fell back to CPU)",
                        self.staging_run, self.dmabuf_blits, self.staging_blits
                    ));
                }
                self.blit_staging(image)
            }
            BlitSource::Slate => {
                // No frame yet: clear to a dim slate so the window is visibly alive.
                let clear = vk::ClearColorValue {
                    float32: [0.05, 0.06, 0.08, 1.0],
                };
                self.device.cmd_clear_color_image(
                    self.cmd,
                    image,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    &clear,
                    &[sub],
                );
            }
        }
        if !matches!(src, BlitSource::Slate) && !self.first_guest_frame_logged {
            if let Some(f) = frame {
                let via = if matches!(src, BlitSource::Imported(_)) {
                    "dmabuf"
                } else if matches!(src, BlitSource::Staging { .. }) {
                    "staging"
                } else {
                    "slate"
                };
                eprintln!(
                    "reims-vgpu-window: first guest frame presented ({}x{}, via {via})",
                    f.width, f.height
                );
            }
            self.first_guest_frame_logged = true;
        }
        if matches!(src, BlitSource::Imported(_)) && !self.first_dmabuf_logged {
            eprintln!(
                "reims-vgpu-window: direct present live — first frame blitted via dmabuf (route B)"
            );
            self.first_dmabuf_logged = true;
        }
        self.barrier(
            image,
            sub,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            vk::ImageLayout::PRESENT_SRC_KHR,
            vk::AccessFlags::TRANSFER_WRITE,
            vk::AccessFlags::empty(),
            vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::BOTTOM_OF_PIPE,
        );
        self.device
            .end_command_buffer(self.cmd)
            .map_err(WindowError::PresentEndCommandBuffer)?;

        let wait = [self.image_available];
        let wait_stage = [vk::PipelineStageFlags::TRANSFER];
        let signal = [self.render_finished];
        let cmds = [self.cmd];
        let submit = vk::SubmitInfo::default()
            .wait_semaphores(&wait)
            .wait_dst_stage_mask(&wait_stage)
            .command_buffers(&cmds)
            .signal_semaphores(&signal);
        self.device
            .queue_submit(self.queue, &[submit], self.in_flight)
            .map_err(WindowError::PresentQueueSubmit)?;

        let swapchains = [self.swapchain];
        let indices = [index];
        let present = vk::PresentInfoKHR::default()
            .wait_semaphores(&signal)
            .swapchains(&swapchains)
            .image_indices(&indices);
        match self.swapchain_loader.queue_present(self.queue, &present) {
            Ok(_) => {
                if !self.first_present_logged {
                    eprintln!(
                        "reims-vgpu-window: first frame presented ({}x{}, {} swapchain images)",
                        self.extent.width,
                        self.extent.height,
                        self.images.len()
                    );
                    self.first_present_logged = true;
                }
                self.presents = self.presents.wrapping_add(1);
                // Throttled skip-ratio proxy: one line per 1024 presents (~8 s
                // at 120 Hz), never per-frame — confirms the upload elision is
                // live without flooding.
                if self.presents.is_multiple_of(1024) {
                    eprintln!(
                        "reims-vgpu-window: present skip-ratio uploads={} presents={} \
                         (elided {} redundant full-frame uploads) source dmabuf={} staging={} \
                         fresh_imports={} redundant_fds={}",
                        self.uploads,
                        self.presents,
                        self.presents.saturating_sub(self.uploads),
                        self.dmabuf_blits,
                        self.staging_blits,
                        self.fresh_imports,
                        self.redundant_fds,
                    );
                    crate::observe::off(direct_present_source_line(
                        self.presents,
                        self.uploads,
                        self.dmabuf_blits,
                        self.staging_blits,
                        self.fresh_imports,
                        self.redundant_fds,
                    ));
                }
                Ok(())
            }
            Err(vk::Result::ERROR_OUT_OF_DATE_KHR) | Err(vk::Result::SUBOPTIMAL_KHR) => {
                self.recreate_swapchain(self.extent.width, self.extent.height);
                Ok(())
            }
            Err(e) => Err(WindowError::PresentQueue(e)),
        }
    }

    fn log_direct_present_degrade(&mut self, reason: &'static str, detail: String) {
        if self.direct_degrade_logged.insert(reason) {
            crate::observe::off(direct_present_degrade_line(reason, detail));
        }
    }

    fn log_direct_present_decline(
        &mut self,
        class: &'static str,
        decline: &WindowError,
        frame: &Frame,
        ring_idx: Option<usize>,
    ) {
        use crate::observe::Decline as _;
        if self.direct_degrade_logged.insert(decline.slug()) {
            let mut emit = crate::observe::Emit::decline("direct_present_degrade", decline)
                .field("class", class)
                .field("seq", frame.seq)
                .field("width", frame.width)
                .field("height", frame.height);
            if let Some(ring_idx) = ring_idx {
                emit = emit.field("ring", ring_idx);
            }
            emit.off();
        }
    }

    /// Pick a memory type for `class` on the window device, through the same
    /// topology-aware policy the engine uses. The local flag-matching helper
    /// this replaces had no notion of preferences at all, so window staging
    /// always took the first host-visible type even when a device-local one
    /// (unified host) or a cached one was available.
    fn memory_type_for(&self, type_bits: u32, class: caps::MemoryClass) -> Option<u32> {
        caps::memory_topology::select_memory_type(
            &self.mem_props,
            type_bits,
            &self.caps.memory_request(class),
        )
    }

    /// Ensure the staging image matches `width`x`height`, (re)creating it — a
    /// host-visible coherent LINEAR BGRA8 image, persistently mapped, in
    /// `PREINITIALIZED` layout (host writes are valid immediately).
    unsafe fn ensure_staging(&mut self, width: u32, height: u32) -> Result<(), WindowError> {
        if let Some(s) = &self.staging {
            if s.width == width && s.height == height {
                return Ok(());
            }
        }
        // Geometry changed (rare): the prior staging is only referenced by the
        // just-completed frame (we waited on in_flight above), so it is safe to
        // drop now.
        if let Some(old) = self.staging.take() {
            old.destroy(&self.device);
        }
        self.staging_general = false;
        // New staging image holds no frame yet — force the next present to
        // upload regardless of its seq.
        self.staged_seq = None;
        let image = self
            .device
            .create_image(
                &vk::ImageCreateInfo::default()
                    .image_type(vk::ImageType::TYPE_2D)
                    .format(crate::backend::vulkan::translate::pixel::SCANOUT_FORMAT)
                    .extent(vk::Extent3D {
                        width,
                        height,
                        depth: 1,
                    })
                    .mip_levels(1)
                    .array_layers(1)
                    .samples(vk::SampleCountFlags::TYPE_1)
                    .tiling(vk::ImageTiling::LINEAR)
                    .usage(vk::ImageUsageFlags::TRANSFER_SRC)
                    .initial_layout(vk::ImageLayout::PREINITIALIZED),
                None,
            )
            .map_err(WindowError::StagingCreateImage)?;
        let req = self.device.get_image_memory_requirements(image);
        let mem_type = self
            .memory_type_for(req.memory_type_bits, caps::MemoryClass::Upload)
            .ok_or(WindowError::StagingMemoryTypeUnavailable {
                type_bits: req.memory_type_bits,
            })?;
        let memory = self
            .device
            .allocate_memory(
                &vk::MemoryAllocateInfo::default()
                    .allocation_size(req.size)
                    .memory_type_index(mem_type),
                None,
            )
            .map_err(|e| {
                self.device.destroy_image(image, None);
                WindowError::StagingAllocateMemory {
                    bytes: req.size,
                    result: e,
                }
            })?;
        self.device
            .bind_image_memory(image, memory, 0)
            .map_err(WindowError::StagingBindMemory)?;
        let layout = self.device.get_image_subresource_layout(
            image,
            vk::ImageSubresource::default().aspect_mask(vk::ImageAspectFlags::COLOR),
        );
        let mapped = self
            .device
            .map_memory(memory, 0, req.size, vk::MemoryMapFlags::empty())
            .map_err(|result| WindowError::StagingMapMemory {
                bytes: req.size,
                result,
            })? as *mut u8;
        self.staging = Some(StagingFrame {
            image,
            memory,
            mapped,
            width,
            height,
            row_pitch: layout.row_pitch,
            offset: layout.offset,
        });
        Ok(())
    }

    /// Copy a tightly-packed BGRA frame into the staging image honoring its
    /// row pitch. Coherent memory: no explicit flush.
    unsafe fn upload_frame(&mut self, f: &Frame) {
        let Some(s) = self.staging.as_ref() else {
            // No staging image: the frame is silently discarded and the
            // swapchain shows whatever it held before.
            self.log_direct_present_degrade(
                "upload_no_staging",
                format!("{}x{}", f.width, f.height),
            );
            return;
        };
        let src_row = f.width as usize * 4;
        for y in 0..f.height as usize {
            let dst = s.mapped.add(s.offset as usize + y * s.row_pitch as usize);
            let src = f.bgra.as_ptr().add(y * src_row);
            std::ptr::copy_nonoverlapping(src, dst, src_row);
        }
    }

    /// Record: make host writes visible, then scale-blit the staging image into
    /// `dst` (already in `TRANSFER_DST_OPTIMAL`).
    unsafe fn blit_staging(&mut self, dst: vk::Image) {
        let (src_image, sw, sh) = match self.staging.as_ref() {
            Some(s) => (s.image, s.width, s.height),
            None => {
                // Returning here presents a swapchain image that was only
                // transitioned, never written — undefined contents on screen.
                self.log_direct_present_degrade("blit_no_staging", String::new());
                return;
            }
        };
        // Staging stays in GENERAL (valid host write + blit src); the barrier
        // publishes this frame's host write to the transfer read. First use
        // transitions from PREINITIALIZED (preserving the uploaded bytes).
        let sub = vk::ImageSubresourceRange::default()
            .aspect_mask(vk::ImageAspectFlags::COLOR)
            .level_count(1)
            .layer_count(1);
        let old = if self.staging_general {
            vk::ImageLayout::GENERAL
        } else {
            vk::ImageLayout::PREINITIALIZED
        };
        let b = vk::ImageMemoryBarrier::default()
            .old_layout(old)
            .new_layout(vk::ImageLayout::GENERAL)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .image(src_image)
            .subresource_range(sub)
            .src_access_mask(vk::AccessFlags::HOST_WRITE)
            .dst_access_mask(vk::AccessFlags::TRANSFER_READ);
        self.device.cmd_pipeline_barrier(
            self.cmd,
            vk::PipelineStageFlags::HOST,
            vk::PipelineStageFlags::TRANSFER,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &[b],
        );
        self.staging_general = true;
        let layers = vk::ImageSubresourceLayers::default()
            .aspect_mask(vk::ImageAspectFlags::COLOR)
            .layer_count(1);
        let region = vk::ImageBlit::default()
            .src_subresource(layers)
            .src_offsets([
                vk::Offset3D { x: 0, y: 0, z: 0 },
                vk::Offset3D {
                    x: sw as i32,
                    y: sh as i32,
                    z: 1,
                },
            ])
            .dst_subresource(layers)
            .dst_offsets([
                vk::Offset3D { x: 0, y: 0, z: 0 },
                vk::Offset3D {
                    x: self.extent.width as i32,
                    y: self.extent.height as i32,
                    z: 1,
                },
            ]);
        self.device.cmd_blit_image(
            self.cmd,
            src_image,
            vk::ImageLayout::GENERAL,
            dst,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            &[region],
            crate::backend::vulkan::translate::sampler::PRESENT_BLIT_FILTER,
        );
    }

    /// Prepare the blit source for `frame`, seq-gated so an unchanged frame is
    /// neither re-uploaded (CPU path) nor re-imported (dmabuf path). Prefers the
    /// zero-copy dmabuf when the frame carries one and the window can import it,
    /// falling back to the CPU staging upload otherwise — so a dmabuf miss (no
    /// exts, import failure, absent handle) degrades to today's path, never to a
    /// blank window.
    unsafe fn prepare_frame(&mut self, frame: Option<&Frame>) -> BlitSource {
        let Some(f) = frame else {
            return BlitSource::Slate;
        };
        if f.width == 0 || f.height == 0 {
            return BlitSource::Slate;
        }

        // NOTE: the CPU buffer's length is deliberately NOT checked here. It is a
        // precondition of the STAGING path only, and is validated at that path
        // below. A dmabuf-carried frame legitimately arrives with an EMPTY `bgra`
        // — the device stopped copying the full frame into it once the proxies
        // moved to a GPU reduction — and gating the whole function on its length
        // would reject a perfectly good dmabuf frame and blank the window.

        // Zero-copy dmabuf path.
        let mut dmabuf_fallback = false;
        if self.import_fd_loader.is_some() {
            if let Some(dm) = f.dmabuf.as_ref() {
                // Same frame already prepared as an import: re-blit the cached
                // ring slot (the engine rewrote its shared memory in place).
                if self.staged_seq == Some(f.seq) && self.active_import == Some(dm.ring_idx) {
                    return BlitSource::Imported(dm.ring_idx);
                }
                // New frame: consume the fd once (import transfers ownership to
                // Vulkan; a redundant dup for an already-cached slot is closed).
                let fd = dm.fd.lock().ok().and_then(|mut g| g.take());
                if let Some(fd) = fd {
                    match self.import_ring_slot(fd, f.width, f.height, dm.ring_idx) {
                        Ok(()) => {
                            dm.import_ack.mark_imported(f.width, f.height, dm.ring_idx);
                            self.staged_seq = Some(f.seq);
                            self.active_import = Some(dm.ring_idx);
                            return BlitSource::Imported(dm.ring_idx);
                        }
                        Err(e) => {
                            self.log_direct_present_decline(
                                "import_failed",
                                &e,
                                f,
                                Some(dm.ring_idx),
                            );
                        }
                    }
                    dmabuf_fallback = true;
                } else if self.cached_import_available(f.width, f.height, dm.ring_idx) {
                    self.staged_seq = Some(f.seq);
                    self.active_import = Some(dm.ring_idx);
                    return BlitSource::Imported(dm.ring_idx);
                } else {
                    self.log_direct_present_degrade(
                        "fd_missing",
                        format!(
                            "seq={} ring={} {}x{} active_import={:?}",
                            f.seq, dm.ring_idx, f.width, f.height, self.active_import
                        ),
                    );
                    dmabuf_fallback = true;
                }
                // fd already consumed or import failed → fall through to CPU.
            }
        }

        // CPU staging path (also the fallback when the dmabuf path did not take).
        // This is the one consumer of `bgra`, so its length is required HERE.
        // A short/empty buffer with no usable dmabuf means we genuinely have
        // nothing to show: hold a slate rather than blit uninitialised memory.
        self.active_import = None;
        if f.bgra.len() < (f.width as usize * f.height as usize * 4) {
            self.log_direct_present_degrade(
                "no_source",
                format!(
                    "seq={} dmabuf={} bgra={} need={} {}x{}",
                    f.seq,
                    f.dmabuf.is_some(),
                    f.bgra.len(),
                    f.width as usize * f.height as usize * 4,
                    f.width,
                    f.height
                ),
            );
            return BlitSource::Slate;
        }
        match self.ensure_staging(f.width, f.height) {
            Ok(()) => {
                if needs_staging_upload(self.staged_seq, f.seq) {
                    self.upload_frame(f);
                    self.staged_seq = Some(f.seq);
                    self.uploads = self.uploads.wrapping_add(1);
                }
                BlitSource::Staging { dmabuf_fallback }
            }
            Err(e) => {
                self.log_direct_present_decline("staging_failed", &e, f, None);
                BlitSource::Slate
            }
        }
    }

    fn cached_import_available(&self, width: u32, height: u32, ring_idx: usize) -> bool {
        self.import_geom == (width, height)
            && self
                .import_ring
                .get(ring_idx)
                .and_then(|slot| slot.as_ref())
                .is_some()
    }

    /// Import the engine's exported dmabuf `fd` (an `OwnedFd`) into ring slot
    /// `ring_idx` as a `TRANSFER_SRC` image (direct-present route B), caching it
    /// so later presents on the same slot re-blit without re-importing (the
    /// engine rewrites the slot's shared memory in place). A geometry change
    /// drops every cached import first; on a slot already imported the incoming
    /// fd is a redundant dup for the same underlying dmabuf and is closed.
    ///
    /// The fd is consumed exactly once here: transferred to Vulkan on a
    /// successful import, else closed (Vulkan does not take it on failure, and a
    /// redundant dup is dropped) — so no fd leaks and none double-closes. The
    /// import barrier + blit are validated cross-device by
    /// `cross_device_dmabuf_import_is_byte_identical`.
    unsafe fn import_ring_slot(
        &mut self,
        fd: std::os::fd::OwnedFd,
        width: u32,
        height: u32,
        ring_idx: usize,
    ) -> Result<(), WindowError> {
        use std::os::fd::{FromRawFd, IntoRawFd};
        let Some(loader) = self.import_fd_loader.clone() else {
            // Cannot import; closing `fd` (dropped at scope end) reclaims it.
            return Err(WindowError::DmabufImportExtensionsMissing);
        };
        if ring_idx >= self.import_ring.len() {
            return Err(WindowError::DmabufRingIndexOutOfRange {
                ring_idx,
                ring_len: self.import_ring.len(),
            });
        }
        // Geometry change: drop every cached import (dmabuf refcounting keeps the
        // engine-side buffers alive until the engine frees them too).
        if self.import_geom != (width, height) {
            let _ = self
                .device
                .wait_for_fences(&[self.in_flight], true, u64::MAX);
            for slot in self.import_ring.iter_mut() {
                if let Some(old) = slot.take() {
                    old.destroy(&self.device);
                }
            }
            self.import_geom = (width, height);
        }
        if self.import_ring[ring_idx].is_some() {
            // Already imported this slot — `fd` is a redundant dup; drop closes it.
            self.redundant_fds = self.redundant_fds.wrapping_add(1);
            return Ok(());
        }
        // Fresh import for this slot; quiesce any in-flight present first.
        let _ = self
            .device
            .wait_for_fences(&[self.in_flight], true, u64::MAX);
        let raw = fd.into_raw_fd();
        match crate::backend::vulkan::engine::dmabuf_export::import_bgra_dmabuf_image(
            &self.device,
            &loader,
            &self.mem_props,
            raw,
            width,
            height,
        ) {
            Ok(img) => {
                self.import_ring[ring_idx] = Some(img);
                self.fresh_imports = self.fresh_imports.wrapping_add(1);
                Ok(())
            }
            Err(e) => {
                // Vulkan did not take the fd on failure — reclaim + close it.
                drop(std::os::fd::OwnedFd::from_raw_fd(raw));
                Err(WindowError::DmabufImport(e))
            }
        }
    }

    /// Scale-blit the imported engine dmabuf at ring slot `idx` into `dst`
    /// (already in `TRANSFER_DST_OPTIMAL`), the zero-copy present path. An
    /// EXTERNAL→graphics acquire barrier from `GENERAL` (the producer's layout)
    /// makes the engine's latest write visible AND preserves the imported content
    /// — never `UNDEFINED` (which discards it). Kept in `GENERAL` across frames so
    /// re-blitting sees each fresh engine write without a round-trip transition.
    /// No-op if the slot is not imported.
    /// Ensure the OPTIMAL blit scratch matches `width`x`height`, (re)creating it on
    /// a geometry change. Returns `false` if creation failed (caller skips the
    /// present; the window holds its last good frame). The scratch is device-local
    /// OPTIMAL BGRA8 with TRANSFER_DST (copy target) + TRANSFER_SRC (blit source).
    unsafe fn ensure_blit_scratch(&mut self, width: u32, height: u32) -> bool {
        if let Some(s) = &self.blit_scratch {
            if s.width == width && s.height == height {
                return true;
            }
        }
        // Geometry change: the old scratch may still be referenced by the last
        // submitted present, so drain the device before destroying it. This runs
        // INSIDE `present()` after `in_flight` was already reset (unsignaled), so a
        // fence wait would deadlock — `device_wait_idle` drains regardless of fence
        // state. Only taken when an old scratch actually exists (rare: resolution
        // switch); the first creation has nothing to drain.
        if let Some(old) = self.blit_scratch.take() {
            let _ = self.device.device_wait_idle();
            old.destroy(&self.device);
        }
        // Every failure below used to be a bare `return false`, which leaves the
        // caller presenting a swapchain image it never wrote — a blank or
        // garbage window with nothing in the fail log. Each one now names its
        // stage; the reasons are latched so a persistent failure logs once.
        let image = match self.device.create_image(
            &vk::ImageCreateInfo::default()
                .image_type(vk::ImageType::TYPE_2D)
                .format(crate::backend::vulkan::translate::pixel::SCANOUT_FORMAT)
                .extent(vk::Extent3D {
                    width,
                    height,
                    depth: 1,
                })
                .mip_levels(1)
                .array_layers(1)
                .samples(vk::SampleCountFlags::TYPE_1)
                .tiling(vk::ImageTiling::OPTIMAL)
                .usage(vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::TRANSFER_SRC)
                .initial_layout(vk::ImageLayout::UNDEFINED),
            None,
        ) {
            Ok(image) => image,
            Err(error) => {
                self.log_direct_present_degrade(
                    "blit_scratch_create",
                    format!("{width}x{height} err={error}"),
                );
                return false;
            }
        };
        let req = self.device.get_image_memory_requirements(image);
        let Some(mt) = self.memory_type_for(
            req.memory_type_bits,
            caps::MemoryClass::DeviceLocalPreferred,
        ) else {
            self.log_direct_present_degrade(
                "blit_scratch_no_memory_type",
                format!("{width}x{height} type_bits={:#x}", req.memory_type_bits),
            );
            self.device.destroy_image(image, None);
            return false;
        };
        let memory = match self.device.allocate_memory(
            &vk::MemoryAllocateInfo::default()
                .allocation_size(req.size)
                .memory_type_index(mt),
            None,
        ) {
            Ok(memory) => memory,
            Err(error) => {
                self.log_direct_present_degrade(
                    "blit_scratch_alloc",
                    format!("{width}x{height} bytes={} err={error}", req.size),
                );
                self.device.destroy_image(image, None);
                return false;
            }
        };
        if let Err(error) = self.device.bind_image_memory(image, memory, 0) {
            self.log_direct_present_degrade(
                "blit_scratch_bind",
                format!("{width}x{height} err={error}"),
            );
            self.device.free_memory(memory, None);
            self.device.destroy_image(image, None);
            return false;
        }
        self.blit_scratch = Some(ScratchImage {
            image,
            memory,
            width,
            height,
            initialized: false,
        });
        true
    }

    unsafe fn blit_ring_slot(&mut self, dst: vk::Image, idx: usize) {
        let (src_image, sw, sh) = match self.import_ring.get(idx).and_then(|s| s.as_ref()) {
            Some(img) => (img.image, img.width, img.height),
            None => {
                // The ring slot was never imported; presenting now would show
                // an unwritten swapchain image.
                self.log_direct_present_degrade(
                    "ring_slot_not_imported",
                    format!("idx={idx} ring={}", self.import_ring.len()),
                );
                return;
            }
        };
        let full = vk::ImageSubresourceRange::default()
            .aspect_mask(vk::ImageAspectFlags::COLOR)
            .level_count(1)
            .layer_count(1);
        // Acquire the imported dmabuf (queue-family EXTERNAL→graphics), GENERAL
        // layout (preserves the engine write; never UNDEFINED which discards it).
        let acquire = vk::ImageMemoryBarrier::default()
            .old_layout(vk::ImageLayout::GENERAL)
            .new_layout(vk::ImageLayout::GENERAL)
            .src_queue_family_index(vk::QUEUE_FAMILY_EXTERNAL)
            .dst_queue_family_index(self.qfamily)
            .image(src_image)
            .subresource_range(full)
            .src_access_mask(vk::AccessFlags::MEMORY_WRITE)
            .dst_access_mask(vk::AccessFlags::TRANSFER_READ);
        self.device.cmd_pipeline_barrier(
            self.cmd,
            vk::PipelineStageFlags::TOP_OF_PIPE,
            vk::PipelineStageFlags::TRANSFER,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &[acquire],
        );
        // Route the LINEAR import through an OPTIMAL scratch: a raw same-size
        // `vkCmdCopyImage` preserves the exact BGRA bytes, then the scaling blit
        // reads a portable OPTIMAL source (blit-scaling a LINEAR source with a
        // linear filter is not portable). See `blit_scratch`.
        if !self.ensure_blit_scratch(sw, sh) {
            // ensure_blit_scratch already named its own failure stage.
            return;
        }
        let scratch = self.blit_scratch.as_ref().expect("ensured above");
        let (scratch_image, was_init) = (scratch.image, scratch.initialized);
        let layers = vk::ImageSubresourceLayers::default()
            .aspect_mask(vk::ImageAspectFlags::COLOR)
            .layer_count(1);
        // Scratch → TRANSFER_DST (UNDEFINED first use, else prior TRANSFER_SRC).
        let (old_layout, src_access) = if was_init {
            (
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                vk::AccessFlags::TRANSFER_READ,
            )
        } else {
            (vk::ImageLayout::UNDEFINED, vk::AccessFlags::empty())
        };
        let to_dst = vk::ImageMemoryBarrier::default()
            .old_layout(old_layout)
            .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .image(scratch_image)
            .subresource_range(full)
            .src_access_mask(src_access)
            .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE);
        self.device.cmd_pipeline_barrier(
            self.cmd,
            vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::TRANSFER,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &[to_dst],
        );
        let copy = vk::ImageCopy::default()
            .src_subresource(layers)
            .dst_subresource(layers)
            .extent(vk::Extent3D {
                width: sw,
                height: sh,
                depth: 1,
            });
        self.device.cmd_copy_image(
            self.cmd,
            src_image,
            vk::ImageLayout::GENERAL,
            scratch_image,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            &[copy],
        );
        // Scratch → TRANSFER_SRC for the scaling blit into the swapchain.
        let to_src = vk::ImageMemoryBarrier::default()
            .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
            .new_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .image(scratch_image)
            .subresource_range(full)
            .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
            .dst_access_mask(vk::AccessFlags::TRANSFER_READ);
        self.device.cmd_pipeline_barrier(
            self.cmd,
            vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::TRANSFER,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &[to_src],
        );
        let region = vk::ImageBlit::default()
            .src_subresource(layers)
            .src_offsets([
                vk::Offset3D { x: 0, y: 0, z: 0 },
                vk::Offset3D {
                    x: sw as i32,
                    y: sh as i32,
                    z: 1,
                },
            ])
            .dst_subresource(layers)
            .dst_offsets([
                vk::Offset3D { x: 0, y: 0, z: 0 },
                vk::Offset3D {
                    x: self.extent.width as i32,
                    y: self.extent.height as i32,
                    z: 1,
                },
            ]);
        self.device.cmd_blit_image(
            self.cmd,
            scratch_image,
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            dst,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            &[region],
            crate::backend::vulkan::translate::sampler::PRESENT_BLIT_FILTER,
        );
        self.blit_scratch
            .as_mut()
            .expect("ensured above")
            .initialized = true;
    }

    #[allow(clippy::too_many_arguments)]
    unsafe fn barrier(
        &self,
        image: vk::Image,
        sub: vk::ImageSubresourceRange,
        old: vk::ImageLayout,
        new: vk::ImageLayout,
        src_access: vk::AccessFlags,
        dst_access: vk::AccessFlags,
        src_stage: vk::PipelineStageFlags,
        dst_stage: vk::PipelineStageFlags,
    ) {
        let b = vk::ImageMemoryBarrier::default()
            .old_layout(old)
            .new_layout(new)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .image(image)
            .subresource_range(sub)
            .src_access_mask(src_access)
            .dst_access_mask(dst_access);
        self.device.cmd_pipeline_barrier(
            self.cmd,
            src_stage,
            dst_stage,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &[b],
        );
    }
}

impl Drop for VkState {
    fn drop(&mut self) {
        unsafe {
            let _ = self.device.device_wait_idle();
            if let Some(staging) = self.staging.take() {
                staging.destroy(&self.device);
            }
            for slot in self.import_ring.iter_mut() {
                if let Some(imported) = slot.take() {
                    imported.destroy(&self.device);
                }
            }
            if let Some(scratch) = self.blit_scratch.take() {
                scratch.destroy(&self.device);
            }
            self.device.destroy_fence(self.in_flight, None);
            self.device.destroy_semaphore(self.image_available, None);
            self.device.destroy_semaphore(self.render_finished, None);
            self.device.destroy_command_pool(self.cmd_pool, None);
            if self.swapchain != vk::SwapchainKHR::null() {
                self.swapchain_loader
                    .destroy_swapchain(self.swapchain, None);
            }
            self.device.destroy_device(None);
            self.surface_loader.destroy_surface(self.surface, None);
            self.instance.destroy_instance(None);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every window bring-up check names itself, its slug is namespaced to the
    /// window rail and distinct, and — the property no crate-wide gate can see —
    /// its `fields()` values are whitespace-free even though `detail` is an
    /// arbitrary driver/winit string. The always-on log is parsed by splitting
    /// on spaces, so a space in a value would corrupt the line.
    #[test]
    fn every_window_bringup_check_names_itself_log_safe() {
        use crate::observe::{Decline as _, Emit};
        let all = [
            WindowError::EventLoopBuild("os error while building".into()),
            WindowError::RunApp("event loop exited".into()),
            WindowError::MainLoopRun("event loop exited".into()),
            WindowError::AlreadyOwned { owner: 3 },
            WindowError::NoRegisteredWindow { id: 4 },
            WindowError::WrongOwner {
                owner: 3,
                requested: 4,
            },
            WindowError::CreateNativeWindow("os error creating window".into()),
            WindowError::AttachDisplayHandle("no display handle".into()),
            WindowError::AttachWindowHandle("no window handle".into()),
            WindowError::AttachEngine("engine attach failed".into()),
            WindowError::VkLoadLoader("libvulkan not found".into()),
            WindowError::VkDisplayHandle("no display handle".into()),
            WindowError::VkWindowHandle("no window handle".into()),
            WindowError::VkRequiredExts("ERROR_EXTENSION_NOT_PRESENT".into()),
            WindowError::VkEnumerateInstanceExts("ERROR_OUT_OF_HOST_MEMORY".into()),
            WindowError::VkCreateInstance("ERROR_INCOMPATIBLE_DRIVER".into()),
            WindowError::VkCreateSurface("ERROR_SURFACE_LOST_KHR".into()),
            WindowError::VkEnumeratePhysicalDevices("ERROR_INITIALIZATION_FAILED".into()),
            WindowError::VkNoUsableDevice(
                "no present-capable device meets the Vulkan 1.2 floor".into(),
            ),
            WindowError::VkEnumerateDeviceExts("ERROR_OUT_OF_HOST_MEMORY".into()),
            WindowError::VkNoSwapchainExtension,
            WindowError::VkCreateDevice("ERROR_DEVICE_LOST".into()),
            WindowError::VkCommandPool("ERROR_OUT_OF_DEVICE_MEMORY".into()),
            WindowError::VkAllocCmd("ERROR_OUT_OF_POOL_MEMORY".into()),
            WindowError::VkSemaphoreImageAvailable("ERROR_OUT_OF_HOST_MEMORY".into()),
            WindowError::VkSemaphoreRenderFinished("ERROR_OUT_OF_HOST_MEMORY".into()),
            WindowError::VkFence("ERROR_OUT_OF_HOST_MEMORY".into()),
        ];
        let mut slugs: Vec<&str> = Vec::new();
        for e in &all {
            assert!(
                e.slug().starts_with("window_"),
                "{} is not namespaced to the window rail",
                e.slug()
            );
            for (k, v) in e.fields() {
                assert!(
                    !k.contains(|c: char| c.is_whitespace())
                        && !v.contains(|c: char| c.is_whitespace()),
                    "{k}={v} carries whitespace and would corrupt the space-split log"
                );
            }
            slugs.push(e.slug());
        }
        let before = slugs.len();
        slugs.sort_unstable();
        slugs.dedup();
        assert_eq!(before, slugs.len(), "duplicate WindowError slug");

        // End-to-end: a multi-word driver string collapses to one safe field.
        assert_eq!(
            Emit::decline(
                "host_window_init",
                &WindowError::VkNoUsableDevice("no present-capable device".into()),
            )
            .render(),
            "host_window_init reason=window_vk_no_usable_device detail=no_present-capable_device"
        );
    }

    #[test]
    fn every_window_runtime_decline_names_the_exact_operation() {
        use crate::observe::Decline as _;
        let all = [
            WindowError::PresentAcquire(vk::Result::ERROR_DEVICE_LOST),
            WindowError::PresentResetFence(vk::Result::ERROR_DEVICE_LOST),
            WindowError::PresentResetCommandBuffer(vk::Result::ERROR_DEVICE_LOST),
            WindowError::PresentBeginCommandBuffer(vk::Result::ERROR_DEVICE_LOST),
            WindowError::PresentEndCommandBuffer(vk::Result::ERROR_DEVICE_LOST),
            WindowError::PresentQueueSubmit(vk::Result::ERROR_DEVICE_LOST),
            WindowError::PresentQueue(vk::Result::ERROR_DEVICE_LOST),
            WindowError::StagingCreateImage(vk::Result::ERROR_OUT_OF_DEVICE_MEMORY),
            WindowError::StagingMemoryTypeUnavailable { type_bits: 0x80 },
            WindowError::StagingAllocateMemory {
                bytes: 4096,
                result: vk::Result::ERROR_OUT_OF_DEVICE_MEMORY,
            },
            WindowError::StagingBindMemory(vk::Result::ERROR_MEMORY_MAP_FAILED),
            WindowError::StagingMapMemory {
                bytes: 4096,
                result: vk::Result::ERROR_MEMORY_MAP_FAILED,
            },
            WindowError::DmabufImportExtensionsMissing,
            WindowError::DmabufRingIndexOutOfRange {
                ring_idx: 4,
                ring_len: 3,
            },
        ];
        let mut slugs = std::collections::HashSet::new();
        for decline in all {
            assert!(slugs.insert(decline.slug()), "duplicate {}", decline.slug());
            assert!(decline.slug().starts_with("window_"));
            for (_, value) in decline.fields() {
                assert!(!value.contains(char::is_whitespace));
            }
        }
        assert_eq!(slugs.len(), 14);
    }

    #[test]
    fn seq_gate_uploads_once_per_new_frame_and_elides_repeats() {
        // Fresh staging (no seq yet) always uploads.
        assert!(needs_staging_upload(None, 0));
        assert!(needs_staging_upload(None, 7));
        // Same seq republished every vblank: no re-upload (the win).
        assert!(!needs_staging_upload(Some(7), 7));
        // A new frame's seq forces exactly one upload.
        assert!(needs_staging_upload(Some(7), 8));
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn engine_present_gate_submits_new_frames_and_forced_redraws_only() {
        assert!(needs_engine_present(None, true, None));
        assert!(!needs_engine_present(Some(7), false, Some(7)));
        assert!(needs_engine_present(Some(7), false, Some(8)));
        assert!(needs_engine_present(Some(7), true, Some(7)));
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn guest_geometry_change_requests_one_matching_native_resize() {
        // First frame at the window's own size: no request.
        assert!(!guest_resize_request(None, (1920, 1080), (1920, 1080)));
        // Guest mode change away from the window size: request once.
        assert!(guest_resize_request(
            Some((1920, 1080)),
            (1440, 1080),
            (1920, 1080)
        ));
        // Same guest geometry re-observed: no duplicate request.
        assert!(!guest_resize_request(
            Some((1440, 1080)),
            (1440, 1080),
            (1920, 1080)
        ));
        // Window already matches the new mode: nothing to do.
        assert!(!guest_resize_request(
            Some((1920, 1080)),
            (1440, 1080),
            (1440, 1080)
        ));
    }

    #[test]
    fn steady_desktop_elides_all_but_the_first_upload() {
        // Simulate 1 published frame (seq=1) held across 120 vblanks: the seq
        // gate must upload once and skip the remaining 119.
        let mut staged: Option<u64> = None;
        let mut uploads = 0u32;
        for _vblank in 0..120 {
            let incoming = 1u64; // static desktop republishes the same seq
            if needs_staging_upload(staged, incoming) {
                uploads += 1;
                staged = Some(incoming);
            }
        }
        assert_eq!(
            uploads, 1,
            "static frame must upload exactly once, not per vblank"
        );
    }

    #[test]
    fn direct_present_degrade_line_names_reason() {
        let line = direct_present_degrade_line(
            "import_failed",
            "seq=9 ring=2 1920x1080 err=VK_ERROR_INVALID_EXTERNAL_HANDLE".to_string(),
        );
        assert!(line.starts_with("direct_present_degrade reason=import_failed"));
        assert!(line.contains("seq=9"));
        assert!(line.contains("ring=2"));
        assert!(line.contains("1920x1080"));
    }

    #[test]
    fn direct_present_source_line_names_import_churn() {
        let line = direct_present_source_line(2048, 12, 2000, 48, 3, 197);
        assert!(line.starts_with("direct_present_source presents=2048"));
        assert!(line.contains("uploads=12"));
        assert!(line.contains("dmabuf_blits=2000"));
        assert!(line.contains("staging_blits=48"));
        assert!(line.contains("fresh_imports=3"));
        assert!(line.contains("redundant_fds=197"));
    }

    #[test]
    fn direct_present_import_ack_is_geometry_scoped() {
        let ack = DirectPresentImportAck::default();
        assert!(!ack.is_imported(1920, 1080, 1));

        ack.mark_imported(1920, 1080, 1);
        assert!(ack.is_imported(1920, 1080, 1));
        assert!(!ack.is_imported(1920, 1080, 2));
        assert!(!ack.is_imported(3840, 2160, 1));

        ack.mark_imported(3840, 2160, 2);
        assert!(ack.is_imported(3840, 2160, 2));
        assert!(!ack.is_imported(1920, 1080, 1));
    }

    #[test]
    fn direct_present_revert_alarm_requires_dmabuf_fallback() {
        assert!(!counts_direct_present_revert_alarm(
            10,
            false,
            BlitSource::Staging {
                dmabuf_fallback: false
            }
        ));
        assert!(counts_direct_present_revert_alarm(
            10,
            false,
            BlitSource::Staging {
                dmabuf_fallback: true
            }
        ));
        assert!(!counts_direct_present_revert_alarm(
            0,
            false,
            BlitSource::Staging {
                dmabuf_fallback: true
            }
        ));
        assert!(!counts_direct_present_revert_alarm(
            10,
            true,
            BlitSource::Staging {
                dmabuf_fallback: true
            }
        ));
        assert_eq!(
            next_direct_present_revert_run(
                7,
                10,
                false,
                BlitSource::Staging {
                    dmabuf_fallback: true
                }
            ),
            8
        );
        assert_eq!(
            next_direct_present_revert_run(
                7,
                10,
                false,
                BlitSource::Staging {
                    dmabuf_fallback: false
                }
            ),
            0
        );
    }
}

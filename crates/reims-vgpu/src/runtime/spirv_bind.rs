//! SPIR-V set-0 binding relocation for metal2vulkan + the internal Vulkan engine (Linux product).
//!
//! metal2vulkan decorates every stage independently at DescriptorSet 0 (buffers
//! `[0,32)`, textures `[32,64)`, samplers `[64,96)`, ColorInput / framebuffer
//! fetch `[96,104)`). The engine builds one merged set 0 and rejects duplicate
//! bindings. When vertex and fragment both bind the same Metal buffer index,
//! fragment buffer decorations move by [`FRAG_BUFFER_BINDING_OFFSET`] (into
//! `[104,136)`). When both stages sample textures, fragment sampled-resource
//! decorations in `[32,96)` move by [`FRAG_SAMPLED_RESOURCE_BINDING_OFFSET`]
//! (textures → `[160,192)`, samplers → `[192,224)`). The ColorInput band never
//! moves — the engine binds the input attachment at its un-relocated number.
//!
//! Port of archive `reims-vgpu-backend-vulkan` `spirv.rs` relocation helpers only —
//! structural SPIR-V `OpDecorate Binding` walks, no name heuristics.

/// SPIR-V `OpDecorate` opcode.
const OP_DECORATE: u16 = 71;
const OP_TYPE_IMAGE: u16 = 25;
const OP_TYPE_SAMPLER: u16 = 26;
const OP_TYPE_SAMPLED_IMAGE: u16 = 27;
const OP_TYPE_POINTER: u16 = 32;
const OP_FUNCTION: u16 = 54;
const OP_VARIABLE: u16 = 59;
const OP_FUNCTION_CALL: u16 = 57;
const OP_IMAGE_TEXEL_POINTER: u16 = 60;
const OP_LOAD: u16 = 61;
const OP_STORE: u16 = 62;
const OP_COPY_MEMORY: u16 = 63;
const OP_COPY_MEMORY_SIZED: u16 = 64;
const OP_ACCESS_CHAIN: u16 = 65;
const OP_IN_BOUNDS_ACCESS_CHAIN: u16 = 66;
const OP_PTR_ACCESS_CHAIN: u16 = 67;
const OP_IN_BOUNDS_PTR_ACCESS_CHAIN: u16 = 70;
const OP_COPY_OBJECT: u16 = 83;
const OP_IMAGE_READ: u16 = 98;
const OP_IMAGE_WRITE: u16 = 99;
const OP_IMAGE_QUERY_FORMAT: u16 = 101;
const OP_IMAGE_QUERY_ORDER: u16 = 102;
const OP_IMAGE_QUERY_SIZE_LOD: u16 = 103;
const OP_IMAGE_QUERY_SIZE: u16 = 104;
const OP_IMAGE_QUERY_LOD: u16 = 105;
const OP_IMAGE_QUERY_LEVELS: u16 = 106;
const OP_IMAGE_QUERY_SAMPLES: u16 = 107;
const OP_CONVERT_PTR_TO_U: u16 = 117;
const OP_PTR_CAST_TO_GENERIC: u16 = 121;
const OP_GENERIC_CAST_TO_PTR: u16 = 122;
const OP_GENERIC_CAST_TO_PTR_EXPLICIT: u16 = 123;
const OP_SELECT: u16 = 169;
const OP_ATOMIC_STORE: u16 = 228;
const OP_ATOMIC_EXCHANGE: u16 = 229;
const OP_ATOMIC_COMPARE_EXCHANGE: u16 = 230;
const OP_ATOMIC_COMPARE_EXCHANGE_WEAK: u16 = 231;
const OP_ATOMIC_I_INCREMENT: u16 = 232;
const OP_ATOMIC_I_DECREMENT: u16 = 233;
const OP_ATOMIC_I_ADD: u16 = 234;
const OP_ATOMIC_I_SUB: u16 = 235;
const OP_ATOMIC_S_MIN: u16 = 236;
const OP_ATOMIC_U_MIN: u16 = 237;
const OP_ATOMIC_S_MAX: u16 = 238;
const OP_ATOMIC_U_MAX: u16 = 239;
const OP_ATOMIC_AND: u16 = 240;
const OP_ATOMIC_OR: u16 = 241;
const OP_ATOMIC_XOR: u16 = 242;
const OP_PHI: u16 = 245;
const OP_RETURN_VALUE: u16 = 254;
const OP_ATOMIC_FLAG_TEST_AND_SET: u16 = 318;
const OP_ATOMIC_FLAG_CLEAR: u16 = 319;
const OP_CAPABILITY: u16 = 17;
/// SPIR-V `Capability StorageImageWriteWithoutFormat` (writes to an `Unknown`
/// format storage image). Paired with the Vulkan feature of the same name.
const CAPABILITY_STORAGE_IMAGE_WRITE_WITHOUT_FORMAT: u32 = 34;
/// SPIR-V `Decoration Binding`.
const DECORATION_BINDING: u32 = 33;
const HEADER_WORDS: usize = 5;
const BUFFER_BINDING_LIMIT: u32 = 32;
const SAMPLED_RESOURCE_BINDING_BASE: u32 = 32;
const STORAGE_CLASS_UNIFORM_CONSTANT: u32 = 0;
const STORAGE_CLASS_STORAGE_BUFFER: u32 = 12;

/// metal2vulkan ColorInput band base: `air.render_target` INPUT params
/// (framebuffer fetch, `dest_N`) emit `SubpassData` images at `96+N`. The band
/// `[96,104)` (MRT ≤ 8) must survive BOTH fragment relocations unchanged — the
/// engine binds the input attachment by this number. m2v-synthesized constexpr
/// samplers currently also land here; they are unbindable either way, so
/// preserving the band never makes them worse.
pub const COLOR_INPUT_BINDING_BASE: u32 = 96;
/// Fragment buffer band destination offset (`[0,32)` → `[104,136)`) — starts
/// past the ColorInput band and ends before the relocated sampled bands
/// (textures `[160,192)`, samplers `[192,224)`).
pub const FRAG_BUFFER_BINDING_OFFSET: u32 = 104;
/// Fragment sampled-resource destination offset (textures/samplers `[32,96)` → `+128`).
pub const FRAG_SAMPLED_RESOURCE_BINDING_OFFSET: u32 = 128;
/// Exclusive upper bound of the sampled-resource source band relocated by
/// [`offset_fragment_sampled_resource_bindings`]: textures `[32,64)` + samplers
/// `[64,96)`. Bindings at [`COLOR_INPUT_BINDING_BASE`] and above stay in place.
const SAMPLED_RESOURCE_BINDING_LIMIT: u32 = COLOR_INPUT_BINDING_BASE;

/// metal2vulkan texture band base (Metal texture index N → binding 32+N).
pub const TEXTURE_BINDING_BASE: u32 = 32;
/// metal2vulkan sampler band base (Metal sampler index N → binding 64+N).
pub const SAMPLER_BINDING_BASE: u32 = 64;

/// Image dimensionality declared by a translated SPIR-V sampled-image binding.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SampledImageKind {
    D1,
    D1Array,
    D2,
    D2Array,
    D3,
    Cube,
    CubeArray,
}

/// Sampled-vs-storage class of a texture binding, derived from the translator's
/// reflection (`TextureShape.writable`): a writable texture is a storage image, a
/// read/sample texture is a sampled image. The declared Metal access qualifier is
/// authoritative, so this is exact at translate time — there is no `Unknown`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ImageAccess {
    Sampled,
    Storage,
}

/// Content access proven from the SPIR-V use graph for one storage image.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum StorageImageAccess {
    ReadOnly,
    WriteOnly,
    ReadWrite,
    /// The image object escaped or participated in an operation whose content
    /// access cannot be classified safely.
    Unknown,
    /// More than one image variable declares the same binding.
    AmbiguousBinding,
}

/// Explicit storage-image texel format declared by `OpTypeImage`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ImageFormat {
    Rgba32Float,
    Rgba16Float,
    R16Float,
    Rgba16Uint,
    Rgba8Uint,
    Rgba8Sint,
    Rgba8Unorm,
    Rg16Float,
    R8Unorm,
    Rg8Unorm,
    Rgba32Uint,
    /// Single-channel 32-bit float (SPIR-V `R32f`, enum value 3). This is what
    /// `metal2vulkan` declares for a generic `texture2d<float, access::write>`
    /// (`storage_format_from_name`: a `<float` scalar lowers to `R32f`), so it
    /// arrives on the wire constantly — a full-width float write target whose
    /// real texel format is whatever guest surface gets bound. Like [`Self::R32ui`]
    /// it is specialized against that surface before use; leaving it undecoded
    /// made every such dispatch `Unsupported(3)` and dropped it.
    R32Float,
    /// Single-channel 32-bit uint (SPIR-V `R32ui`). Not emitted by the
    /// translator (which declares `Rgba8ui` for a generic `texture2d<uint,
    /// write>`); the device *specializes* a storage image to this format when
    /// the bound guest surface is `MTLPixelFormatR32Uint`, so the view is
    /// `VK_FORMAT_R32_UINT` and a written `uint4`'s `.x` lane is the full u32
    /// (a `Rgba8ui` raw view would keep only the low byte of each lane).
    R32ui,
    /// SPIR-V `Unknown` storage format (enum value 0): the image carries no
    /// declared texel format, so its `VkImageView` may be any compatible format
    /// and the GPU converts written vec4s to that view's channel order. Reads
    /// need `StorageImageReadWithoutFormat`, writes `StorageImageWriteWithoutFormat`.
    /// The device targets this deliberately for a guest `BGRA8Unorm` storage
    /// surface (viewed `B8G8R8A8_UNORM`), which SPIR-V cannot name directly.
    Unknown,
    /// An explicit format outside the product engine's supported surface.
    Unsupported(u32),
}

impl ImageFormat {
    fn from_raw(raw: u32) -> Self {
        match raw {
            0 => Self::Unknown,
            1 => Self::Rgba32Float,
            2 => Self::Rgba16Float,
            3 => Self::R32Float,
            4 => Self::Rgba8Unorm,
            7 => Self::Rg16Float,
            9 => Self::R16Float,
            13 => Self::Rg8Unorm,
            15 => Self::R8Unorm,
            23 => Self::Rgba8Sint,
            30 => Self::Rgba32Uint,
            31 => Self::Rgba16Uint,
            32 => Self::Rgba8Uint,
            33 => Self::R32ui,
            _ => Self::Unsupported(raw),
        }
    }

    fn raw(self) -> u32 {
        match self {
            Self::Rgba32Float => 1,
            Self::Rgba16Float => 2,
            Self::R32Float => 3,
            Self::Rgba8Unorm => 4,
            Self::Rg16Float => 7,
            Self::R16Float => 9,
            Self::Rg8Unorm => 13,
            Self::R8Unorm => 15,
            Self::Rgba8Sint => 23,
            Self::Rgba32Uint => 30,
            Self::Rgba16Uint => 31,
            Self::Rgba8Uint => 32,
            Self::R32ui => 33,
            Self::Unknown => 0,
            Self::Unsupported(raw) => raw,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ImageFormatSpecializeError {
    MalformedModule,
    MissingBinding(u32),
    AmbiguousBinding(u32),
}

impl crate::observe::Decline for ImageFormatSpecializeError {
    /// The slug table used to live at the *caller*, as a `match` inside
    /// `compute_exec.rs` mapping each variant to a string. That works until a
    /// second caller appears and writes its own table — which is how one check
    /// ends up with two names. It belongs to the type.
    fn slug(&self) -> &'static str {
        match self {
            Self::MalformedModule => "spirv_format_specialize_malformed",
            Self::MissingBinding(_) => "spirv_format_specialize_missing_binding",
            Self::AmbiguousBinding(_) => "spirv_format_specialize_ambiguous_binding",
        }
    }

    fn fields(&self) -> Vec<(&'static str, String)> {
        match self {
            Self::MalformedModule => Vec::new(),
            Self::MissingBinding(b) | Self::AmbiguousBinding(b) => {
                vec![("binding", b.to_string())]
            }
        }
    }
}

/// Write access proven from the SPIR-V pointer-use graph for one storage buffer.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BufferAccess {
    ReadOnly,
    Writable,
    /// A pointer escapes through a function call/return, so local provenance
    /// cannot prove whether the callee writes it.
    PointerEscape,
    /// More than one storage-buffer variable declares the same binding.
    AmbiguousBinding,
}

/// Reflect whether a storage-buffer descriptor can be written by the module.
///
/// Pointer provenance follows the SPIR-V operations that can preserve a buffer
/// pointer (`AccessChain`, `CopyObject`, `Select`, and `Phi`). Stores, copy
/// destinations, and atomics make the binding writable. Pointer calls/returns
/// fail closed as unknown; this deliberately avoids inferring mutability from
/// debug names, guest object ids, or corpus-specific function names.
pub fn buffer_access(words: &[u32], wanted_binding: u32) -> Option<BufferAccess> {
    let bound = *words.get(3)? as usize;
    if words.len() < HEADER_WORDS || bound == 0 {
        return None;
    }
    let mut bindings = vec![None; bound];
    let mut storage = vec![None; bound];
    let mut i = HEADER_WORDS;
    while i < words.len() {
        let word0 = words[i];
        let word_count = (word0 >> 16) as usize;
        let opcode = (word0 & 0xffff) as u16;
        if word_count == 0 || i + word_count > words.len() {
            return None;
        }
        match opcode {
            OP_DECORATE if word_count >= 4 && words[i + 2] == DECORATION_BINDING => {
                let id = words[i + 1] as usize;
                if id < bound {
                    bindings[id] = Some(words[i + 3]);
                }
            }
            OP_VARIABLE if word_count >= 4 => {
                let id = words[i + 2] as usize;
                if id < bound {
                    storage[id] = Some(words[i + 3]);
                }
            }
            _ => {}
        }
        i += word_count;
    }
    let mut roots = bindings.iter().enumerate().filter_map(|(id, binding)| {
        (*binding == Some(wanted_binding) && storage[id] == Some(STORAGE_CLASS_STORAGE_BUFFER))
            .then_some(id)
    });
    let root = roots.next()?;
    if roots.next().is_some() {
        return Some(BufferAccess::AmbiguousBinding);
    }

    let mut derived = vec![false; bound];
    derived[root] = true;
    loop {
        let mut changed = false;
        let mut i = HEADER_WORDS;
        while i < words.len() {
            let word_count = (words[i] >> 16) as usize;
            let opcode = (words[i] & 0xffff) as u16;
            let result_from = match opcode {
                OP_ACCESS_CHAIN
                | OP_IN_BOUNDS_ACCESS_CHAIN
                | OP_PTR_ACCESS_CHAIN
                | OP_IN_BOUNDS_PTR_ACCESS_CHAIN
                    if word_count >= 4 =>
                {
                    derived.get(words[i + 3] as usize).copied() == Some(true)
                }
                OP_COPY_OBJECT
                | OP_PTR_CAST_TO_GENERIC
                | OP_GENERIC_CAST_TO_PTR
                | OP_GENERIC_CAST_TO_PTR_EXPLICIT
                    if word_count >= 4 =>
                {
                    derived.get(words[i + 3] as usize).copied() == Some(true)
                }
                OP_SELECT if word_count >= 6 => [words[i + 4], words[i + 5]]
                    .iter()
                    .any(|id| derived.get(*id as usize).copied() == Some(true)),
                OP_PHI if word_count >= 5 => (i + 3..i + word_count)
                    .step_by(2)
                    .any(|at| derived.get(words[at] as usize).copied() == Some(true)),
                _ => false,
            };
            if result_from && word_count >= 3 {
                let result = words[i + 2] as usize;
                if result < bound && !derived[result] {
                    derived[result] = true;
                    changed = true;
                }
            }
            i += word_count;
        }
        if !changed {
            break;
        }
    }

    let is_derived = |id: u32| derived.get(id as usize).copied() == Some(true);
    let mut unknown = false;
    let mut i = HEADER_WORDS;
    while i < words.len() {
        let word_count = (words[i] >> 16) as usize;
        let opcode = (words[i] & 0xffff) as u16;
        let writable = match opcode {
            OP_STORE | OP_COPY_MEMORY | OP_COPY_MEMORY_SIZED | OP_ATOMIC_STORE
                if word_count >= 2 =>
            {
                is_derived(words[i + 1])
            }
            OP_ATOMIC_EXCHANGE
            | OP_ATOMIC_COMPARE_EXCHANGE
            | OP_ATOMIC_COMPARE_EXCHANGE_WEAK
            | OP_ATOMIC_I_INCREMENT
            | OP_ATOMIC_I_DECREMENT
            | OP_ATOMIC_I_ADD
            | OP_ATOMIC_I_SUB
            | OP_ATOMIC_S_MIN
            | OP_ATOMIC_U_MIN
            | OP_ATOMIC_S_MAX
            | OP_ATOMIC_U_MAX
            | OP_ATOMIC_AND
            | OP_ATOMIC_OR
            | OP_ATOMIC_XOR
            | OP_ATOMIC_FLAG_TEST_AND_SET
                if word_count >= 4 =>
            {
                is_derived(words[i + 3])
            }
            OP_ATOMIC_FLAG_CLEAR if word_count >= 2 => is_derived(words[i + 1]),
            _ => false,
        };
        if writable {
            return Some(BufferAccess::Writable);
        }
        if opcode == OP_FUNCTION_CALL
            && word_count >= 5
            && words[i + 4..i + word_count].iter().copied().any(is_derived)
        {
            unknown = true;
        }
        if opcode == OP_RETURN_VALUE && word_count >= 2 && is_derived(words[i + 1]) {
            unknown = true;
        }
        if opcode == OP_CONVERT_PTR_TO_U && word_count >= 4 && is_derived(words[i + 3]) {
            unknown = true;
        }
        i += word_count;
    }
    Some(if unknown {
        BufferAccess::PointerEscape
    } else {
        BufferAccess::ReadOnly
    })
}

/// Reflect whether a storage image consumes its pre-dispatch contents.
///
/// This follows `OpLoad` from the descriptor variable through the SSA image
/// operations that preserve identity. `OpImageRead` and `OpImageWrite` then
/// provide the content-access contract. Queries do not consume texels;
/// pointer/image escapes fail closed as [`StorageImageAccess::Unknown`].
pub fn storage_image_access(words: &[u32], wanted_binding: u32) -> Option<StorageImageAccess> {
    let bound = *words.get(3)? as usize;
    if words.len() < HEADER_WORDS || bound == 0 {
        return None;
    }
    let mut bindings = vec![None; bound];
    let mut storage = vec![None; bound];
    let mut i = HEADER_WORDS;
    while i < words.len() {
        let word_count = (words[i] >> 16) as usize;
        let opcode = (words[i] & 0xffff) as u16;
        if word_count == 0 || i + word_count > words.len() {
            return None;
        }
        match opcode {
            OP_DECORATE if word_count >= 4 && words[i + 2] == DECORATION_BINDING => {
                let id = words[i + 1] as usize;
                if id < bound {
                    bindings[id] = Some(words[i + 3]);
                }
            }
            OP_VARIABLE if word_count >= 4 => {
                let id = words[i + 2] as usize;
                if id < bound {
                    storage[id] = Some(words[i + 3]);
                }
            }
            _ => {}
        }
        i += word_count;
    }
    let mut roots = bindings.iter().enumerate().filter_map(|(id, binding)| {
        (*binding == Some(wanted_binding) && storage[id] == Some(STORAGE_CLASS_UNIFORM_CONSTANT))
            .then_some(id)
    });
    let root = roots.next()?;
    if roots.next().is_some() {
        return Some(StorageImageAccess::AmbiguousBinding);
    }

    let mut derived = vec![false; bound];
    loop {
        let mut changed = false;
        let mut i = HEADER_WORDS;
        while i < words.len() {
            let word_count = (words[i] >> 16) as usize;
            let opcode = (words[i] & 0xffff) as u16;
            let result_from = match opcode {
                OP_LOAD if word_count >= 4 => words[i + 3] as usize == root,
                OP_COPY_OBJECT if word_count >= 4 => {
                    derived.get(words[i + 3] as usize).copied() == Some(true)
                }
                OP_SELECT if word_count >= 6 => [words[i + 4], words[i + 5]]
                    .iter()
                    .any(|id| derived.get(*id as usize).copied() == Some(true)),
                OP_PHI if word_count >= 5 => (i + 3..i + word_count)
                    .step_by(2)
                    .any(|at| derived.get(words[at] as usize).copied() == Some(true)),
                _ => false,
            };
            if result_from && word_count >= 3 {
                let result = words[i + 2] as usize;
                if result < bound && !derived[result] {
                    derived[result] = true;
                    changed = true;
                }
            }
            i += word_count;
        }
        if !changed {
            break;
        }
    }

    let is_derived = |id: u32| derived.get(id as usize).copied() == Some(true);
    let mut read = false;
    let mut write = false;
    let mut unknown = false;
    let mut i = HEADER_WORDS;
    while i < words.len() {
        let word_count = (words[i] >> 16) as usize;
        let opcode = (words[i] & 0xffff) as u16;
        match opcode {
            OP_IMAGE_READ if word_count >= 5 && is_derived(words[i + 3]) => read = true,
            OP_IMAGE_WRITE if word_count >= 4 && is_derived(words[i + 1]) => write = true,
            OP_IMAGE_QUERY_FORMAT
            | OP_IMAGE_QUERY_ORDER
            | OP_IMAGE_QUERY_SIZE_LOD
            | OP_IMAGE_QUERY_SIZE
            | OP_IMAGE_QUERY_LOD
            | OP_IMAGE_QUERY_LEVELS
            | OP_IMAGE_QUERY_SAMPLES => {
                // Shape/format queries do not consume image contents.
            }
            OP_IMAGE_TEXEL_POINTER if word_count >= 5 && is_derived(words[i + 3]) => unknown = true,
            OP_FUNCTION_CALL
                if word_count >= 5
                    && words[i + 4..i + word_count].iter().copied().any(is_derived) =>
            {
                unknown = true
            }
            OP_RETURN_VALUE if word_count >= 2 && is_derived(words[i + 1]) => unknown = true,
            OP_LOAD | OP_COPY_OBJECT | OP_SELECT | OP_PHI => {}
            _ if words[i + 1..i + word_count].iter().copied().any(is_derived) => unknown = true,
            _ => {}
        }
        i += word_count;
    }
    Some(if unknown || (!read && !write) {
        StorageImageAccess::Unknown
    } else if read && write {
        StorageImageAccess::ReadWrite
    } else if read {
        StorageImageAccess::ReadOnly
    } else {
        StorageImageAccess::WriteOnly
    })
}

/// Reflect the explicit texel format for one image descriptor binding.
///
/// The SPIR-V image-format operand is structural shader ABI. It is independent
/// of debug names and may intentionally differ from the guest texture's Metal
/// pixel format when the shader uses a raw integer view.
pub fn image_format(words: &[u32], wanted_binding: u32) -> Option<ImageFormat> {
    let bound = *words.get(3)? as usize;
    if words.len() < HEADER_WORDS || bound == 0 {
        return None;
    }
    let mut bindings = vec![None; bound];
    let mut pointer_pointee = vec![None; bound];
    let mut variable_type = vec![None; bound];
    let mut formats = vec![None; bound];

    let mut i = HEADER_WORDS;
    while i < words.len() {
        let word0 = words[i];
        let word_count = (word0 >> 16) as usize;
        let opcode = (word0 & 0xffff) as u16;
        if word_count == 0 || i + word_count > words.len() {
            return None;
        }
        match opcode {
            OP_DECORATE if word_count >= 4 && words[i + 2] == DECORATION_BINDING => {
                let id = words[i + 1] as usize;
                if id < bound {
                    bindings[id] = Some(words[i + 3]);
                }
            }
            OP_TYPE_IMAGE if word_count >= 9 => {
                let id = words[i + 1] as usize;
                let raw = words[i + 8];
                let format = ImageFormat::from_raw(raw);
                if id < bound {
                    formats[id] = Some(format);
                }
            }
            OP_TYPE_POINTER if word_count >= 4 => {
                let id = words[i + 1] as usize;
                if id < bound {
                    pointer_pointee[id] = Some(words[i + 3] as usize);
                }
            }
            OP_VARIABLE if word_count >= 4 => {
                let id = words[i + 2] as usize;
                if id < bound {
                    variable_type[id] = Some(words[i + 1] as usize);
                }
            }
            _ => {}
        }
        i += word_count;
    }

    bindings.iter().enumerate().find_map(|(variable, binding)| {
        if *binding != Some(wanted_binding) {
            return None;
        }
        let pointer = variable_type[variable]?;
        let image = pointer_pointee.get(pointer).copied().flatten()?;
        formats.get(image).copied().flatten()
    })
}

/// Specialize storage-image formats from runtime resource ABI, by binding.
///
/// Metal carries the concrete pixel format on the bound texture rather than in
/// the AIR function type. SPIR-V requires that format in `OpTypeImage`, so the
/// product Vulkan path patches only that structural operand after resolving the
/// guest texture. When multiple bindings share one translated image type but
/// resolve to different runtime formats, the helper clones only the image and
/// UniformConstant pointer types and retargets the affected variables.
pub fn specialize_image_formats(
    words: &mut Vec<u32>,
    requested: &[(u32, ImageFormat)],
) -> Result<usize, ImageFormatSpecializeError> {
    let bound = *words
        .get(3)
        .ok_or(ImageFormatSpecializeError::MalformedModule)? as usize;
    if words.len() < HEADER_WORDS || bound == 0 {
        return Err(ImageFormatSpecializeError::MalformedModule);
    }
    let mut bindings = vec![None; bound];
    let mut pointer_pointee = vec![None; bound];
    let mut variable_type = vec![None; bound];
    let mut image_format_word = vec![None; bound];
    let mut image_instruction = vec![None; bound];
    let mut pointer_instruction = vec![None; bound];
    let mut variable_type_word = vec![None; bound];
    let mut insert_at = None;
    let mut i = HEADER_WORDS;
    while i < words.len() {
        let word_count = (words[i] >> 16) as usize;
        let opcode = (words[i] & 0xffff) as u16;
        if word_count == 0 || i + word_count > words.len() {
            return Err(ImageFormatSpecializeError::MalformedModule);
        }
        match opcode {
            OP_DECORATE if word_count >= 4 && words[i + 2] == DECORATION_BINDING => {
                let id = words[i + 1] as usize;
                if id < bound {
                    bindings[id] = Some(words[i + 3]);
                }
            }
            OP_TYPE_IMAGE if word_count >= 9 => {
                let id = words[i + 1] as usize;
                if id < bound {
                    image_format_word[id] = Some(i + 8);
                    image_instruction[id] = Some((i, word_count));
                }
            }
            OP_TYPE_POINTER if word_count >= 4 => {
                let id = words[i + 1] as usize;
                if id < bound {
                    pointer_pointee[id] = Some(words[i + 3] as usize);
                    pointer_instruction[id] = Some((i, word_count));
                }
            }
            OP_VARIABLE if word_count >= 4 => {
                insert_at.get_or_insert(i);
                let id = words[i + 2] as usize;
                if id < bound {
                    variable_type[id] = Some(words[i + 1] as usize);
                    variable_type_word[id] = Some(i + 1);
                }
            }
            OP_FUNCTION => {
                insert_at.get_or_insert(i);
            }
            _ => {}
        }
        i += word_count;
    }

    let mut requested_by_variable = std::collections::BTreeMap::<usize, ImageFormat>::new();
    let mut touched_image_types = std::collections::BTreeSet::<usize>::new();
    for &(wanted_binding, format) in requested {
        let mut variables = bindings
            .iter()
            .enumerate()
            .filter_map(|(id, binding)| (*binding == Some(wanted_binding)).then_some(id));
        let variable = variables
            .next()
            .ok_or(ImageFormatSpecializeError::MissingBinding(wanted_binding))?;
        if variables.next().is_some() {
            return Err(ImageFormatSpecializeError::AmbiguousBinding(wanted_binding));
        }
        let pointer = variable_type[variable]
            .ok_or(ImageFormatSpecializeError::MissingBinding(wanted_binding))?;
        let image_type = pointer_pointee
            .get(pointer)
            .copied()
            .flatten()
            .ok_or(ImageFormatSpecializeError::MissingBinding(wanted_binding))?;
        if image_format_word
            .get(image_type)
            .copied()
            .flatten()
            .is_none()
        {
            return Err(ImageFormatSpecializeError::MissingBinding(wanted_binding));
        }
        requested_by_variable.insert(variable, format);
        touched_image_types.insert(image_type);
    }

    let mut changed = 0;
    let mut next_id = bound as u32;
    let mut extra = Vec::new();
    for image_type in touched_image_types {
        let at = image_format_word[image_type].expect("validated image type");
        let original = ImageFormat::from_raw(words[at]);
        let mut variables = Vec::new();
        for (variable, pointer) in variable_type.iter().enumerate() {
            let Some(pointer) = *pointer else {
                continue;
            };
            if pointer_pointee.get(pointer).copied().flatten() == Some(image_type) {
                variables.push((
                    variable,
                    pointer,
                    requested_by_variable
                        .get(&variable)
                        .copied()
                        .unwrap_or(original),
                ));
            }
        }
        let keep = variables
            .iter()
            .find_map(|(variable, _, format)| {
                (!requested_by_variable.contains_key(variable)).then_some(*format)
            })
            .or_else(|| variables.first().map(|(_, _, format)| *format))
            .ok_or(ImageFormatSpecializeError::MalformedModule)?;
        if words[at] != keep.raw() {
            words[at] = keep.raw();
        }

        let mut clone_groups = std::collections::BTreeMap::<u32, Vec<(usize, usize)>>::new();
        for (variable, pointer, format) in variables {
            if requested_by_variable.get(&variable).copied() == Some(format) && format != original {
                changed += 1;
            }
            if format != keep {
                clone_groups
                    .entry(format.raw())
                    .or_default()
                    .push((variable, pointer));
            }
        }
        for (format_raw, group) in clone_groups {
            let (image_start, image_len) =
                image_instruction[image_type].ok_or(ImageFormatSpecializeError::MalformedModule)?;
            let new_image = next_id;
            next_id += 1;
            let mut image_words = words[image_start..image_start + image_len].to_vec();
            image_words[1] = new_image;
            image_words[8] = format_raw;
            extra.extend(image_words);

            let mut pointer_clones = std::collections::BTreeMap::<usize, u32>::new();
            for &(_, pointer) in &group {
                if pointer_clones.contains_key(&pointer) {
                    continue;
                }
                let (pointer_start, pointer_len) = pointer_instruction[pointer]
                    .ok_or(ImageFormatSpecializeError::MalformedModule)?;
                let new_pointer = next_id;
                next_id += 1;
                let mut pointer_words = words[pointer_start..pointer_start + pointer_len].to_vec();
                pointer_words[1] = new_pointer;
                pointer_words[3] = new_image;
                extra.extend(pointer_words);
                pointer_clones.insert(pointer, new_pointer);
            }
            for (variable, pointer) in group {
                let type_word = variable_type_word[variable]
                    .ok_or(ImageFormatSpecializeError::MalformedModule)?;
                words[type_word] = pointer_clones[&pointer];
            }
        }
    }
    if !extra.is_empty() {
        let at = insert_at.unwrap_or(words.len());
        words.splice(at..at, extra);
        words[3] = next_id;
    }
    for &(binding, format) in requested {
        if image_format(words, binding) != Some(format) {
            return Err(ImageFormatSpecializeError::MissingBinding(binding));
        }
    }
    Ok(changed)
}

/// Ensure the module declares `OpCapability StorageImageWriteWithoutFormat`.
///
/// A storage image whose `OpTypeImage` format is `Unknown` (SPIR-V value 0) may
/// only be written when the module declares this capability (and the device
/// enables the matching Vulkan feature). The device retargets a guest
/// `BGRA8Unorm` storage surface to an `Unknown`-format image viewed
/// `B8G8R8A8_UNORM`, so it must add the capability to the translated kernel,
/// which declares only `Shader`/`Float16`/… Idempotent: does nothing if already
/// present. Capabilities occupy the module's first section (right after the
/// 5-word header), so the new instruction is spliced immediately after the last
/// existing `OpCapability`. Returns `true` if it inserted the capability.
pub fn ensure_storage_write_without_format_capability(words: &mut Vec<u32>) -> bool {
    if words.len() < HEADER_WORDS {
        return false;
    }
    let mut i = HEADER_WORDS;
    let mut insert_at = HEADER_WORDS;
    while i < words.len() {
        let word_count = (words[i] >> 16) as usize;
        let opcode = (words[i] & 0xffff) as u16;
        if word_count == 0 || i + word_count > words.len() {
            break;
        }
        if opcode != OP_CAPABILITY {
            break;
        }
        if word_count >= 2 && words[i + 1] == CAPABILITY_STORAGE_IMAGE_WRITE_WITHOUT_FORMAT {
            return false;
        }
        i += word_count;
        insert_at = i;
    }
    let instr = [
        (2u32 << 16) | OP_CAPABILITY as u32,
        CAPABILITY_STORAGE_IMAGE_WRITE_WITHOUT_FORMAT,
    ];
    words.splice(insert_at..insert_at, instr);
    true
}

/// Reflect every set-0 sampler descriptor binding declared by a SPIR-V module.
///
/// This includes separate `OpTypeSampler` descriptors used by explicit and AIR
/// static samplers, plus combined `OpTypeSampledImage` descriptors. The walk is
/// structural and does not depend on debug names or guest object identifiers.
pub fn sampler_bindings(words: &[u32]) -> Vec<u32> {
    use std::collections::HashSet;

    let mut sampler_types = HashSet::new();
    let mut sampler_ptrs = HashSet::new();
    let mut sampler_vars = HashSet::new();
    let mut decorations = Vec::new();
    let mut i = HEADER_WORDS;
    while i < words.len() {
        let word0 = words[i];
        let word_count = (word0 >> 16) as usize;
        let opcode = (word0 & 0xffff) as u16;
        if word_count == 0 || i + word_count > words.len() {
            break;
        }
        match opcode {
            OP_TYPE_SAMPLER | OP_TYPE_SAMPLED_IMAGE if word_count >= 2 => {
                sampler_types.insert(words[i + 1]);
            }
            OP_TYPE_POINTER if word_count >= 4 => {
                if words[i + 2] == STORAGE_CLASS_UNIFORM_CONSTANT
                    && sampler_types.contains(&words[i + 3])
                {
                    sampler_ptrs.insert(words[i + 1]);
                }
            }
            OP_VARIABLE if word_count >= 4 => {
                if sampler_ptrs.contains(&words[i + 1])
                    && words[i + 3] == STORAGE_CLASS_UNIFORM_CONSTANT
                {
                    sampler_vars.insert(words[i + 2]);
                }
            }
            OP_DECORATE if word_count >= 4 && words[i + 2] == DECORATION_BINDING => {
                decorations.push((words[i + 1], words[i + 3]));
            }
            _ => {}
        }
        i += word_count;
    }
    let mut bindings: Vec<u32> = decorations
        .into_iter()
        .filter_map(|(id, binding)| sampler_vars.contains(&id).then_some(binding))
        .collect();
    bindings.sort_unstable();
    bindings.dedup();
    bindings
}

/// Rewrite fragment SPIR-V: buffer bindings in `[0,32)` += [`FRAG_BUFFER_BINDING_OFFSET`]
/// (destination band `[104,136)`, clear of the `[96,104)` ColorInput band).
pub fn offset_fragment_buffer_bindings(words: &mut [u32]) -> usize {
    let mut rewritten = 0usize;
    let mut i = HEADER_WORDS;
    while i < words.len() {
        let word0 = words[i];
        let word_count = (word0 >> 16) as usize;
        let opcode = (word0 & 0xffff) as u16;
        if word_count == 0 {
            break;
        }
        if opcode == OP_DECORATE && word_count >= 4 && i + 3 < words.len() {
            let decoration = words[i + 2];
            let binding = words[i + 3];
            if decoration == DECORATION_BINDING && binding < BUFFER_BINDING_LIMIT {
                words[i + 3] = binding + FRAG_BUFFER_BINDING_OFFSET;
                rewritten += 1;
            }
        }
        i += word_count;
    }
    rewritten
}

/// Rewrite fragment SPIR-V: sampled bindings `[32,96)` += [`FRAG_SAMPLED_RESOURCE_BINDING_OFFSET`].
///
/// The `[96,104)` ColorInput band ([`COLOR_INPUT_BINDING_BASE`]) is deliberately
/// NOT relocated: the engine binds the framebuffer-fetch input attachment by its
/// un-relocated number, exactly like the storage/descriptor reflectors key on
/// un-relocated bindings.
pub fn offset_fragment_sampled_resource_bindings(words: &mut [u32]) -> usize {
    let mut rewritten = 0usize;
    let mut i = HEADER_WORDS;
    while i < words.len() {
        let word0 = words[i];
        let word_count = (word0 >> 16) as usize;
        let opcode = (word0 & 0xffff) as u16;
        if word_count == 0 || i + word_count > words.len() {
            break;
        }
        if opcode == OP_DECORATE && word_count >= 4 && words[i + 2] == DECORATION_BINDING {
            let binding = words[i + 3];
            if (SAMPLED_RESOURCE_BINDING_BASE..SAMPLED_RESOURCE_BINDING_LIMIT).contains(&binding) {
                // The contract bands make overflow impossible: `binding < 96`
                // and the relocation is 128, so the largest result is 223.
                // Keeping a fallible branch here would register a decline the
                // product cannot produce.
                words[i + 3] = binding + FRAG_SAMPLED_RESOURCE_BINDING_OFFSET;
                rewritten += 1;
            }
        }
        i += word_count;
    }
    rewritten
}

// ---------------------------------------------------------------------------
// Reflection-derived reflectors (single source of truth) + divergence census
// ---------------------------------------------------------------------------
//
// `metal2vulkan::reflect::ShaderReflection` already carries the decoded texture
// shape / access per binding, parsed from the AIR by the SAME decoder the emit
// path uses to write the `OpTypeImage`. The functions below read those facts
// directly, so a consumer never re-walks the emitted SPIR-V. They are keyed on
// the descriptor binding EXACTLY as reflection reports it — the UN-relocated
// number (`TEXTURE_BINDING_BASE + metal_index`), before any fragment +128
// relocation a merged-stage draw later applies.
//
// The `census_reflection_wellformed` guard runs once per translate (miss path)
// and validates, on the live guest's own shaders, that the AIR-derived reflection
// is internally consistent and ABI-versioned. It is the always-on regression
// proxy for the hot path now that texture shape/access is read solely from
// reflection (no second SPIR-V walk to cross-check against).

use metal2vulkan::meta::{TextureDimension, TextureShape};
use metal2vulkan::reflect::{
    ResourceAccess, ResourceKind, ShaderReflection, ShaderStage, REFLECTION_VERSION,
    RESOURCE_DESCRIPTOR_SET,
};

/// Map a decoded [`TextureShape`] to a [`SampledImageKind`] via its `OpTypeImage`
/// Dim + Arrayed. `None` for shapes `SampledImageKind` cannot express (a texel
/// `Buffer`, or a 3D array) — those are legitimate reflection shapes the sampled
/// render path does not support and rejects fail-visibly at the call site.
fn sampled_image_kind_from_shape(shape: &TextureShape) -> Option<SampledImageKind> {
    match (shape.dimension, shape.arrayed) {
        (TextureDimension::D1, false) => Some(SampledImageKind::D1),
        (TextureDimension::D1, true) => Some(SampledImageKind::D1Array),
        (TextureDimension::D2, false) => Some(SampledImageKind::D2),
        (TextureDimension::D2, true) => Some(SampledImageKind::D2Array),
        (TextureDimension::D3, false) => Some(SampledImageKind::D3),
        (TextureDimension::Cube, false) => Some(SampledImageKind::Cube),
        (TextureDimension::Cube, true) => Some(SampledImageKind::CubeArray),
        _ => None,
    }
}

/// Find the texture shape reflection reports for descriptor `binding` (the
/// UN-relocated number). `None` when no binding matches or it carries no shape.
fn texture_shape_for_binding(reflection: &ShaderReflection, binding: u32) -> Option<&TextureShape> {
    reflection.bindings.iter().find_map(|b| {
        (b.descriptor.map(|d| d.binding) == Some(binding))
            .then_some(b.texture_shape.as_ref())
            .flatten()
    })
}

/// Read the texture dimensionality for descriptor `binding` from the translator's
/// reflection — the single source of truth. `binding` is the un-relocated
/// descriptor number (`TEXTURE_BINDING_BASE + metal_index`).
pub fn sampled_image_kind_from_reflection(
    reflection: &ShaderReflection,
    binding: u32,
) -> Option<SampledImageKind> {
    sampled_image_kind_from_shape(texture_shape_for_binding(reflection, binding)?)
}

/// How reflection describes descriptor `binding` for the sampled render path.
/// Lets the call site log a genuine gap fail-visibly while staying silent on the
/// expected "bound but not sampled by this shader" case.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ReflectedSampledKind {
    /// Reflection carries a sampled dimensionality the render path can express.
    Kind(SampledImageKind),
    /// Reflection lists a texture shape here the sampled path cannot express (a
    /// texel `Buffer` or a 3D array) — a genuine unsupported shape.
    Unsupported,
    /// Reflection lists no texture shape at this binding — an unused/unbound slot
    /// (Metal permits binding a texture a shader never samples).
    Absent,
}

/// Classify descriptor `binding` for the sampled render path from reflection.
pub fn reflected_sampled_kind(reflection: &ShaderReflection, binding: u32) -> ReflectedSampledKind {
    match texture_shape_for_binding(reflection, binding) {
        None => ReflectedSampledKind::Absent,
        Some(shape) => match sampled_image_kind_from_shape(shape) {
            Some(kind) => ReflectedSampledKind::Kind(kind),
            None => ReflectedSampledKind::Unsupported,
        },
    }
}

/// Sampled-vs-storage class for descriptor `binding` from the translator's
/// reflection: a `writable` (write/read_write) texture is a storage image, a
/// read/sample texture is sampled. The declared Metal qualifier is authoritative,
/// so this is exact at translate time (never `Unknown`). `None` when reflection
/// carries no texture shape for the binding (an unbound/unused descriptor).
pub fn image_access_from_reflection(
    reflection: &ShaderReflection,
    binding: u32,
) -> Option<ImageAccess> {
    let shape = texture_shape_for_binding(reflection, binding)?;
    Some(if shape.writable {
        ImageAccess::Storage
    } else {
        ImageAccess::Sampled
    })
}

/// The shader-pulled-vertex coverage gate, sourced from reflection instead of a
/// SPIR-V walk: the vertex stage writes `Position`, reads `VertexIndex`, and binds
/// at least one buffer to pull vertices from. The builtins come straight from
/// `vertex_builtins` (the AIR roles that also drive the emit, so no walk can
/// disagree); buffer presence from the reflected `Buffer` bindings. `false` when
/// the reflection carries no vertex builtins (a non-vertex stage). This only
/// GATES the coverage proof — `draw_full_target_coverage_shader_pulled` still
/// evaluates the emitted SPIR-V against the bound buffers (genuine execution, not
/// an interface walk).
pub fn vertex_position_pull_gate(reflection: &ShaderReflection) -> bool {
    let Some(vb) = reflection.vertex_builtins else {
        return false;
    };
    vb.writes_position
        && vb.uses_vertex_index
        && reflection
            .bindings
            .iter()
            .any(|b| matches!(b.kind, ResourceKind::Buffer))
}

/// The reflected buffer descriptor bindings a vertex stage pulls from (sorted,
/// deduped) — the reflection twin of the walk's `storage_bindings`, for the
/// shader-pull coverage-gap diagnostic.
pub fn vertex_pull_buffer_bindings(reflection: &ShaderReflection) -> Vec<u32> {
    let mut v: Vec<u32> = reflection
        .bindings
        .iter()
        .filter(|b| matches!(b.kind, ResourceKind::Buffer))
        .filter_map(|b| b.descriptor.map(|d| d.binding))
        .collect();
    v.sort_unstable();
    v.dedup();
    v
}

/// Validate that the translator's reflection is internally well-formed, once per
/// translate (miss path). This is the always-on regression proxy that replaces
/// the former reflection-vs-SPIR-V cross-check: the hot path now reads texture
/// shape and sampled-vs-storage access solely from reflection, so the guard must
/// catch a reflection that is self-contradictory or emitted against a different
/// ABI — without a second walk of the SPIR-V. Checks:
///   - `reflection_version` matches the ABI this consumer was built against
///     (catches a translator/consumer version skew);
///   - each static sampler carries decoded state and a set-0 descriptor inside
///     the sampler ABI band;
///   - each texture-family binding carries a descriptor location;
///   - the redundant sampled-vs-storage encodings agree — `ResourceKind`
///     (`StorageImage`), `TextureShape.writable`, and `ResourceAccess`. The
///     translator derives all three from one decoded `TextureShape`, so any
///     disagreement is a translator regression the consumer must not trust.
///
/// Logs `m2v_reflect_malformed reason=<slug>` fail-visibly; quiet on a healthy
/// boot (returns the number of violations found, 0 when clean).
pub fn census_reflection_wellformed(reflection: &ShaderReflection, pipeline_ref: u32) -> usize {
    let mut bad = 0;
    if reflection.reflection_version != REFLECTION_VERSION {
        bad += 1;
        crate::observe::fail(format!(
            "m2v_reflect_malformed pipe={pipeline_ref} reason=reflection_version_mismatch \
             got={} want={REFLECTION_VERSION}",
            reflection.reflection_version
        ));
    }
    for b in &reflection.bindings {
        if b.kind == ResourceKind::StaticSampler {
            match (b.descriptor, b.static_sampler) {
                (None, _) => {
                    bad += 1;
                    crate::observe::fail(format!(
                        "m2v_reflect_malformed pipe={pipeline_ref} \
                         reason=static_sampler_no_descriptor metal_index={}",
                        b.metal_index
                    ));
                }
                (Some(descriptor), None) => {
                    bad += 1;
                    crate::observe::fail(format!(
                        "m2v_reflect_malformed pipe={pipeline_ref} \
                         reason=static_sampler_no_state bind={}",
                        descriptor.binding
                    ));
                }
                (Some(descriptor), Some(_))
                    if descriptor.set != RESOURCE_DESCRIPTOR_SET
                        || !(SAMPLER_BINDING_BASE..COLOR_INPUT_BINDING_BASE)
                            .contains(&descriptor.binding) =>
                {
                    bad += 1;
                    crate::observe::fail(format!(
                        "m2v_reflect_malformed pipe={pipeline_ref} \
                         reason=static_sampler_descriptor_out_of_band set={} bind={} \
                         expected_set={RESOURCE_DESCRIPTOR_SET} expected_band={}..{}",
                        descriptor.set,
                        descriptor.binding,
                        SAMPLER_BINDING_BASE,
                        COLOR_INPUT_BINDING_BASE
                    ));
                }
                (Some(_), Some(_)) => {}
            }
            continue;
        }
        if b.static_sampler.is_some() {
            bad += 1;
            crate::observe::fail(format!(
                "m2v_reflect_malformed pipe={pipeline_ref} \
                 reason=static_sampler_state_on_nonstatic kind={:?} metal_index={}",
                b.kind, b.metal_index
            ));
        }
        let texture_family = matches!(
            b.kind,
            ResourceKind::Texture
                | ResourceKind::TextureArray
                | ResourceKind::StorageImage
                | ResourceKind::EmbeddedArgBufferTexture
        );
        if !texture_family {
            continue;
        }
        let binding = b.descriptor.map(|d| d.binding);
        if binding.is_none() {
            bad += 1;
            crate::observe::fail(format!(
                "m2v_reflect_malformed pipe={pipeline_ref} reason=texture_binding_no_descriptor \
                 kind={:?} metal_index={}",
                b.kind, b.metal_index
            ));
        }
        let bind = binding.unwrap_or(0);
        // Storage-vs-sampled must agree across the three encodings the consumer
        // and the translator both derive from the one `TextureShape`.
        let kind_storage = matches!(b.kind, ResourceKind::StorageImage);
        if let Some(writable) = b.texture_shape.as_ref().map(|s| s.writable) {
            if writable != kind_storage {
                bad += 1;
                crate::observe::fail(format!(
                    "m2v_reflect_malformed pipe={pipeline_ref} reason=kind_writable_disagree \
                     bind={bind} kind={:?} writable={writable}",
                    b.kind
                ));
            }
        }
        let access_storage = match b.access {
            Some(ResourceAccess::Storage) => Some(true),
            Some(ResourceAccess::Sampled) => Some(false),
            _ => None,
        };
        if let Some(access_storage) = access_storage {
            if access_storage != kind_storage {
                bad += 1;
                crate::observe::fail(format!(
                    "m2v_reflect_malformed pipe={pipeline_ref} reason=kind_access_disagree \
                     bind={bind} kind={:?} access={:?}",
                    b.kind, b.access
                ));
            }
        }
    }
    bad
}

/// Report a shader's runtime `[[function_constant(N)]]` inventory when it declares any.
///
/// metal2vulkan does not plumb runtime function-constant specialization: its emit
/// path folds every `[[function_constant]]` load to its disabled default (0) and
/// selects that variant (`passes::transform_with_options` `fold_function_constants`).
/// The paravirt command stream, in turn, carries no `MTLFunctionConstantValues` for
/// us to apply — the pipeline/function descriptors decode only refs + the AIR blob,
/// and that AIR is unspecialized (its `air.fc_initializer` globals are
/// `externally_initialized undef`). So a shader that declares runtime function
/// constants is always translated as its FC-disabled variant.
///
/// That is the current, accepted behavior — it renders the system UI clean — but it
/// is a real gap for any shader whose guest-selected FC values differ from the
/// disabled default. This once-per-translate line makes the reliance MEASURABLE:
/// which shaders (by Metal entry name) carry runtime FCs, so a future rendering
/// delta can be correlated with FC usage and the specialization gap sized before any
/// fix. It is diagnostic, not a per-draw failure, so it goes to the OFF-prefixed
/// analysis sink (not `fail`, which must read zero on a healthy boot).
///
/// The input is the reflection's `function_constants` — the translator's single
/// source of truth, scanned once from the AIR `air.fc_initializer` ABI globals — so
/// there is no SPIR-V re-walk. Silent for the common FC-free shader. Returns the
/// count reported (0 = silent) for tests.
pub fn log_folded_function_constants(reflection: &ShaderReflection) -> usize {
    if reflection.function_constants.is_empty() {
        return 0;
    }
    let stage = match reflection.stage {
        ShaderStage::Vertex => "v",
        ShaderStage::Fragment => "f",
        ShaderStage::Kernel => "k",
    };
    let entry = reflection.entry_point.as_deref().unwrap_or("?");
    let inventory: Vec<String> = reflection
        .function_constants
        .iter()
        .map(|fc| format!("{}:{}:{}", fc.index, fc.name, fc.type_name))
        .collect();
    crate::observe::off(format!(
        "fc_folded_disabled stage={stage} entry={entry} count={} fcs=[{}]",
        inventory.len(),
        inventory.join(",")
    ));
    reflection.function_constants.len()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn injects_storage_write_without_format_capability_once() {
        // 5-word header, `OpCapability Shader`, then `OpMemoryModel` (opcode 14).
        let mut words = vec![
            0x0723_0203,       // magic
            0x0001_0600,       // version
            0,                 // generator
            16,                // bound
            0,                 // schema
            (2u32 << 16) | 17, // OpCapability ...
            1,                 // Shader
            (3u32 << 16) | 14, // OpMemoryModel ...
            0,                 // Logical
            1,                 // GLSL450
        ];
        let before = words.len();
        assert!(ensure_storage_write_without_format_capability(&mut words));
        assert_eq!(words.len(), before + 2);
        // The original Shader capability is untouched at the front.
        assert_eq!(words[5], (2u32 << 16) | 17);
        assert_eq!(words[6], 1);
        // The new capability is spliced after the last OpCapability, before the
        // OpMemoryModel section (capabilities must precede everything else).
        assert_eq!(words[7], (2u32 << 16) | 17);
        assert_eq!(words[8], CAPABILITY_STORAGE_IMAGE_WRITE_WITHOUT_FORMAT);
        assert_eq!(
            words[9] & 0xffff,
            14,
            "OpMemoryModel follows the capabilities"
        );
        // Idempotent: a second call is a no-op.
        assert!(!ensure_storage_write_without_format_capability(&mut words));
        assert_eq!(words.len(), before + 2);
    }

    #[test]
    fn image_format_unknown_round_trips_raw_zero() {
        // SPIR-V ImageFormat 0 is `Unknown`; it must survive a raw round trip so
        // `specialize_image_formats` can request and verify it.
        assert_eq!(ImageFormat::from_raw(0), ImageFormat::Unknown);
        assert_eq!(ImageFormat::Unknown.raw(), 0);
    }

    use metal2vulkan::meta::{FunctionConstant, TextureComponent, TextureShape};
    use metal2vulkan::reflect::{
        DescriptorLocation, ResourceBinding, ResourceKind, ShaderReflection, ShaderStage,
        REFLECTION_VERSION,
    };

    fn empty_reflection(stage: ShaderStage) -> ShaderReflection {
        ShaderReflection {
            reflection_version: REFLECTION_VERSION,
            stage,
            entry_point: None,
            bindings: vec![],
            vertex_attributes: vec![],
            varyings: vec![],
            render_targets: vec![],
            depth_members: vec![],
            stencil_members: vec![],
            local_size: None,
            vertex_builtins: None,
            imageblock_layouts: vec![],
            datalayout: None,
            function_constants: vec![],
        }
    }

    fn texture_binding(binding: u32, shape: TextureShape) -> ResourceBinding {
        ResourceBinding {
            kind: ResourceKind::Texture,
            metal_index: binding - TEXTURE_BINDING_BASE,
            descriptor: Some(DescriptorLocation { set: 0, binding }),
            param_index: None,
            address_space: None,
            declared_size: None,
            type_layout: None,
            type_name: None,
            texture_shape: Some(shape),
            embedded_source: None,
            access: None,
            static_sampler: None,
        }
    }

    fn static_sampler_binding(binding: u32) -> ResourceBinding {
        use metal2vulkan::reflect::{
            SamplerAddressMode, SamplerBorderColor, SamplerCompareFunction, SamplerCoordinates,
            SamplerFilter, SamplerMipFilter, SamplerReduction, StaticSamplerState,
        };

        ResourceBinding {
            kind: ResourceKind::StaticSampler,
            metal_index: binding - SAMPLER_BINDING_BASE,
            descriptor: Some(DescriptorLocation {
                set: RESOURCE_DESCRIPTOR_SET,
                binding,
            }),
            param_index: None,
            address_space: None,
            declared_size: None,
            type_layout: None,
            type_name: None,
            texture_shape: None,
            embedded_source: None,
            access: None,
            static_sampler: Some(StaticSamplerState {
                min_filter: SamplerFilter::Linear,
                mag_filter: SamplerFilter::Linear,
                mip_filter: SamplerMipFilter::None,
                address_mode_s: SamplerAddressMode::ClampToEdge,
                address_mode_t: SamplerAddressMode::ClampToEdge,
                address_mode_r: SamplerAddressMode::ClampToEdge,
                coordinates: SamplerCoordinates::Normalized,
                compare_function: SamplerCompareFunction::Never,
                max_anisotropy: 1,
                lod_min_clamp: 0.0,
                lod_max_clamp: 65504.0,
                border_color: SamplerBorderColor::TransparentBlack,
                reduction: SamplerReduction::WeightedAverage,
                lod_bias: 0.0,
                raw_words: [0x807b_ff00_0008_0a49, 0],
            }),
        }
    }

    fn shape(dimension: TextureDimension, arrayed: bool, writable: bool) -> TextureShape {
        TextureShape {
            dimension,
            arrayed,
            multisampled: false,
            component: TextureComponent::Float,
            writable,
            array_ref: false,
            storage_format: None,
        }
    }

    #[test]
    fn reflection_derived_kind_and_access_cover_every_shape() {
        // Dimensionality mapping matches the SPIR-V-walk `SampledImageKind`.
        let cases = [
            (TextureDimension::D1, false, Some(SampledImageKind::D1)),
            (TextureDimension::D1, true, Some(SampledImageKind::D1Array)),
            (TextureDimension::D2, false, Some(SampledImageKind::D2)),
            (TextureDimension::D2, true, Some(SampledImageKind::D2Array)),
            (TextureDimension::D3, false, Some(SampledImageKind::D3)),
            (TextureDimension::Cube, false, Some(SampledImageKind::Cube)),
            (
                TextureDimension::Cube,
                true,
                Some(SampledImageKind::CubeArray),
            ),
            // SPIR-V's SampledImageKind cannot express these — the walk returns
            // None for them too, so reflection must agree.
            (TextureDimension::D3, true, None),
            (TextureDimension::Buffer, false, None),
        ];
        for (dim, arrayed, want) in cases {
            let mut r = empty_reflection(ShaderStage::Fragment);
            let bind = TEXTURE_BINDING_BASE + 3;
            r.bindings
                .push(texture_binding(bind, shape(dim, arrayed, false)));
            assert_eq!(
                sampled_image_kind_from_reflection(&r, bind),
                want,
                "dim={dim:?} arrayed={arrayed}"
            );
        }

        // Access mapping: writable => storage image, else sampled.
        let mut r = empty_reflection(ShaderStage::Kernel);
        let sampled = TEXTURE_BINDING_BASE;
        let storage = TEXTURE_BINDING_BASE + 1;
        r.bindings.push(texture_binding(
            sampled,
            shape(TextureDimension::D2, false, false),
        ));
        r.bindings.push(texture_binding(
            storage,
            shape(TextureDimension::D2, false, true),
        ));
        assert_eq!(
            image_access_from_reflection(&r, sampled),
            Some(ImageAccess::Sampled)
        );
        assert_eq!(
            image_access_from_reflection(&r, storage),
            Some(ImageAccess::Storage)
        );
        // A binding reflection does not carry => None (the walk's miss).
        assert_eq!(
            image_access_from_reflection(&r, TEXTURE_BINDING_BASE + 9),
            None
        );
        assert_eq!(
            sampled_image_kind_from_reflection(&r, TEXTURE_BINDING_BASE + 9),
            None
        );
    }

    #[test]
    fn wellformed_guard_passes_consistent_reflection_and_catches_desync() {
        use metal2vulkan::reflect::ResourceAccess;

        // A consistent sampled 2D texture: kind Texture, !writable, access Sampled.
        let bind = TEXTURE_BINDING_BASE + 2;
        let mut r = empty_reflection(ShaderStage::Fragment);
        let mut b = texture_binding(bind, shape(TextureDimension::D2, false, false));
        b.access = Some(ResourceAccess::Sampled);
        r.bindings.push(b);
        assert_eq!(census_reflection_wellformed(&r, 0), 0);

        // Static samplers must carry decoded state in set 0 inside [64,96).
        let mut static_reflection = empty_reflection(ShaderStage::Fragment);
        static_reflection
            .bindings
            .push(static_sampler_binding(SAMPLER_BINDING_BASE + 1));
        assert_eq!(census_reflection_wellformed(&static_reflection, 0), 0);
        let mut missing_state = static_reflection.clone();
        missing_state.bindings[0].static_sampler = None;
        assert_eq!(census_reflection_wellformed(&missing_state, 0), 1);
        let mut out_of_band = static_reflection.clone();
        out_of_band.bindings[0].descriptor.as_mut().unwrap().binding = COLOR_INPUT_BINDING_BASE;
        assert_eq!(census_reflection_wellformed(&out_of_band, 0), 1);

        // A consistent storage image: kind StorageImage, writable, access Storage.
        let mut rs = empty_reflection(ShaderStage::Kernel);
        let mut sb = texture_binding(
            TEXTURE_BINDING_BASE + 1,
            shape(TextureDimension::D2, false, true),
        );
        sb.kind = ResourceKind::StorageImage;
        sb.access = Some(ResourceAccess::Storage);
        rs.bindings.push(sb);
        assert_eq!(census_reflection_wellformed(&rs, 0), 0);

        // Desync writable=true while kind stays Texture: one violation.
        let mut rbad = empty_reflection(ShaderStage::Fragment);
        rbad.bindings.push(texture_binding(
            bind,
            shape(TextureDimension::D2, false, true),
        ));
        assert_eq!(census_reflection_wellformed(&rbad, 0), 1);

        // Desync access=Storage while kind stays Texture: one violation.
        let mut racc = empty_reflection(ShaderStage::Fragment);
        let mut ba = texture_binding(bind, shape(TextureDimension::D2, false, false));
        ba.access = Some(ResourceAccess::Storage);
        racc.bindings.push(ba);
        assert_eq!(census_reflection_wellformed(&racc, 0), 1);

        // A stale reflection ABI version is a violation on its own.
        let mut rver = empty_reflection(ShaderStage::Fragment);
        rver.reflection_version = REFLECTION_VERSION.wrapping_add(1);
        assert_eq!(census_reflection_wellformed(&rver, 0), 1);
    }

    #[test]
    fn vertex_pull_gate_and_buffer_bindings_from_reflection() {
        use metal2vulkan::reflect::{ResourceBinding, ResourceKind, VertexBuiltins};

        let buffer_binding = |binding: u32| ResourceBinding {
            kind: ResourceKind::Buffer,
            metal_index: binding,
            descriptor: Some(DescriptorLocation { set: 0, binding }),
            param_index: None,
            address_space: None,
            declared_size: None,
            type_layout: None,
            type_name: None,
            texture_shape: None,
            embedded_source: None,
            access: None,
            static_sampler: None,
        };

        // Full shader-pull vertex stage: writes Position, reads VertexIndex, binds
        // buffers 7 and 1 → gate true, buffer list sorted+deduped.
        let mut r = empty_reflection(ShaderStage::Vertex);
        r.vertex_builtins = Some(VertexBuiltins {
            uses_vertex_index: true,
            uses_instance_index: false,
            writes_position: true,
        });
        r.bindings.push(buffer_binding(7));
        r.bindings.push(buffer_binding(1));
        assert!(vertex_position_pull_gate(&r));
        assert_eq!(vertex_pull_buffer_bindings(&r), vec![1, 7]);

        // Missing VertexIndex → gate false.
        let mut no_vi = r.clone();
        no_vi.vertex_builtins = Some(VertexBuiltins {
            uses_vertex_index: false,
            uses_instance_index: false,
            writes_position: true,
        });
        assert!(!vertex_position_pull_gate(&no_vi));

        // No buffer binding → gate false (nothing to pull from).
        let mut no_buf = empty_reflection(ShaderStage::Vertex);
        no_buf.vertex_builtins = r.vertex_builtins;
        assert!(!vertex_position_pull_gate(&no_buf));

        // Non-vertex stage (no builtins) → gate false.
        assert!(!vertex_position_pull_gate(&empty_reflection(
            ShaderStage::Fragment
        )));
    }

    #[test]
    fn folded_function_constants_reported_only_when_present() {
        // FC-free shader: silent (returns 0), no analysis line.
        let none = empty_reflection(ShaderStage::Fragment);
        assert_eq!(log_folded_function_constants(&none), 0);

        // Shader declaring runtime function constants: consumed straight from the
        // reflection (single source of truth), reported once. The count is the
        // inventory length — no SPIR-V re-walk.
        let mut r = empty_reflection(ShaderStage::Kernel);
        r.entry_point = Some("gaussian_blur".to_string());
        r.function_constants = vec![
            FunctionConstant {
                index: 0,
                name: "enable_tap".to_string(),
                type_name: "i1".to_string(),
            },
            FunctionConstant {
                index: 3,
                name: "channel_count".to_string(),
                type_name: "i32".to_string(),
            },
        ];
        assert_eq!(log_folded_function_constants(&r), 2);
    }

    #[test]
    fn offset_buffer_bindings_only_in_band() {
        // Minimal fake module: header + one OpDecorate Binding 3 (4 words).
        let mut words = vec![0u32; 5];
        words[0] = 0x0723_0203; // magic-ish
                                // OpDecorate: opcode 71, wordcount 4 → word0 = (4<<16)|71
        words.push((4u32 << 16) | 71);
        words.push(1); // target id
        words.push(DECORATION_BINDING);
        words.push(3); // binding
                       // Binding 40 (texture band) must not move
        words.push((4u32 << 16) | 71);
        words.push(2);
        words.push(DECORATION_BINDING);
        words.push(40);
        let n = offset_fragment_buffer_bindings(&mut words);
        assert_eq!(n, 1);
        assert_eq!(words[8], 3 + FRAG_BUFFER_BINDING_OFFSET);
        assert_eq!(words[12], 40);
    }

    #[test]
    #[allow(
        clippy::assertions_on_constants,
        reason = "the test pins the binding-band contract constants"
    )]
    fn relocations_preserve_color_input_band_and_stay_collision_free() {
        // The ColorInput band [96,104) (framebuffer-fetch SubpassData images,
        // plus today's synthesized constexpr samplers) must survive BOTH
        // fragment relocations unchanged, and the relocated buffer band
        // [104,136) must not land on it — the engine binds the input
        // attachment at its un-relocated 96+N number.
        let decorate = |id: u32, binding: u32| {
            vec![
                (4u32 << 16) | OP_DECORATE as u32,
                id,
                DECORATION_BINDING,
                binding,
            ]
        };
        let mut words = vec![0x0723_0203, 0x0001_0000, 0, 7, 0];
        words.extend(decorate(1, 1)); // fragment buffer → relocates
        words.extend(decorate(2, 97)); // ColorInput band → stays
        words.extend(decorate(3, 35)); // texture → sampled reloc
        words.extend(decorate(4, 64)); // sampler → sampled reloc

        assert_eq!(offset_fragment_sampled_resource_bindings(&mut words), 2);
        assert_eq!(offset_fragment_buffer_bindings(&mut words), 1);

        let bindings = [words[8], words[12], words[16], words[20]];
        assert_eq!(bindings[0], 1 + FRAG_BUFFER_BINDING_OFFSET);
        assert_eq!(bindings[1], 97);
        assert_eq!(bindings[2], 35 + FRAG_SAMPLED_RESOURCE_BINDING_OFFSET);
        assert_eq!(bindings[3], 64 + FRAG_SAMPLED_RESOURCE_BINDING_OFFSET);
        // All four distinct — no merged-set duplicate bindings.
        let mut sorted = bindings;
        sorted.sort_unstable();
        assert!(sorted.windows(2).all(|w| w[0] != w[1]));
        // Band-map invariants the engine binding math relies on.
        assert!(FRAG_BUFFER_BINDING_OFFSET >= COLOR_INPUT_BINDING_BASE + 8);
        assert!(31 + FRAG_BUFFER_BINDING_OFFSET < 32 + FRAG_SAMPLED_RESOURCE_BINDING_OFFSET);
        assert!(
            (SAMPLED_RESOURCE_BINDING_LIMIT - 1) + FRAG_SAMPLED_RESOURCE_BINDING_OFFSET < u32::MAX
        );
        // The engine's INPUT_ATTACHMENT descriptor and the m2v ColorInput band
        // are the same number by contract; the two constants live on opposite
        // sides of the runtime/engine layering, so pin their equality here.
        #[cfg(feature = "backend-vulkan")]
        assert_eq!(
            COLOR_INPUT_BINDING_BASE,
            crate::backend::vulkan::engine::COLOR_INPUT_BINDING
        );
    }

    #[test]
    fn reflects_storage_image_format_without_names() {
        // A storage image at binding 34 with an explicit Rgba8Uint format operand.
        // `image_format` stays a structural SPIR-V walk (reflection carries no
        // explicit storage-format for the format-specialization path).
        let mut words = vec![0x0723_0203, 0x0001_0000, 0, 7, 0];
        words.extend([
            (9u32 << 16) | OP_TYPE_IMAGE as u32,
            4,
            99,
            1,
            0,
            0,
            0,
            2,
            32,
        ]);
        words.extend([(4u32 << 16) | OP_TYPE_POINTER as u32, 5, 0, 4]);
        words.extend([(4u32 << 16) | OP_VARIABLE as u32, 5, 6, 0]);
        words.extend([(4u32 << 16) | OP_DECORATE as u32, 6, DECORATION_BINDING, 34]);
        assert_eq!(image_format(&words, 34), Some(ImageFormat::Rgba8Uint));
        assert_eq!(image_format(&words, 33), None);
    }

    #[test]
    fn reflects_storage_image_content_access_without_names() {
        // %1=image type, %2=pointer, %3=variable(binding 34), %4=loaded
        // image, %5=read result. The first module reads and writes; removing
        // OpImageRead proves write-only access without decorations or names.
        let mut words = vec![0x0723_0203, 0x0001_0000, 0, 6, 0];
        words.extend([
            (9u32 << 16) | OP_TYPE_IMAGE as u32,
            1,
            99,
            1,
            0,
            0,
            0,
            2,
            32,
        ]);
        words.extend([
            (4u32 << 16) | OP_TYPE_POINTER as u32,
            2,
            STORAGE_CLASS_UNIFORM_CONSTANT,
            1,
        ]);
        words.extend([
            (4u32 << 16) | OP_VARIABLE as u32,
            2,
            3,
            STORAGE_CLASS_UNIFORM_CONSTANT,
        ]);
        words.extend([(4u32 << 16) | OP_DECORATE as u32, 3, DECORATION_BINDING, 34]);
        words.extend([(4u32 << 16) | OP_LOAD as u32, 1, 4, 3]);
        let write_at = words.len();
        words.extend([(4u32 << 16) | OP_IMAGE_WRITE as u32, 4, 90, 91]);
        assert_eq!(
            storage_image_access(&words, 34),
            Some(StorageImageAccess::WriteOnly)
        );
        words.splice(
            write_at..write_at,
            [(5u32 << 16) | OP_IMAGE_READ as u32, 92, 5, 4, 90],
        );
        assert_eq!(
            storage_image_access(&words, 34),
            Some(StorageImageAccess::ReadWrite)
        );

        words.extend([
            (4u32 << 16) | OP_VARIABLE as u32,
            2,
            5,
            STORAGE_CLASS_UNIFORM_CONSTANT,
        ]);
        words.extend([(4u32 << 16) | OP_DECORATE as u32, 5, DECORATION_BINDING, 34]);
        assert_eq!(
            storage_image_access(&words, 34),
            Some(StorageImageAccess::AmbiguousBinding)
        );
    }

    #[test]
    fn specializes_image_format_by_binding_without_names() {
        let mut words = vec![0x0723_0203, 0x0001_0000, 0, 7, 0];
        words.extend([(9u32 << 16) | OP_TYPE_IMAGE as u32, 1, 99, 1, 0, 0, 0, 2, 1]);
        words.extend([(4u32 << 16) | OP_TYPE_POINTER as u32, 2, 0, 1]);
        words.extend([(4u32 << 16) | OP_VARIABLE as u32, 2, 3, 0]);
        words.extend([(4u32 << 16) | OP_DECORATE as u32, 3, DECORATION_BINDING, 34]);
        words.extend([(4u32 << 16) | OP_VARIABLE as u32, 2, 4, 0]);
        words.extend([(4u32 << 16) | OP_DECORATE as u32, 4, DECORATION_BINDING, 35]);

        assert_eq!(
            specialize_image_formats(&mut words, &[(34, ImageFormat::Rgba16Float)]),
            Ok(1)
        );
        assert_eq!(image_format(&words, 34), Some(ImageFormat::Rgba16Float));
        assert_eq!(image_format(&words, 35), Some(ImageFormat::Rgba32Float));
        assert_eq!(
            specialize_image_formats(
                &mut words,
                &[
                    (34, ImageFormat::Rgba16Float),
                    (35, ImageFormat::Rgba8Unorm)
                ]
            ),
            Ok(1)
        );
        assert_eq!(image_format(&words, 34), Some(ImageFormat::Rgba16Float));
        assert_eq!(image_format(&words, 35), Some(ImageFormat::Rgba8Unorm));
    }

    #[test]
    fn specializes_rgba8ui_write_image_to_r32ui() {
        // The exact device patch for an R32Uint-bound `texture2d<uint, write>`:
        // the translator declares the storage image `Rgba8ui` (SPIR-V format
        // token 32); the device re-targets it to `R32ui` (token 33) so the view
        // is VK_FORMAT_R32_UINT. Verify the reflection reads Rgba8ui, the patch
        // rewrites the format operand, and it reads back as R32ui.
        let mut words = vec![0x0723_0203, 0x0001_0000, 0, 6, 0];
        // OpTypeImage %1 : uint 2D depth=0 arrayed=0 ms=0 sampled=2 format=32.
        words.extend([
            (9u32 << 16) | OP_TYPE_IMAGE as u32,
            1,
            99,
            1,
            0,
            0,
            0,
            2,
            32,
        ]);
        words.extend([(4u32 << 16) | OP_TYPE_POINTER as u32, 2, 0, 1]);
        words.extend([(4u32 << 16) | OP_VARIABLE as u32, 2, 3, 0]);
        words.extend([(4u32 << 16) | OP_DECORATE as u32, 3, DECORATION_BINDING, 33]);

        assert_eq!(image_format(&words, 33), Some(ImageFormat::Rgba8Uint));
        assert_eq!(ImageFormat::R32ui.raw(), 33);
        assert_eq!(ImageFormat::from_raw(33), ImageFormat::R32ui);
        assert_eq!(
            specialize_image_formats(&mut words, &[(33, ImageFormat::R32ui)]),
            Ok(1)
        );
        assert_eq!(image_format(&words, 33), Some(ImageFormat::R32ui));
    }

    /// SPIR-V `R32f` is enum value 3 and is what `metal2vulkan` declares for a
    /// generic `texture2d<float, access::write>`. Leaving 3 out of the decode
    /// table turned every such storage image into `Unsupported(3)`, which the
    /// device cannot specialize — the dispatch was dropped rather than run
    /// against the bound guest surface.
    #[test]
    fn r32f_write_image_decodes_and_specializes() {
        let mut words = vec![0x0723_0203, 0x0001_0000, 0, 6, 0];
        // OpTypeImage %1 : float 2D depth=0 arrayed=0 ms=0 sampled=2 format=3.
        words.extend([
            (9u32 << 16) | OP_TYPE_IMAGE as u32,
            1,
            99,
            1,
            0,
            0,
            0,
            2,
            3,
        ]);
        words.extend([(4u32 << 16) | OP_TYPE_POINTER as u32, 2, 0, 1]);
        words.extend([(4u32 << 16) | OP_VARIABLE as u32, 2, 3, 0]);
        words.extend([(4u32 << 16) | OP_DECORATE as u32, 3, DECORATION_BINDING, 33]);

        assert_eq!(ImageFormat::from_raw(3), ImageFormat::R32Float);
        assert_eq!(ImageFormat::R32Float.raw(), 3);
        assert_eq!(image_format(&words, 33), Some(ImageFormat::R32Float));
        assert_eq!(
            specialize_image_formats(&mut words, &[(33, ImageFormat::Rgba32Float)]),
            Ok(1)
        );
        assert_eq!(image_format(&words, 33), Some(ImageFormat::Rgba32Float));
    }

    fn storage_buffer_module(binding: u32) -> Vec<u32> {
        let mut words = vec![0x0723_0203, 0x0001_0000, 0, 12, 0];
        words.extend([
            (4u32 << 16) | OP_VARIABLE as u32,
            1,
            2,
            STORAGE_CLASS_STORAGE_BUFFER,
        ]);
        words.extend([
            (4u32 << 16) | OP_DECORATE as u32,
            2,
            DECORATION_BINDING,
            binding,
        ]);
        // %4 = access-chain %2; pointer provenance must reach the leaf.
        words.extend([(5u32 << 16) | OP_ACCESS_CHAIN as u32, 3, 4, 2, 5]);
        words
    }

    #[test]
    fn reflects_storage_buffer_read_only_without_names() {
        let words = storage_buffer_module(1);
        assert_eq!(buffer_access(&words, 1), Some(BufferAccess::ReadOnly));
        assert_eq!(buffer_access(&words, 0), None);
    }

    #[test]
    fn reflects_storage_buffer_write_through_access_chain() {
        let mut words = storage_buffer_module(1);
        words.extend([(3u32 << 16) | OP_STORE as u32, 4, 6]);
        assert_eq!(buffer_access(&words, 1), Some(BufferAccess::Writable));
    }

    #[test]
    fn storage_buffer_pointer_call_fails_access_closed() {
        let mut words = storage_buffer_module(1);
        // OpFunctionCall result-type, result-id, function-id, pointer arg.
        words.extend([(5u32 << 16) | OP_FUNCTION_CALL as u32, 7, 8, 9, 4]);
        assert_eq!(buffer_access(&words, 1), Some(BufferAccess::PointerEscape));
    }

    #[test]
    fn duplicate_storage_buffer_binding_fails_access_closed() {
        let mut words = storage_buffer_module(1);
        words.extend([
            (4u32 << 16) | OP_VARIABLE as u32,
            1,
            6,
            STORAGE_CLASS_STORAGE_BUFFER,
        ]);
        words.extend([(4u32 << 16) | OP_DECORATE as u32, 6, DECORATION_BINDING, 1]);
        assert_eq!(
            buffer_access(&words, 1),
            Some(BufferAccess::AmbiguousBinding)
        );
    }

    #[test]
    fn reflects_only_declared_sampler_bindings_without_names() {
        // %1 sampler, %2 UniformConstant pointer, %3 sampler variable. A
        // decorated non-sampler %4 must not appear in the result.
        let mut words = vec![0x0723_0203, 0x0001_0000, 0, 5, 0];
        words.extend([(2u32 << 16) | OP_TYPE_SAMPLER as u32, 1]);
        words.extend([
            (4u32 << 16) | OP_TYPE_POINTER as u32,
            2,
            STORAGE_CLASS_UNIFORM_CONSTANT,
            1,
        ]);
        words.extend([
            (4u32 << 16) | OP_VARIABLE as u32,
            2,
            3,
            STORAGE_CLASS_UNIFORM_CONSTANT,
        ]);
        words.extend([(4u32 << 16) | OP_DECORATE as u32, 3, DECORATION_BINDING, 66]);
        words.extend([(4u32 << 16) | OP_DECORATE as u32, 4, DECORATION_BINDING, 99]);
        assert_eq!(sampler_bindings(&words), vec![66]);
    }
}

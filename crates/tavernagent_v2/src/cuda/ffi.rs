//! Dynamically loaded CUDA driver/cuBLAS. No CUDA SDK is needed on the build host.
use libloading::Library;
use std::{
    cell::RefCell,
    collections::HashMap,
    ffi::{CString, c_char, c_int, c_void},
    rc::Rc,
};

type Handle = *mut c_void;
type DevicePtr = u64;
type Status = c_int;
mod sealed {
    pub trait Sealed {}
    impl Sealed for f32 {}
    impl Sealed for f64 {}
    impl Sealed for i32 {}
}
pub trait Element: Copy + Default + sealed::Sealed {}
impl Element for f32 {}
impl Element for f64 {}
impl Element for i32 {}
type Gemm = unsafe extern "C" fn(
    Handle,
    c_int,
    c_int,
    c_int,
    c_int,
    c_int,
    *const f32,
    DevicePtr,
    c_int,
    DevicePtr,
    c_int,
    *const f32,
    DevicePtr,
    c_int,
) -> Status;
type Launch = unsafe extern "C" fn(
    Handle,
    u32,
    u32,
    u32,
    u32,
    u32,
    u32,
    u32,
    Handle,
    *mut *mut c_void,
    *mut *mut c_void,
) -> Status;

struct Api {
    _driver: Library,
    _blas: Library,
    ctx_create: unsafe extern "C" fn(*mut Handle, u32, c_int) -> Status,
    ctx_set: unsafe extern "C" fn(Handle) -> Status,
    ctx_destroy: unsafe extern "C" fn(Handle) -> Status,
    synchronize: unsafe extern "C" fn() -> Status,
    alloc: unsafe extern "C" fn(*mut DevicePtr, usize) -> Status,
    free: unsafe extern "C" fn(DevicePtr) -> Status,
    h2d: unsafe extern "C" fn(DevicePtr, *const c_void, usize) -> Status,
    d2h: unsafe extern "C" fn(*mut c_void, DevicePtr, usize) -> Status,
    d2d: unsafe extern "C" fn(DevicePtr, DevicePtr, usize) -> Status,
    module_load: unsafe extern "C" fn(*mut Handle, *const c_void) -> Status,
    module_unload: unsafe extern "C" fn(Handle) -> Status,
    function: unsafe extern "C" fn(*mut Handle, Handle, *const c_char) -> Status,
    launch: Launch,
    blas_create: unsafe extern "C" fn(*mut Handle) -> Status,
    blas_destroy: unsafe extern "C" fn(Handle) -> Status,
    math_mode: unsafe extern "C" fn(Handle, c_int) -> Status,
    gemm: Gemm,
}

fn check(code: Status, name: &str) -> Result<(), String> {
    if code == 0 {
        Ok(())
    } else {
        Err(format!("{name} failed with status {code}"))
    }
}

impl Api {
    fn load() -> Result<Self, String> {
        // SAFETY: all signatures are the CUDA driver/cuBLAS C ABI. Libraries outlive symbols.
        unsafe {
            let driver = Library::new("libcuda.so.1").map_err(|e| format!("CUDA driver: {e}"))?;
            let blas = Library::new("libcublas.so.12").map_err(|e| format!("cuBLAS 12: {e}"))?;
            macro_rules! sym {
                ($lib:ident,$name:literal) => {
                    *$lib
                        .get(concat!($name, "\0").as_bytes())
                        .map_err(|e| e.to_string())?
                };
            }
            let init: unsafe extern "C" fn(u32) -> Status = sym!(driver, "cuInit");
            check(init(0), "cuInit")?;
            Ok(Self {
                ctx_create: sym!(driver, "cuCtxCreate_v2"),
                ctx_set: sym!(driver, "cuCtxSetCurrent"),
                ctx_destroy: sym!(driver, "cuCtxDestroy_v2"),
                synchronize: sym!(driver, "cuCtxSynchronize"),
                alloc: sym!(driver, "cuMemAlloc_v2"),
                free: sym!(driver, "cuMemFree_v2"),
                h2d: sym!(driver, "cuMemcpyHtoD_v2"),
                d2h: sym!(driver, "cuMemcpyDtoH_v2"),
                d2d: sym!(driver, "cuMemcpyDtoD_v2"),
                module_load: sym!(driver, "cuModuleLoadData"),
                module_unload: sym!(driver, "cuModuleUnload"),
                function: sym!(driver, "cuModuleGetFunction"),
                launch: sym!(driver, "cuLaunchKernel"),
                blas_create: sym!(blas, "cublasCreate_v2"),
                blas_destroy: sym!(blas, "cublasDestroy_v2"),
                math_mode: sym!(blas, "cublasSetMathMode"),
                gemm: sym!(blas, "cublasSgemm_v2"),
                _driver: driver,
                _blas: blas,
            })
        }
    }
}

/// Intentionally !Send/!Sync: one owner thread, one context and one ordered stream.
pub struct Gpu {
    api: Api,
    context: Handle,
    module: Handle,
    blas: Handle,
    functions: RefCell<HashMap<&'static str, Handle>>,
    pool: RefCell<Vec<(DevicePtr, usize)>>,
}

pub enum Arg {
    Ptr(u64),
    Int(i32),
    Float(f32),
    Double(f64),
}
impl Arg {
    fn address(&mut self) -> *mut c_void {
        match self {
            Self::Ptr(v) => (v as *mut u64).cast(),
            Self::Int(v) => (v as *mut i32).cast(),
            Self::Float(v) => (v as *mut f32).cast(),
            Self::Double(v) => (v as *mut f64).cast(),
        }
    }
}

impl Gpu {
    pub fn new(device: usize) -> Result<Rc<Self>, String> {
        let api = Api::load()?;
        let mut gpu = Self {
            api,
            context: std::ptr::null_mut(),
            module: std::ptr::null_mut(),
            blas: std::ptr::null_mut(),
            functions: RefCell::new(HashMap::new()),
            pool: RefCell::new(Vec::new()),
        };
        let device = c_int::try_from(device).map_err(|_| "CUDA device index too large")?;
        let ptx = CString::new(include_str!("kernels.ptx")).map_err(|e| e.to_string())?;
        // SAFETY: valid output handles and NUL-terminated embedded PTX. Partial initialization is dropped.
        unsafe {
            check(
                (gpu.api.ctx_create)(&mut gpu.context, 0, device),
                "cuCtxCreate",
            )?;
            check(
                (gpu.api.module_load)(&mut gpu.module, ptx.as_ptr().cast()),
                "cuModuleLoadData",
            )?;
            check((gpu.api.blas_create)(&mut gpu.blas), "cublasCreate")?;
            check(
                (gpu.api.math_mode)(gpu.blas, 2),
                "cublasSetMathMode(PEDANTIC)",
            )?;
        }
        Ok(Rc::new(gpu))
    }
    fn activate(&self) -> Result<(), String> {
        // SAFETY: context is owned by this thread and remains live.
        unsafe { check((self.api.ctx_set)(self.context), "cuCtxSetCurrent") }
    }
    pub fn sync(&self) -> Result<(), String> {
        self.activate()?;
        // SAFETY: this object owns the current context on this thread.
        unsafe { check((self.api.synchronize)(), "cuCtxSynchronize") }
    }
    pub fn buffer(self: &Rc<Self>, bytes: usize) -> Result<Buffer, String> {
        self.activate()?;
        if bytes == 0 {
            return Err("empty CUDA allocation".into());
        }
        let reused = {
            let mut pool = self.pool.borrow_mut();
            let i = pool
                .iter()
                .enumerate()
                .filter(|(_, (_, n))| *n >= bytes)
                .min_by_key(|(_, (_, n))| *n)
                .map(|(i, _)| i);
            i.map(|i| pool.swap_remove(i))
        };
        let (ptr, capacity) = if let Some(v) = reused {
            v
        } else {
            let mut ptr = 0;
            // SAFETY: driver fills a device pointer for the requested allocation.
            unsafe {
                check((self.api.alloc)(&mut ptr, bytes), "cuMemAlloc")?;
            }
            (ptr, bytes)
        };
        Ok(Buffer {
            gpu: self.clone(),
            ptr,
            bytes,
            capacity,
        })
    }
    pub fn upload<T: Element>(self: &Rc<Self>, data: &[T]) -> Result<Buffer, String> {
        let buffer = self.buffer(std::mem::size_of_val(data))?;
        // SAFETY: destination is sized exactly for data, synchronous copy keeps data alive.
        unsafe {
            check(
                (self.api.h2d)(buffer.ptr, data.as_ptr().cast(), buffer.bytes),
                "cuMemcpyHtoD",
            )?;
        }
        Ok(buffer)
    }
    pub fn launch(
        &self,
        name: &'static str,
        blocks: usize,
        args: &mut [Arg],
    ) -> Result<(), String> {
        self.activate()?;
        if blocks == 0 {
            return Ok(());
        }
        let blocks = u32::try_from(blocks).map_err(|_| "CUDA grid too large")?;
        let cached = { self.functions.borrow().get(name).copied() };
        let function = if let Some(f) = cached {
            f
        } else {
            let c = CString::new(name).map_err(|e| e.to_string())?;
            let mut f = std::ptr::null_mut();
            // SAFETY: module is live and function name is NUL terminated.
            unsafe {
                check(
                    (self.api.function)(&mut f, self.module, c.as_ptr()),
                    "cuModuleGetFunction",
                )?;
            }
            self.functions.borrow_mut().insert(name, f);
            f
        };
        let mut pointers: Vec<_> = args.iter_mut().map(Arg::address).collect();
        // SAFETY: argument layouts match the embedded C kernels; CUDA copies arguments during launch.
        unsafe {
            check(
                (self.api.launch)(
                    function,
                    blocks,
                    1,
                    1,
                    256,
                    1,
                    1,
                    0,
                    std::ptr::null_mut(),
                    pointers.as_mut_ptr(),
                    std::ptr::null_mut(),
                ),
                name,
            )
        }
    }
    /// Row-major GEMM. Transpose and reverse operands for the column-major cuBLAS ABI.
    pub fn gemm(&self, a: &Matrix, ta: bool, b: &Matrix, tb: bool) -> Result<Matrix, String> {
        let (m, k) = if ta {
            (a.cols, a.rows)
        } else {
            (a.rows, a.cols)
        };
        let (bk, n) = if tb {
            (b.cols, b.rows)
        } else {
            (b.rows, b.cols)
        };
        if k != bk {
            return Err("CUDA GEMM shape mismatch".into());
        }
        let out = a.buffer.gpu.matrix(m, n)?;
        let one = 1.0f32;
        let zero = 0.0f32;
        // SAFETY: dimensions/strides describe live buffers; cuBLAS uses host scalar pointers by default.
        unsafe {
            check(
                (self.api.gemm)(
                    self.blas,
                    i32::from(tb),
                    i32::from(ta),
                    n as i32,
                    m as i32,
                    k as i32,
                    &one,
                    b.ptr(),
                    b.cols as i32,
                    a.ptr(),
                    a.cols as i32,
                    &zero,
                    out.ptr(),
                    n as i32,
                ),
                "cublasSgemm",
            )?;
        }
        Ok(out)
    }
    pub fn matrix(self: &Rc<Self>, rows: usize, cols: usize) -> Result<Matrix, String> {
        let count = rows
            .checked_mul(cols)
            .filter(|&n| n > 0 && n <= i32::MAX as usize)
            .ok_or("unsupported CUDA matrix size")?;
        Ok(Matrix {
            buffer: self.buffer(count * 4)?,
            rows,
            cols,
        })
    }
    pub fn from_f32(
        self: &Rc<Self>,
        rows: usize,
        cols: usize,
        data: &[f32],
    ) -> Result<Matrix, String> {
        if rows.checked_mul(cols) != Some(data.len()) || data.len() > i32::MAX as usize {
            return Err("CUDA input shape mismatch".into());
        }
        Ok(Matrix {
            buffer: self.upload(data)?,
            rows,
            cols,
        })
    }
}
impl Drop for Gpu {
    fn drop(&mut self) {
        // SAFETY: !Send ensures destruction on the owner thread, after all owned buffers are returned.
        unsafe {
            if !self.context.is_null() {
                let _ = (self.api.ctx_set)(self.context);
                let _ = (self.api.synchronize)();
            }
            for (ptr, _) in self.pool.get_mut().drain(..) {
                let _ = (self.api.free)(ptr);
            }
            if !self.blas.is_null() {
                let _ = (self.api.blas_destroy)(self.blas);
            }
            if !self.module.is_null() {
                let _ = (self.api.module_unload)(self.module);
            }
            if !self.context.is_null() {
                let _ = (self.api.ctx_destroy)(self.context);
            }
        }
    }
}
pub struct Buffer {
    gpu: Rc<Gpu>,
    pub(super) ptr: u64,
    bytes: usize,
    capacity: usize,
}
impl Buffer {
    pub fn read<T: Element>(&self) -> Result<Vec<T>, String> {
        self.gpu.activate()?;
        if !self.bytes.is_multiple_of(std::mem::size_of::<T>()) {
            return Err("CUDA read alignment mismatch".into());
        }
        let mut out = vec![T::default(); self.bytes / std::mem::size_of::<T>()];
        // SAFETY: output is sized for the full allocation; D2H synchronizes preceding stream work.
        unsafe {
            check(
                (self.gpu.api.d2h)(out.as_mut_ptr().cast(), self.ptr, self.bytes),
                "cuMemcpyDtoH",
            )?;
        }
        Ok(out)
    }
}
impl Drop for Buffer {
    fn drop(&mut self) {
        self.gpu.pool.borrow_mut().push((self.ptr, self.capacity));
    }
}
pub struct Matrix {
    pub(super) buffer: Buffer,
    pub(super) rows: usize,
    pub(super) cols: usize,
}
impl Matrix {
    pub(super) fn belongs_to(&self, gpu: &Rc<Gpu>) -> bool {
        Rc::ptr_eq(&self.buffer.gpu, gpu)
    }
    pub fn ptr(&self) -> u64 {
        self.buffer.ptr
    }
    pub fn len(&self) -> usize {
        self.rows * self.cols
    }
    pub fn copy(&self) -> Result<Self, String> {
        let out = self.buffer.gpu.matrix(self.rows, self.cols)?;
        // SAFETY: both allocations have the same shape and belong to this context.
        unsafe {
            check(
                (self.buffer.gpu.api.d2d)(out.ptr(), self.ptr(), self.len() * 4),
                "cuMemcpyDtoD",
            )?;
        }
        Ok(out)
    }
}

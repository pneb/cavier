use std::fmt;
use std::marker::PhantomData;
use std::mem;

use libc::c_void;

use cudart::memory::{DevicePointer, HostPointer};
use cublas::dtypes::{CublasOperation, CublasPointerMode, CublasStatus_t};
use cublas::{cublasHandle_t, cublasSgemv};

use lib::array2::Array2;

#[derive(Debug, Clone)]
pub struct Array<T>
where
    T: Copy + Clone,
{
    len: usize,
    data: DevicePointer<T>,
    _phantom: PhantomData<T>,
}

unsafe impl<T> Send for Array<T> where T: Copy + Clone {}
unsafe impl<T> Sync for Array<T> where T: Copy + Clone {}

impl<T> Array<T>
where
    T: Copy + Clone,
{
    pub fn new(len: usize) -> Option<Self> {
        let data = DevicePointer::<T>::new(len)?;

        Some(Array {
            len,
            data,
            _phantom: PhantomData,
        })
    }

    pub fn new_from_host(host: &[T]) -> Option<Self> {
        let len = host.len();
        let mut data = DevicePointer::<T>::new(len)?;

        data.memcpy_h2d(host, len)?;

        Some(Array {
            len,
            data,
            _phantom: PhantomData,
        })
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn as_slice(&self) -> HostPointer<T> {
        self.data.as_slice(self.len)
    }

    pub fn as_slice_mut(&mut self) -> HostPointer<T> {
        self.data.as_slice_mut(self.len)
    }

    pub fn add(&mut self, other: &Self) -> Option<()> {
        assert_eq!(self.len, other.len, "Array lengths not equal");

        unsafe {
            let mut handle = cublasHandle_t::default();
            cublasCreate_v2(&mut handle).ok()?;

            cublasSgeam(
                handle,
                CublasOperation::CUBLAS_OP_N,
                CublasOperation::CUBLAS_OP_N,
                self.len as i32,
                1,
                mem::transmute(&1.0),
                self.data.as_cublas_ptr(),
                self.len as i32,
                mem::transmute(&1.0),
                other.data.as_cublas_ptr(),
                other.len as i32,
                self.data.as_cublas_mut_ptr(),
                self.len as i32,
            )
            .ok()?;

            cublasDestroy_v2(handle).ok()?;
        }

        Some(())
    }

    pub fn dot(&self, other: &Self) -> Option<T>
    where
        T: Default,
    {
        assert_eq!(self.len, other.len, "Array lengths not equal");

        let mut res = T::default();

        unsafe {
            let mut handle = cublasHandle_t::default();
            cublasCreate_v2(&mut handle).ok()?;
            cublasSetPointerMode_v2(handle, CublasPointerMode::CUBLAS_POINTER_MODE_DEVICE).ok()?;

            cublasSdot_v2(
                handle,
                self.len as i32,
                self.data.as_cublas_ptr(),
                1,
                other.data.as_cublas_ptr(),
                1,
                mem::transmute(&mut res),
            )
            .ok()?;

            cublasDestroy_v2(handle).ok()?;
        }

        Some(res)
    }
}

impl<T> std::ops::Index<usize> for Array<T>
where
    T: Copy + Clone,
{
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        if index >= self.len {
            panic!("Index out of bounds");
        }

        unsafe { self.data.as_slice(self.len).get_unchecked(index) }
    }
}

impl<T> std::ops::IndexMut<usize> for Array<T>
where
    T: Copy + Clone,
{
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        if index >= self.len {
            panic!("Index out of bounds");
        }

        unsafe { self.data.as_slice_mut(self.len).get_unchecked_mut(index) }
    }
}

impl<T> fmt::Display for Array<T>
where
    T: fmt::Display + Copy + Clone,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let slice = self.as_slice();
        let mut buf = vec![];

        for i in 0..self.len {
            buf.push(slice[i]);
        }

        write!(f, "{:?}", buf)
    }
}

extern "C" {
    fn cublasCreate_v2(handle: *mut cublasHandle_t) -> CublasStatus_t;
    fn cublasDestroy_v2(handle: cublasHandle_t) -> CublasStatus_t;
    fn cublasSgeam(
        handle: cublasHandle_t,
        transa: CublasOperation,
        transb: CublasOperation,
        m: i32,
        n: i32,
        alpha: *const c_void,
        A: *const c_void,
        lda: i32,
        beta: *const c_void,
        B: *const c_void,
        ldb: i32,
        C: *mut c_void,
        ldc: i32,
    ) -> CublasStatus_t;
    fn cublasSetPointerMode_v2(
        handle: cublasHandle_t,
        mode: CublasPointerMode,
    ) -> CublasStatus_t;
    fn cublasSdot_v2(
        handle: cublasHandle_t,
        n: i32,
        x: *const c_void,
        incx: i32,
        y: *const c_void,
        incy: i32,
        result: *mut c_void,
    ) -> CublasStatus_t;
}

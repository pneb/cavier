use std::fmt;
use std::marker::PhantomData;
use std::mem;

use libc::c_void;

use cudart::memory::{DevicePointer, HostPointer};
use cublas::dtypes::{CublasOperation, CublasPointerMode, CublasStatus_t};
use cublas::{cublasHandle_t, cublasSgemv, cublasSgemm};

use lib::array::Array;

#[derive(Debug, Clone)]
pub struct Array2<T>
where
    T: Copy + Clone,
{
    rows: usize,
    columns: usize,
    data: DevicePointer<T>,
    _phantom: PhantomData<T>,
}

unsafe impl<T> Send for Array2<T> where T: Copy + Clone {}
unsafe impl<T> Sync for Array2<T> where T: Copy + Clone {}

impl<T> Array2<T>
where
    T: Copy + Clone,
{
    pub fn new(rows: usize, columns: usize) -> Option<Self> {
        let data = DevicePointer::<T>::new(rows * columns)?;

        Some(Array2 {
            rows,
            columns,
            data,
            _phantom: PhantomData,
        })
    }

    pub fn new_from_host(host: &[T], rows: usize, columns: usize) -> Option<Self> {
        let len = host.len();
        let mut data = DevicePointer::<T>::new(len)?;

        data.memcpy_h2d(host, len)?;

        Some(Array2 {
            rows,
            columns,
            data,
            _phantom: PhantomData,
        })
    }

    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn columns(&self) -> usize {
        self.columns
    }

    pub fn as_slice(&self) -> HostPointer<T> {
        self.data.as_slice(self.rows * self.columns)
    }

    pub fn as_slice_mut(&mut self) -> HostPointer<T> {
        self.data.as_slice_mut(self.rows * self.columns)
    }

    pub fn add(&mut self, other: &Self) -> Option<()> {
        assert_eq!(self.rows, other.rows, "Array2 rows not equal");
        assert_eq!(self.columns, other.columns, "Array2 columns not equal");

        unsafe {
            let mut handle = cublasHandle_t::default();
            cublasCreate_v2(&mut handle).ok()?;

            cublasSgeam(
                handle,
                CublasOperation::CUBLAS_OP_N,
                CublasOperation::CUBLAS_OP_N,
                self.rows as i32,
                self.columns as i32,
                mem::transmute(&1.0),
                self.data.as_cublas_ptr(),
                self.rows as i32,
                mem::transmute(&1.0),
                other.data.as_cublas_ptr(),
                other.rows as i32,
                self.data.as_cublas_mut_ptr(),
                self.rows as i32,
            )
            .ok()?;

            cublasDestroy_v2(handle).ok()?;
        }

        Some(())
    }

    pub fn dot(&self, other: &Self) -> Option<Array2<T>>
    where
        T: Default,
    {
        assert_eq!(
            self.columns, other.rows,
            "Array2 columns not equal to Array2 rows"
        );

        let mut res = Array2::<T>::new(self.rows, other.columns).unwrap();

        unsafe {
            let mut handle = cublasHandle_t::default();
            cublasCreate_v2(&mut handle).ok()?;
            cublasSetPointerMode_v2(handle, CublasPointerMode::CUBLAS_POINTER_MODE_DEVICE).ok()?;

            cublasSgemm(
                handle,
                CublasOperation::CUBLAS_OP_N,
                CublasOperation::CUBLAS_OP_N,
                self.rows as i32,
                other.columns as i32,
                self.columns as i32,
                mem::transmute(&1.0),
                self.data.as_cublas_ptr(),
                self.rows as i32,
                other.data.as_cublas_ptr(),
                other.rows as i32,
                mem::transmute(&0.0),
                res.data.as_cublas_mut_ptr(),
                self.rows as i32,
            )
            .ok()?;

            cublasDestroy_v2(handle).ok()?;
        }

        Some(res)
    }

    pub fn matmul(&self, other: &Array<T>) -> Option<Array<T>>
    where
        T: Default,
    {
        assert_eq!(self.columns, other.len, "Array2 columns not equal to Array length");

        let mut res = Array::<T>::new(self.rows).unwrap();

        unsafe {
            let mut handle = cublasHandle_t::default();
            cublasCreate_v2(&mut handle).ok()?;

            cublasSgemv(
                handle,
                CublasOperation::CUBLAS_OP_N,
                self.rows as i32,
                self.columns as i32,
                mem::transmute(&1.0),
                self.data.as_cublas_ptr(),
                self.rows as i32,
                other.data.as_cublas_ptr(),
                1,
                mem::transmute(&0.0),
                res.data.as_cublas_mut_ptr(),
                1,
            )
            .ok()?;

            cublasDestroy_v2(handle).ok()?;
        }

        Some(res)
    }
}

impl<T> std::ops::Index<(usize, usize)> for Array2<T>
where
    T: Copy + Clone,
{
    type Output = T;

    fn index(&self, index: (usize, usize)) -> &Self::Output {
        let row = index.0;
        let column = index.1;

        if row >= self.rows || column >= self.columns {
            panic!("Index out of bounds");
        }

        unsafe {
            self.data
                .as_slice(self.rows * self.columns)
                .get_unchecked(row * self.columns + column)
        }
    }
}

impl<T> std::ops::IndexMut<(usize, usize)> for Array2<T>
where
    T: Copy + Clone,
{
    fn index_mut(&mut self, index: (usize, usize)) -> &mut Self::Output {
        let row = index.0;
        let column = index.1;

        if row >= self.rows || column >= self.columns {
            panic!("Index out of bounds");
        }

        unsafe {
            self.data
                .as_slice_mut(self.rows * self.columns)
                .get_unchecked_mut(row * self.columns + column)
        }
    }
}

impl<T> fmt::Display for Array2<T>
where
    T: fmt::Display + Copy + Clone,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let slice = self.as_slice();
        let mut buf = vec![];

        for i in 0..self.rows {
            for j in 0..self.columns {
                buf.push(slice[i * self.columns + j]);
            }
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
    fn cublasSgemm(
        handle: cublasHandle_t,
        transa: CublasOperation,
        transb: CublasOperation,
        m: i32,
        n: i32,
        k: i32,
        alpha: *const c_void,
        A: *const c_void,
        lda: i32,
        B: *const c_void,
        ldb: i32,
        beta: *const c_void,
        C: *mut c_void,
        ldc: i32,
    ) -> CublasStatus_t;
    fn cublasSgemv(
        handle: cublasHandle_t,
        trans: CublasOperation,
        m: i32,
        n: i32,
        alpha: *const c_void,
        A: *const c_void,
        lda: i32,
        x: *const c_void,
        incx: i32,
        beta: *const c_void,
        y: *mut c_void,
        incy: i32,
    ) -> CublasStatus_t;
}

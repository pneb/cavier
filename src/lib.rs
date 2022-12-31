#[macro_use]
extern crate lazy_static;
extern crate libc;
extern crate cublas;
extern crate cudart;

pub mod array;
pub mod array2;

pub use array::Array;
pub use array2::Array2;

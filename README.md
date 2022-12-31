# cavier

A small Rust library that helps users to allocate un-allocate  vectors to GPU to accelerate the computation.

## Operational Usage

```rs
extern crate gpu_array;

use gpu_array::{Array, Array2};

fn main() {
    let mut a = Array::<f64>::new(10).unwrap();
    a.fill(1.0).unwrap();

    let mut b = Array::<f64>::new(10).unwrap();
    b.fill(2.0).unwrap();

    println!("a = {:?}", a);
    println!("b = {:?}", b);

    a.add(&b).unwrap();

    println!("a = {:?}", a);
    println!("b = {:?}", b);

    let mut c = Array2::<f64>::new(10, 10).unwrap();
    c.fill(1.0).unwrap();

    let d = c.clone();

    c.add(&d).unwrap();

    println!("c = {:?}", c);
    println!("d = {:?}", d);

    let e = c.dot(&d).unwrap();

    println!("e = {:?}", e);
}
```

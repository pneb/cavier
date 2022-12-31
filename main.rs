extern crate gpu_array;

use gpu_array::{Array, Array2};

fn main() {
    let mut a = Array::<f64>::new(3).unwrap();
    a[0] = 1.0;
    a[1] = 2.0;
    a[2] = 4.0;

    let mut b = Array::<f64>::new(3).unwrap();
    b[0] = 2.0;
    b[1] = 3.0;
    b[2] = 5.0;

    println!("a = {:?}", a);
    println!("b = {:?}", b);

    a.add(&b).unwrap();

    println!("a = {:?}", a);
    println!("b = {:?}", b);

    let mut c = Array2::<f64>::new(2, 2).unwrap();
    c[(0, 0)] = 1.0;
    c[(0, 1)] = 2.0;
    c[(1, 0)] = 3.0;
    c[(1, 1)] = 4.0;

    let d = c.clone();

    c.add(&d).unwrap();

    println!("c = {:?}", c);
    println!("d = {:?}", d);

    let e = c.dot(&d).unwrap();

    println!("e = {:?}", e);
}

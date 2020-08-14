#[macro_use]
extern crate criterion;

// TODO: Should be using black_box for input
//use criterion::black_box;
use criterion::Criterion;

use rand;
use rand::prelude::*;
use rayon::prelude::*;

// long form, with samples specified
criterion_group! {
    name = benches;
    config = Criterion::default().sample_size(10);
    targets =
        // target1,
        // target2
}

criterion_main!(benches);

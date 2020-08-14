#[macro_use]
extern crate criterion;

// TODO: Should be using black_box for input
//use criterion::black_box;
use criterion::{black_box, Criterion};

//use rand;
//use rand::prelude::*;
//use rayon::prelude::*;

// example from https://github.com/bheisler/criterion.rs
fn fibonacci(n: u64) -> u64 {
    match n {
        0 => 1,
        1 => 1,
        n => fibonacci(n - 1) + fibonacci(n - 2),
    }
}

fn example_benchmark(c: &mut Criterion) {
    c.bench_function("fib 20", |b| b.iter(|| fibonacci(black_box(20))));
}

// long form, with samples specified
criterion_group! {
    name = benches;
    config = Criterion::default().sample_size(10);
    targets =
        example_benchmark
        // target1,
        // target2
}

criterion_main!(benches);

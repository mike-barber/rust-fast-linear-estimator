#[macro_use]
extern crate criterion;

// TODO: Should be using black_box for constant inputs
//use criterion::black_box;

use criterion::Criterion;
use ndarray::Array2;
use rand::prelude::*;
use rand::Rng;

#[cfg(target_arch = "x86_64")]
use fast_linear_estimator::matrix_avx::MatrixF32;

#[cfg(target_arch = "aarch64")]
use fast_linear_estimator::matrix_arm::MatrixF32;

// sizes
const NUM_INPUT: usize = 20;
const NUM_OUTPUT: usize = 20;
const NUM_INPUT_SETS: usize = 250;

fn bench_logistic(crit: &mut Criterion) {
    // build random input sets
    let mut rnd = rand::thread_rng();
    let input_sets: Vec<Vec<f32>> = std::iter::repeat_with(|| {
        std::iter::repeat_with(|| rnd.gen_range(-5.0 * 0.5, 5.0 * 0.5))
            .take(NUM_INPUT)
            .collect()
    })
    .take(NUM_INPUT_SETS)
    .collect();

    // coefficients
    let coeff_min: f32 = -0.05;
    let coeff_max: f32 = 0.05;
    let intercept_min: f32 = -0.01;
    let intercept_max: f32 = 0.01;

    // build random coefficients -- row major order;
    //   inputs are rows sequential; outputs are columns stride
    let mut coeff = [[0f32; NUM_INPUT]; NUM_OUTPUT];
    let mut coeff_transpose = [[0f32; NUM_OUTPUT]; NUM_INPUT];
    let mut coeff_nd = Array2::<f32>::zeros((NUM_OUTPUT, NUM_INPUT));
    let mut coeff_nd_transpose = Array2::<f32>::zeros((NUM_INPUT, NUM_OUTPUT));
    for ip in 0..NUM_INPUT {
        for op in 0..NUM_OUTPUT {
            let v = rnd.gen_range(coeff_min, coeff_max);
            // normal arrays
            coeff[op][ip] = v;
            coeff_transpose[ip][op] = v;
            // ndarray
            coeff_nd[[op, ip]] = v;
            coeff_nd_transpose[[ip, op]] = v;
        }
    }

    let mut intercepts = [0f32; NUM_OUTPUT];
    for op in 0..NUM_OUTPUT {
        let v = rnd.gen_range(intercept_min, intercept_max);
        intercepts[op] = v;
    }

    crit.bench_function("choose-input", |b| {
        b.iter(|| {
            let input = input_sets.iter().choose(&mut rnd).unwrap();
            input[0]
        })
    });

    // MatrixF32 benchmark
    {
        let vec_coeff: Vec<Vec<f32>> = coeff_transpose.iter().map(|r| r.to_vec()).collect();
        let mat = MatrixF32::create_from_rows(&vec_coeff, &intercepts).unwrap();

        // copied to f32 output directly
        let mut output_f32 = vec![0f32; mat.num_columns];
        crit.bench_function("matrix-product", |b| {
            b.iter(|| {
                let input = input_sets.iter().choose(&mut rnd).unwrap();

                let some = mat.product(&input, &mut output_f32);
                assert!(some.is_some());

                output_f32[0]
            })
        });

        // copied to f32 output directly
        let mut output_f32 = vec![0f32; mat.num_columns];
        crit.bench_function("matrix-softmax", |b| {
            b.iter(|| {
                let input = input_sets.iter().choose(&mut rnd).unwrap();

                let some = mat.product_softmax_cumulative_approx(&input, &mut output_f32);
                assert!(some.is_some());

                output_f32[0]
            })
        });

        // copied to f32 output directly, sleef exp
        let mut output_f32 = vec![0f32; mat.num_columns];
        crit.bench_function("matrix-softmax-sleef", |b| {
            b.iter(|| {
                let input = input_sets.iter().choose(&mut rnd).unwrap();

                let some = mat.product_softmax_cumulative_sleef(&input, &mut output_f32);
                assert!(some.is_some());

                output_f32[0]
            })
        });
    }

    // directly implemented with iterators
    {
        // directly implemented with iterators
        // note: this is misleadingly fast; it relies on the fact that the dimensions
        //       are constant, and the compiler takes advantage of this. 
        //       performance is a lot lower with variable input sizes :)
        //
        //       the MatrixAvxF32 accepts different dimensions. This does not, essentially.
        // 
        crit.bench_function("matrix-direct-product (const size)", |b| {
            b.iter(|| {
                let a = input_sets.iter().choose(&mut rnd).unwrap();
                let mut r = [0.0; NUM_OUTPUT];

                // matrix mult
                for j in 0..NUM_OUTPUT {
                    r[j] = a
                        .iter()
                        .zip(coeff[j].iter())
                        .map(|(u, v)| u * v)
                        .sum::<f32>();
                }
                r[0]
            })
        });

        crit.bench_function("matrix-direct-softmax (const size)", |b| {
            b.iter(|| {
                let input_index = rnd.gen_range(0, NUM_INPUT_SETS);
                let a = &(input_sets[input_index]);

                let mut r = [0.0; NUM_OUTPUT];

                // matrix mult
                for j in 0..NUM_OUTPUT {
                    r[j] = a
                        .iter()
                        .zip(coeff[j].iter())
                        .map(|(u, v)| u * v)
                        .sum::<f32>();
                }

                // softmax with normalise
                r.iter_mut().for_each(|v| *v = f32::exp(*v));
                let mut cumulative = 0f32;
                for v in r.iter_mut() {
                    cumulative += *v;
                    *v = cumulative;
                }
                r[0]
            })
        });
    }

    // ndarray benchmark
    {
        use ndarray::*;

        // without allocation
        crit.bench_function("ndarray-setup-input-view", |b| {
            b.iter(|| {
                let input = input_sets.iter().choose(&mut rnd).unwrap();
                let a = ArrayView1::from(input);
                a[0]
            })
        });

        // you're still paying for allocation from `dot`
        crit.bench_function("ndarray-product", |b| {
            b.iter(|| {
                let input = input_sets.iter().choose(&mut rnd).unwrap();
                let a = ArrayView1::from(input);
                let res = coeff_nd.dot(&a);
                res[0]
            })
        });

        // you're still paying for allocation from `dot`
        crit.bench_function("ndarray-product-transposed", |b| {
            b.iter(|| {
                let input = input_sets.iter().choose(&mut rnd).unwrap();
                let a = ArrayView1::from(input);
                let res = &a.dot(&coeff_nd_transpose);
                res[0]
            })
        });

        // using an iterator
        crit.bench_function("ndarray-product-transposed-iterators", |b| {
            // pre-allocated and reusable result
            let mut res = ndarray::Array1::<f32>::zeros((NUM_OUTPUT,));
            b.iter(|| {
                let input = input_sets.iter().choose(&mut rnd).unwrap();
                let a = ArrayView1::from(input);
                res.fill(0.0);
                Zip::from(coeff_nd_transpose.genrows())
                    .and(&a)
                    .apply(|cf,inp| {
                        Zip::from(cf)
                            .and(&mut res)
                            .apply(|cc,rr| {
                                *rr = cc * inp;
                            });
                    });
                res[0]
            })
        });
    }
}

// long form, with samples specified
criterion_group! {
    name = benches;
    config = Criterion::default().sample_size(10);
    targets =
        //example_benchmark,
        bench_logistic
}

criterion_main!(benches);

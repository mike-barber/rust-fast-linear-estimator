# Fast linear and logistic estimation using Rust intrinsics and C# 

This is more of a proof of concept than an actual library. However, it does work and is pretty fast. These are the objectives I had in mind: 

- test using a natively compiled Rust library from C# (.net core, of course) on Linux and Windows (x86_64)
    - create a safe C# wrapper that preserves the invariants required
    - reduce the costs of calling the native library by using `Span` and `stackalloc` where appropriate
    - avoid any allocations in Rust or C#
- use AVX intrinsics in Rust to perform a fast matrix multiplication, and compare with
    - direct multiplication using iterators
    - a normal lib appropriate for this kind of task, like `ndarray`
- work out a fast, relatively low accuracy way to approximate an exponential function
    - this is is required for the [softmax](https://en.wikipedia.org/wiki/Softmax_function) part of the logistic estimation
    - normal `exp` has way more accuracy than required for inference tasks, and is generally quite slow; implementations vary. 
    - approximate implementation using avx2 intrinsics is really fast
    - in the interests of performance over accuracy, I'm using a 4th order interpolation; refer to the resources below for the sources.

## BLAS

The obvious question might be: why not BLAS or MKL? There are a few reasons for this.

- it's not much fun (this was a learning exercise as much as anything)
- it's quite complex to get the build working on both Windows and Linux, so I haven't included it in this project; I might throw it on a branch or something if anyone is interested
- it's actually *not* so fast for small matrices like these based on initial testing.
    - BLAS will probably significantly outperform all of this stuff with larger matrices 
    - I have a deliberately simple algorithm, but the simplicity of it works in our favour for small matrices
- MKL: it's quite Intel specific; I'm running an AMD processor and interested in playing around with ARM too.

## Structure

This code does two things

### Linear estimate from a regression model

`y = x * [coeff] + [intercepts]`

In R, 

```R
x = 1:2
coeff = t(matrix(1:6, ncol=2))
intercept = c(10,20,30)
x %*% coeff + intercept
```
with results
```
> x
[1] 1 2
> coeff
     [,1] [,2] [,3]
[1,]    1    2    3
[2,]    4    5    6
> x %*% coeff + intercept
     [,1] [,2] [,3]
[1,]   19   32   45
```

### Logistic estimate from a regression model 

Because of the way I want to use the results, I'm returning the cumulative sum of the softmax, without normalising it. Normally we'd sum the vector and divide it by this sum. I'm doing it a bit differently here. It's fairly trivial to add a method to return the probabilities or most likely class if desired.

Example is similar to the above:
```R
coeff = t(matrix(1:6, ncol=2))
intercept = c(0.1, 0.2, 0.3)
x = c(0.1, 0.5)
logit = x %*% coeff + intercept
cumsum(exp(logit))
```
with results
```
> logit
     [,1] [,2] [,3]
[1,]  2.2  2.9  3.6
> cumsum(exp(logit))
[1]  9.025013 27.199159 63.797393
```

# Future plans

## C# intrinsics

C# now supports x86_64 intrinsics, so I will probably add the identical algorithm in C# and compare the performance. This wasn't really the point of this work though: I really wanted to work out how to attach a fast algorithm written in Rust to C#. 

Some initial experimentation with C# suggested the performance would be good, but not quite on par with Rust since the compiler can't optimise as effectively. However, we are paying a small interop cost calling Rust, so it may prove to be just as effective overall.

## ARM support

On Rust `nightly`, we have support for `aarch64` (ARM 64) intrinsics. It'll be interesting to add a variant of the same algorithm to test it on ARM too. I've already verified this works on my RaspberryPi 4 (with Ubuntu, because Raspbian is still 32b). Some work needs to be done around figuring out how to use the conditional compilation effectively, and working out how to cross-compile for ARM from my workstation. It takes a while to build on the Pi ;)

It's also an interesting potential use, as C# does not have ARM intrinsics yet. If it works well on the Pi, I should probably benchmark it on some powerful hardware, like an AWS Graviton2 instance. It could be useful as a way to leverage that sort of hardware until C# supports the intrinsics natively.

It'll be interesting to keep an eye on Rust SIMD in general, particularly the [packed_simd](https://rust-lang.github.io/packed_simd/packed_simd/) work going on.

## Build process improvements

- Figure out how to build multiple Rust libs for different targets (Windows, Linux, ARM), and package them into a Nuget so the library is usable across platforms. This will require a bit of fiddling. But it has been done before, as in the Confluent Kafka libraries for C#, that rely on the native librdkafka.
- Improve the build process for C# so I don't need to do as much file copying with native libs.

# Resources and acknowledgements

I didn't make up my own exponential approximation algorithm. There are several algorithms out there for fast exponential approximation, including some SSE and AVX ones. The resources that were particularly useful are noted here. The exponential approximation used in this code is a synthesis of the approaches noted below, and most of the credit is due to those authors:

- math_avxfun: http://software-lisc.fbk.eu/avx_mathfun/avx_mathfun.h
- inavec: https://gitlab.mpcdf.mpg.de/bbramas/inastemp 
    - constants taken from https://gitlab.mpcdf.mpg.de/bbramas/inastemp/-/blob/master/Src/Common/InaFastExp.hpp
    - as explained here: http://berenger.eu/blog/csimd-fast-exponential-computation-on-simd-architectures-implementation/
    - Remez approach is more accurate across the range than doing a least squares fit in of the polynomial in R with lm(...)
- shibatch's Sleef library: https://github.com/shibatch/sleef seems to follow a similar approach

# Results 

Environment:

- Windows 10 (Version 2004)
- Ryzen 3900X, 3600MHz RAM, eco mode, otherwise stock settings
- Rust stable 1.45.2 - with config `target-cpu=skylake`, release mode of course
- .NET Core SDK 3.1.401
- logs where applicable saved [here](saved_results)

## Rust benchmarks

- using the excellent [Criterion](https://crates.io/crates/criterion) crate
- benchmarks relevant to this implementation are marked with `*`.
- other timings are for comparison; have a look at the code to see the implementation
- 20 inputs, 20 outputs

```
matrix-product          time:   [35.450 ns 35.680 ns 35.979 ns] *
matrix-softmax          time:   [62.219 ns 62.401 ns 62.583 ns] *
matrix-direct-product   time:   [48.816 ns 49.097 ns 49.591 ns]
matrix-direct-softmax   time:   [190.48 ns 191.06 ns 191.57 ns]
ndarray-product         time:   [230.89 ns 231.82 ns 232.83 ns]
```


## C# benchmarks

- using the excellent [BenchmarkDotNet](https://github.com/dotnet/BenchmarkDotNet) library 
- benchmarks relevant to this implementation are marked with `*`.
- other timings are for comparison and simple C# implementations; have a look at the code for details.
- the final two benchmarks are for a parallel test of the library over a large number of iterations; these would be relevant for someone interested using this for inference for a large input set, for instance.
- 20 inputs, 20 outputs as per [EstimatorBench.cs](csharp/FastLinearEstimator.Bench/EstimatorBench.cs)

|                     Method |              Mean |            Error |           StdDev |  |
|--------------------------- |------------------:|-----------------:|-----------------:|--|
|           BenchRustProduct |          60.52 ns |         0.767 ns |         0.507 ns | *|
|         BenchCSharpProduct |         415.55 ns |         2.938 ns |         1.943 ns |  |
|           BenchRustSoftmax |          88.60 ns |         0.504 ns |         0.300 ns | *| 
|         BenchCSharpSoftmax |         548.54 ns |         5.493 ns |         3.633 ns |  |
| LargeParallelCSharpSoftmax | 241,181,575.00 ns | 3,817,938.772 ns | 2,525,330.106 ns |  |
|   LargeParallelRustSoftmax |  34,845,397.32 ns |   370,337.996 ns |   193,693.933 ns |  |


For testing various different input and output sizes, [EstimatorBenchSizeVariations.cs](csharp/FastLinearEstimator.Bench/EstimatorBenchSizeVariations.cs) is relevant, producing the following results.

The native C# algorithm is only faster for very small problems: 2 outputs, and less than 6 inputs; this is because we're not paying an interop and method call cost.

|             Method | NumInputs | NumOutputs |         Mean |      Error |     StdDev |
|------------------- |---------- |----------- |-------------:|-----------:|-----------:|
| BenchCSharpSoftmax |         2 |          2 |     33.60 ns |   0.468 ns |   0.309 ns |
|   BenchRustSoftmax |         2 |          2 |     41.32 ns |   0.566 ns |   0.375 ns |
| BenchCSharpSoftmax |         2 |          3 |     41.57 ns |   0.339 ns |   0.202 ns |
|   BenchRustSoftmax |         2 |          3 |     42.03 ns |   0.279 ns |   0.185 ns |
| BenchCSharpSoftmax |         2 |          4 |     47.98 ns |   0.423 ns |   0.280 ns |
|   BenchRustSoftmax |         2 |          4 |     42.98 ns |   1.061 ns |   0.632 ns |
| BenchCSharpSoftmax |         2 |          6 |     60.89 ns |   1.056 ns |   0.629 ns |
|   BenchRustSoftmax |         2 |          6 |     44.15 ns |   0.243 ns |   0.161 ns |
| BenchCSharpSoftmax |         2 |          8 |     72.95 ns |   0.861 ns |   0.570 ns |
|   BenchRustSoftmax |         2 |          8 |     45.44 ns |   0.353 ns |   0.210 ns |
| BenchCSharpSoftmax |         2 |         10 |     87.01 ns |   0.934 ns |   0.618 ns |
|   BenchRustSoftmax |         2 |         10 |     52.22 ns |   0.533 ns |   0.353 ns |
| BenchCSharpSoftmax |         2 |         15 |    126.41 ns |   0.907 ns |   0.600 ns |
|   BenchRustSoftmax |         2 |         15 |     51.73 ns |   0.504 ns |   0.333 ns |
| BenchCSharpSoftmax |         2 |         20 |    161.46 ns |   1.170 ns |   0.774 ns |
|   BenchRustSoftmax |         2 |         20 |     60.90 ns |   0.292 ns |   0.174 ns |
| BenchCSharpSoftmax |         2 |         30 |    226.45 ns |   1.343 ns |   0.888 ns |
|   BenchRustSoftmax |         2 |         30 |     73.05 ns |   0.587 ns |   0.349 ns |
| BenchCSharpSoftmax |         2 |         50 |    362.48 ns |   2.094 ns |   1.095 ns |
|   BenchRustSoftmax |         2 |         50 |     99.58 ns |   0.222 ns |   0.132 ns |
| BenchCSharpSoftmax |         2 |        100 |    702.54 ns |   6.144 ns |   4.064 ns |
|   BenchRustSoftmax |         2 |        100 |    157.55 ns |   1.532 ns |   1.013 ns |
| BenchCSharpSoftmax |         3 |          2 |     35.34 ns |   0.369 ns |   0.220 ns |
|   BenchRustSoftmax |         3 |          2 |     41.77 ns |   0.264 ns |   0.157 ns |
| BenchCSharpSoftmax |         3 |          3 |     44.49 ns |   0.297 ns |   0.197 ns |
|   BenchRustSoftmax |         3 |          3 |     42.82 ns |   0.281 ns |   0.186 ns |
| BenchCSharpSoftmax |         3 |          4 |     51.20 ns |   0.149 ns |   0.078 ns |
|   BenchRustSoftmax |         3 |          4 |     43.42 ns |   0.354 ns |   0.234 ns |
| BenchCSharpSoftmax |         3 |          6 |     66.70 ns |   0.544 ns |   0.360 ns |
|   BenchRustSoftmax |         3 |          6 |     45.15 ns |   0.624 ns |   0.413 ns |
| BenchCSharpSoftmax |         3 |          8 |     81.01 ns |   0.729 ns |   0.482 ns |
|   BenchRustSoftmax |         3 |          8 |     46.59 ns |   0.672 ns |   0.444 ns |
| BenchCSharpSoftmax |         3 |         10 |     97.54 ns |   0.970 ns |   0.641 ns |
|   BenchRustSoftmax |         3 |         10 |     52.32 ns |   0.239 ns |   0.125 ns |
| BenchCSharpSoftmax |         3 |         15 |    144.18 ns |   1.251 ns |   0.828 ns |
|   BenchRustSoftmax |         3 |         15 |     54.02 ns |   0.219 ns |   0.130 ns |
| BenchCSharpSoftmax |         3 |         20 |    181.35 ns |   2.206 ns |   1.459 ns |
|   BenchRustSoftmax |         3 |         20 |     63.34 ns |   0.385 ns |   0.255 ns |
| BenchCSharpSoftmax |         3 |         30 |    260.60 ns |   2.430 ns |   1.446 ns |
|   BenchRustSoftmax |         3 |         30 |     75.99 ns |   0.387 ns |   0.230 ns |
| BenchCSharpSoftmax |         3 |         50 |    412.48 ns |   3.162 ns |   2.092 ns |
|   BenchRustSoftmax |         3 |         50 |    102.77 ns |   0.788 ns |   0.469 ns |
| BenchCSharpSoftmax |         3 |        100 |    796.50 ns |   4.776 ns |   2.842 ns |
|   BenchRustSoftmax |         3 |        100 |    164.82 ns |   1.330 ns |   0.880 ns |
| BenchCSharpSoftmax |         4 |          2 |     37.83 ns |   0.378 ns |   0.225 ns |
|   BenchRustSoftmax |         4 |          2 |     42.82 ns |   0.352 ns |   0.209 ns |
| BenchCSharpSoftmax |         4 |          3 |     47.43 ns |   0.243 ns |   0.160 ns |
|   BenchRustSoftmax |         4 |          3 |     43.35 ns |   0.346 ns |   0.229 ns |
| BenchCSharpSoftmax |         4 |          4 |     56.33 ns |   0.595 ns |   0.394 ns |
|   BenchRustSoftmax |         4 |          4 |     43.61 ns |   0.155 ns |   0.092 ns |
| BenchCSharpSoftmax |         4 |          6 |     73.94 ns |   0.723 ns |   0.430 ns |
|   BenchRustSoftmax |         4 |          6 |     45.39 ns |   0.292 ns |   0.193 ns |
| BenchCSharpSoftmax |         4 |          8 |     91.73 ns |   0.948 ns |   0.564 ns |
|   BenchRustSoftmax |         4 |          8 |     46.67 ns |   0.258 ns |   0.170 ns |
| BenchCSharpSoftmax |         4 |         10 |    109.54 ns |   1.358 ns |   0.808 ns |
|   BenchRustSoftmax |         4 |         10 |     51.94 ns |   0.786 ns |   0.520 ns |
| BenchCSharpSoftmax |         4 |         15 |    161.88 ns |   1.270 ns |   0.840 ns |
|   BenchRustSoftmax |         4 |         15 |     54.06 ns |   0.409 ns |   0.271 ns |
| BenchCSharpSoftmax |         4 |         20 |    206.63 ns |   2.300 ns |   1.521 ns |
|   BenchRustSoftmax |         4 |         20 |     63.24 ns |   0.523 ns |   0.346 ns |
| BenchCSharpSoftmax |         4 |         30 |    295.56 ns |   2.220 ns |   1.468 ns |
|   BenchRustSoftmax |         4 |         30 |     76.30 ns |   0.509 ns |   0.337 ns |
| BenchCSharpSoftmax |         4 |         50 |    473.31 ns |   3.426 ns |   2.266 ns |
|   BenchRustSoftmax |         4 |         50 |    104.22 ns |   1.117 ns |   0.739 ns |
| BenchCSharpSoftmax |         4 |        100 |    923.95 ns |   2.021 ns |   1.057 ns |
|   BenchRustSoftmax |         4 |        100 |    168.24 ns |   1.028 ns |   0.680 ns |
| BenchCSharpSoftmax |         6 |          2 |     43.90 ns |   0.351 ns |   0.232 ns |
|   BenchRustSoftmax |         6 |          2 |     43.59 ns |   0.193 ns |   0.128 ns |
| BenchCSharpSoftmax |         6 |          3 |     52.93 ns |   0.497 ns |   0.328 ns |
|   BenchRustSoftmax |         6 |          3 |     44.51 ns |   0.371 ns |   0.246 ns |
| BenchCSharpSoftmax |         6 |          4 |     64.52 ns |   0.481 ns |   0.318 ns |
|   BenchRustSoftmax |         6 |          4 |     44.99 ns |   0.398 ns |   0.263 ns |
| BenchCSharpSoftmax |         6 |          6 |     85.86 ns |   0.559 ns |   0.332 ns |
|   BenchRustSoftmax |         6 |          6 |     46.89 ns |   0.421 ns |   0.278 ns |
| BenchCSharpSoftmax |         6 |          8 |    108.60 ns |   0.881 ns |   0.583 ns |
|   BenchRustSoftmax |         6 |          8 |     48.32 ns |   0.562 ns |   0.371 ns |
| BenchCSharpSoftmax |         6 |         10 |    137.20 ns |   0.959 ns |   0.571 ns |
|   BenchRustSoftmax |         6 |         10 |     54.70 ns |   0.652 ns |   0.432 ns |
| BenchCSharpSoftmax |         6 |         15 |    192.95 ns |   1.415 ns |   0.842 ns |
|   BenchRustSoftmax |         6 |         15 |     57.17 ns |   0.340 ns |   0.202 ns |
| BenchCSharpSoftmax |         6 |         20 |    248.35 ns |   1.232 ns |   0.733 ns |
|   BenchRustSoftmax |         6 |         20 |     66.63 ns |   0.505 ns |   0.334 ns |
| BenchCSharpSoftmax |         6 |         30 |    359.19 ns |   0.695 ns |   0.364 ns |
|   BenchRustSoftmax |         6 |         30 |     80.27 ns |   0.480 ns |   0.286 ns |
| BenchCSharpSoftmax |         6 |         50 |    581.49 ns |   1.515 ns |   0.792 ns |
|   BenchRustSoftmax |         6 |         50 |    110.89 ns |   0.316 ns |   0.165 ns |
| BenchCSharpSoftmax |         6 |        100 |  1,144.33 ns |  10.626 ns |   7.028 ns |
|   BenchRustSoftmax |         6 |        100 |    181.72 ns |   1.349 ns |   0.892 ns |
| BenchCSharpSoftmax |         8 |          2 |     46.22 ns |   0.365 ns |   0.241 ns |
|   BenchRustSoftmax |         8 |          2 |     45.70 ns |   0.353 ns |   0.184 ns |
| BenchCSharpSoftmax |         8 |          3 |     60.56 ns |   0.310 ns |   0.205 ns |
|   BenchRustSoftmax |         8 |          3 |     46.32 ns |   0.283 ns |   0.169 ns |
| BenchCSharpSoftmax |         8 |          4 |     72.21 ns |   1.025 ns |   0.678 ns |
|   BenchRustSoftmax |         8 |          4 |     46.80 ns |   0.407 ns |   0.269 ns |
| BenchCSharpSoftmax |         8 |          6 |     97.30 ns |   0.576 ns |   0.301 ns |
|   BenchRustSoftmax |         8 |          6 |     48.18 ns |   0.245 ns |   0.162 ns |
| BenchCSharpSoftmax |         8 |          8 |    131.55 ns |   0.786 ns |   0.468 ns |
|   BenchRustSoftmax |         8 |          8 |     49.68 ns |   0.324 ns |   0.215 ns |
| BenchCSharpSoftmax |         8 |         10 |    158.65 ns |   2.593 ns |   1.543 ns |
|   BenchRustSoftmax |         8 |         10 |     56.57 ns |   0.382 ns |   0.228 ns |
| BenchCSharpSoftmax |         8 |         15 |    222.57 ns |   3.559 ns |   2.354 ns |
|   BenchRustSoftmax |         8 |         15 |     58.94 ns |   0.176 ns |   0.092 ns |
| BenchCSharpSoftmax |         8 |         20 |    287.67 ns |   1.634 ns |   1.081 ns |
|   BenchRustSoftmax |         8 |         20 |     69.84 ns |   0.302 ns |   0.158 ns |
| BenchCSharpSoftmax |         8 |         30 |    417.26 ns |   3.551 ns |   2.113 ns |
|   BenchRustSoftmax |         8 |         30 |     83.30 ns |   0.378 ns |   0.225 ns |
| BenchCSharpSoftmax |         8 |         50 |    674.99 ns |   2.419 ns |   1.265 ns |
|   BenchRustSoftmax |         8 |         50 |    117.13 ns |   1.002 ns |   0.663 ns |
| BenchCSharpSoftmax |         8 |        100 |  1,328.05 ns |  16.941 ns |  10.081 ns |
|   BenchRustSoftmax |         8 |        100 |    191.76 ns |   1.370 ns |   0.906 ns |
| BenchCSharpSoftmax |        10 |          2 |     50.25 ns |   0.190 ns |   0.113 ns |
|   BenchRustSoftmax |        10 |          2 |     46.95 ns |   0.543 ns |   0.323 ns |
| BenchCSharpSoftmax |        10 |          3 |     64.83 ns |   0.653 ns |   0.432 ns |
|   BenchRustSoftmax |        10 |          3 |     47.63 ns |   0.315 ns |   0.208 ns |
| BenchCSharpSoftmax |        10 |          4 |     79.28 ns |   0.728 ns |   0.481 ns |
|   BenchRustSoftmax |        10 |          4 |     48.48 ns |   0.349 ns |   0.231 ns |
| BenchCSharpSoftmax |        10 |          6 |    108.35 ns |   0.902 ns |   0.537 ns |
|   BenchRustSoftmax |        10 |          6 |     50.09 ns |   0.665 ns |   0.440 ns |
| BenchCSharpSoftmax |        10 |          8 |    147.07 ns |   1.402 ns |   0.834 ns |
|   BenchRustSoftmax |        10 |          8 |     51.26 ns |   0.430 ns |   0.284 ns |
| BenchCSharpSoftmax |        10 |         10 |    177.78 ns |   5.162 ns |   3.414 ns |
|   BenchRustSoftmax |        10 |         10 |     59.15 ns |   0.773 ns |   0.512 ns |
| BenchCSharpSoftmax |        10 |         15 |    246.70 ns |   5.313 ns |   3.162 ns |
|   BenchRustSoftmax |        10 |         15 |     60.91 ns |   0.364 ns |   0.217 ns |
| BenchCSharpSoftmax |        10 |         20 |    320.65 ns |   3.318 ns |   2.194 ns |
|   BenchRustSoftmax |        10 |         20 |     72.27 ns |   0.421 ns |   0.278 ns |
| BenchCSharpSoftmax |        10 |         30 |    465.94 ns |   4.178 ns |   2.185 ns |
|   BenchRustSoftmax |        10 |         30 |     87.10 ns |   0.269 ns |   0.141 ns |
| BenchCSharpSoftmax |        10 |         50 |    755.29 ns |   5.483 ns |   3.626 ns |
|   BenchRustSoftmax |        10 |         50 |    122.70 ns |   0.854 ns |   0.508 ns |
| BenchCSharpSoftmax |        10 |        100 |  1,487.73 ns |  13.129 ns |   7.813 ns |
|   BenchRustSoftmax |        10 |        100 |    199.72 ns |   1.101 ns |   0.655 ns |
| BenchCSharpSoftmax |        15 |          2 |     58.99 ns |   0.393 ns |   0.234 ns |
|   BenchRustSoftmax |        15 |          2 |     49.53 ns |   0.371 ns |   0.245 ns |
| BenchCSharpSoftmax |        15 |          3 |     80.03 ns |   0.419 ns |   0.249 ns |
|   BenchRustSoftmax |        15 |          3 |     50.23 ns |   0.431 ns |   0.285 ns |
| BenchCSharpSoftmax |        15 |          4 |    101.55 ns |   0.319 ns |   0.190 ns |
|   BenchRustSoftmax |        15 |          4 |     50.58 ns |   0.378 ns |   0.225 ns |
| BenchCSharpSoftmax |        15 |          6 |    151.31 ns |   0.523 ns |   0.273 ns |
|   BenchRustSoftmax |        15 |          6 |     52.54 ns |   0.272 ns |   0.180 ns |
| BenchCSharpSoftmax |        15 |          8 |    194.53 ns |   0.619 ns |   0.324 ns |
|   BenchRustSoftmax |        15 |          8 |     53.40 ns |   0.301 ns |   0.179 ns |
| BenchCSharpSoftmax |        15 |         10 |    237.89 ns |   1.455 ns |   0.761 ns |
|   BenchRustSoftmax |        15 |         10 |     64.66 ns |   0.577 ns |   0.344 ns |
| BenchCSharpSoftmax |        15 |         15 |    344.77 ns |   2.060 ns |   1.363 ns |
|   BenchRustSoftmax |        15 |         15 |     66.71 ns |   0.274 ns |   0.163 ns |
| BenchCSharpSoftmax |        15 |         20 |    450.07 ns |   2.360 ns |   1.561 ns |
|   BenchRustSoftmax |        15 |         20 |     79.83 ns |   0.675 ns |   0.402 ns |
| BenchCSharpSoftmax |        15 |         30 |    664.12 ns |   4.028 ns |   2.397 ns |
|   BenchRustSoftmax |        15 |         30 |     96.39 ns |   0.771 ns |   0.510 ns |
| BenchCSharpSoftmax |        15 |         50 |  1,089.79 ns |   3.638 ns |   1.903 ns |
|   BenchRustSoftmax |        15 |         50 |    137.68 ns |   0.166 ns |   0.087 ns |
| BenchCSharpSoftmax |        15 |        100 |  2,159.30 ns |  10.192 ns |   6.742 ns |
|   BenchRustSoftmax |        15 |        100 |    228.58 ns |   1.333 ns |   0.882 ns |
| BenchCSharpSoftmax |        20 |          2 |     69.92 ns |   0.316 ns |   0.209 ns |
|   BenchRustSoftmax |        20 |          2 |     52.85 ns |   0.762 ns |   0.504 ns |
| BenchCSharpSoftmax |        20 |          3 |     95.70 ns |   0.586 ns |   0.388 ns |
|   BenchRustSoftmax |        20 |          3 |     53.39 ns |   0.395 ns |   0.262 ns |
| BenchCSharpSoftmax |        20 |          4 |    121.64 ns |   1.321 ns |   0.874 ns |
|   BenchRustSoftmax |        20 |          4 |     53.58 ns |   0.405 ns |   0.241 ns |
| BenchCSharpSoftmax |        20 |          6 |    180.81 ns |   1.353 ns |   0.805 ns |
|   BenchRustSoftmax |        20 |          6 |     54.90 ns |   0.541 ns |   0.322 ns |
| BenchCSharpSoftmax |        20 |          8 |    232.39 ns |   1.477 ns |   0.879 ns |
|   BenchRustSoftmax |        20 |          8 |     56.66 ns |   0.272 ns |   0.180 ns |
| BenchCSharpSoftmax |        20 |         10 |    283.86 ns |   1.754 ns |   1.160 ns |
|   BenchRustSoftmax |        20 |         10 |     81.11 ns |   0.335 ns |   0.175 ns |
| BenchCSharpSoftmax |        20 |         15 |    414.21 ns |   2.758 ns |   1.641 ns |
|   BenchRustSoftmax |        20 |         15 |     73.17 ns |   0.571 ns |   0.378 ns |
| BenchCSharpSoftmax |        20 |         20 |    543.33 ns |   3.291 ns |   1.721 ns |
|   BenchRustSoftmax |        20 |         20 |     87.75 ns |   0.612 ns |   0.405 ns |
| BenchCSharpSoftmax |        20 |         30 |    802.19 ns |   4.855 ns |   2.889 ns |
|   BenchRustSoftmax |        20 |         30 |    106.20 ns |   0.653 ns |   0.432 ns |
| BenchCSharpSoftmax |        20 |         50 |  1,330.62 ns |  15.057 ns |   9.959 ns |
|   BenchRustSoftmax |        20 |         50 |    154.25 ns |   0.951 ns |   0.629 ns |
| BenchCSharpSoftmax |        20 |        100 |  2,630.60 ns |  36.283 ns |  21.591 ns |
|   BenchRustSoftmax |        20 |        100 |    256.61 ns |   0.472 ns |   0.247 ns |
| BenchCSharpSoftmax |        30 |          2 |     89.06 ns |   0.525 ns |   0.347 ns |
|   BenchRustSoftmax |        30 |          2 |     59.28 ns |   0.449 ns |   0.297 ns |
| BenchCSharpSoftmax |        30 |          3 |    124.37 ns |   0.820 ns |   0.542 ns |
|   BenchRustSoftmax |        30 |          3 |     60.10 ns |   0.222 ns |   0.116 ns |
| BenchCSharpSoftmax |        30 |          4 |    162.22 ns |   1.472 ns |   0.876 ns |
|   BenchRustSoftmax |        30 |          4 |     60.78 ns |   0.413 ns |   0.273 ns |
| BenchCSharpSoftmax |        30 |          6 |    235.68 ns |   0.432 ns |   0.226 ns |
|   BenchRustSoftmax |        30 |          6 |     64.22 ns |   0.531 ns |   0.351 ns |
| BenchCSharpSoftmax |        30 |          8 |    305.39 ns |   1.452 ns |   0.961 ns |
|   BenchRustSoftmax |        30 |          8 |     64.18 ns |   0.907 ns |   0.540 ns |
| BenchCSharpSoftmax |        30 |         10 |    375.32 ns |   3.685 ns |   2.437 ns |
|   BenchRustSoftmax |        30 |         10 |     83.44 ns |   0.421 ns |   0.278 ns |
| BenchCSharpSoftmax |        30 |         15 |    551.00 ns |   2.809 ns |   1.858 ns |
|   BenchRustSoftmax |        30 |         15 |     87.44 ns |   0.541 ns |   0.358 ns |
| BenchCSharpSoftmax |        30 |         20 |    724.51 ns |   2.765 ns |   1.446 ns |
|   BenchRustSoftmax |        30 |         20 |    108.70 ns |   0.652 ns |   0.432 ns |
| BenchCSharpSoftmax |        30 |         30 |  1,071.75 ns |   1.975 ns |   1.033 ns |
|   BenchRustSoftmax |        30 |         30 |    133.70 ns |   0.436 ns |   0.228 ns |
| BenchCSharpSoftmax |        30 |         50 |  1,773.07 ns |   8.264 ns |   4.322 ns |
|   BenchRustSoftmax |        30 |         50 |    203.94 ns |   1.317 ns |   0.871 ns |
| BenchCSharpSoftmax |        30 |        100 |  3,531.38 ns |  13.472 ns |   7.046 ns |
|   BenchRustSoftmax |        30 |        100 |    352.42 ns |   1.661 ns |   1.099 ns |
| BenchCSharpSoftmax |        50 |          2 |    125.77 ns |   0.762 ns |   0.453 ns |
|   BenchRustSoftmax |        50 |          2 |     72.69 ns |   0.690 ns |   0.456 ns |
| BenchCSharpSoftmax |        50 |          3 |    182.63 ns |   1.418 ns |   0.844 ns |
|   BenchRustSoftmax |        50 |          3 |     73.70 ns |   0.531 ns |   0.351 ns |
| BenchCSharpSoftmax |        50 |          4 |    238.67 ns |   0.736 ns |   0.487 ns |
|   BenchRustSoftmax |        50 |          4 |     74.36 ns |   0.529 ns |   0.315 ns |
| BenchCSharpSoftmax |        50 |          6 |    347.18 ns |   2.737 ns |   1.811 ns |
|   BenchRustSoftmax |        50 |          6 |     75.92 ns |   0.649 ns |   0.429 ns |
| BenchCSharpSoftmax |        50 |          8 |    452.58 ns |   2.715 ns |   1.615 ns |
|   BenchRustSoftmax |        50 |          8 |     77.59 ns |   0.658 ns |   0.435 ns |
| BenchCSharpSoftmax |        50 |         10 |    560.70 ns |   4.131 ns |   2.459 ns |
|   BenchRustSoftmax |        50 |         10 |    114.90 ns |   1.337 ns |   0.884 ns |
| BenchCSharpSoftmax |        50 |         15 |    829.03 ns |   5.391 ns |   3.566 ns |
|   BenchRustSoftmax |        50 |         15 |    118.65 ns |   1.175 ns |   0.777 ns |
| BenchCSharpSoftmax |        50 |         20 |  1,096.20 ns |   6.049 ns |   3.600 ns |
|   BenchRustSoftmax |        50 |         20 |    157.71 ns |   0.730 ns |   0.382 ns |
| BenchCSharpSoftmax |        50 |         30 |  1,626.88 ns |  10.096 ns |   5.280 ns |
|   BenchRustSoftmax |        50 |         30 |    201.69 ns |   1.351 ns |   0.893 ns |
| BenchCSharpSoftmax |        50 |         50 |  2,711.00 ns |  20.992 ns |  12.492 ns |
|   BenchRustSoftmax |        50 |         50 |    327.84 ns |   1.556 ns |   1.029 ns |
| BenchCSharpSoftmax |        50 |        100 |  5,389.39 ns |  33.112 ns |  21.901 ns |
|   BenchRustSoftmax |        50 |        100 |    607.38 ns |   3.247 ns |   2.148 ns |
| BenchCSharpSoftmax |       100 |          2 |    241.65 ns |   3.934 ns |   2.602 ns |
|   BenchRustSoftmax |       100 |          2 |    108.10 ns |   1.142 ns |   0.755 ns |
| BenchCSharpSoftmax |       100 |          3 |    345.58 ns |   3.018 ns |   1.996 ns |
|   BenchRustSoftmax |       100 |          3 |    108.14 ns |   0.567 ns |   0.338 ns |
| BenchCSharpSoftmax |       100 |          4 |    451.14 ns |   7.788 ns |   4.635 ns |
|   BenchRustSoftmax |       100 |          4 |    108.88 ns |   0.538 ns |   0.356 ns |
| BenchCSharpSoftmax |       100 |          6 |    658.11 ns |   8.127 ns |   5.376 ns |
|   BenchRustSoftmax |       100 |          6 |    110.32 ns |   0.574 ns |   0.380 ns |
| BenchCSharpSoftmax |       100 |          8 |    869.04 ns |   9.037 ns |   5.978 ns |
|   BenchRustSoftmax |       100 |          8 |    111.87 ns |   1.587 ns |   0.944 ns |
| BenchCSharpSoftmax |       100 |         10 |  1,089.58 ns |  10.411 ns |   6.886 ns |
|   BenchRustSoftmax |       100 |         10 |    183.73 ns |   0.899 ns |   0.595 ns |
| BenchCSharpSoftmax |       100 |         15 |  1,613.98 ns |  14.096 ns |   8.388 ns |
|   BenchRustSoftmax |       100 |         15 |    187.17 ns |   1.072 ns |   0.709 ns |
| BenchCSharpSoftmax |       100 |         20 |  2,138.29 ns |  27.733 ns |  18.344 ns |
|   BenchRustSoftmax |       100 |         20 |    260.53 ns |   1.670 ns |   1.105 ns |
| BenchCSharpSoftmax |       100 |         30 |  3,202.58 ns |  40.027 ns |  26.476 ns |
|   BenchRustSoftmax |       100 |         30 |    339.24 ns |   1.903 ns |   1.259 ns |
| BenchCSharpSoftmax |       100 |         50 |  5,308.62 ns |  55.470 ns |  36.690 ns |
|   BenchRustSoftmax |       100 |         50 |    597.26 ns |   2.631 ns |   1.740 ns |
| BenchCSharpSoftmax |       100 |        100 | 10,616.00 ns | 168.861 ns | 111.691 ns |
|   BenchRustSoftmax |       100 |        100 |  1,029.09 ns |   4.749 ns |   2.826 ns |


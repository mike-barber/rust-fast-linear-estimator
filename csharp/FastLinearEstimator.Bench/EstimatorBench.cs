using BenchmarkDotNet.Attributes;
using System;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Threading.Tasks;
using Xunit;

namespace FastLinearEstimator.Bench
{
    public class EstimatorBench
    {
        private const int _numInputSets = 250;
        readonly private Random _rng = new Random();

        public int NumInputs = 15;
        public int NumOutputs = 75;

        private float[][] _inputSets;
        private float[,] _coeff;
        private float[] _intercepts;

        LinearEstimator _linearEstimator = null;

        [GlobalSetup]
        public void GlobalSetup()
        {
            // setup input sets
            _inputSets = Enumerable.Range(0, 250).Select(_ =>
            {
                var inp = Enumerable.Range(0, NumInputs).Select(_ => (float)_rng.NextDouble()).ToArray();
                return inp;
            }).ToArray();

            // setup coefficients 
            _coeff = new float[NumInputs, NumOutputs];
            _intercepts = new float[NumOutputs];
            for (var ip = 0; ip < NumInputs; ++ip)
            {
                for (var op = 0; op < NumOutputs; ++op)
                {
                    var v = (float)_rng.NextDouble();
                    _coeff[ip, op] = v; // column-major (normal)
                }
            }

            for (var op = 0; op < NumOutputs; ++op)
            {
                _intercepts[op] = (float)_rng.NextDouble();
            }
            _linearEstimator = new LinearEstimator(_coeff, _intercepts);
        }

        [GlobalCleanup]
        public void GlobalCleanup()
        {
            _linearEstimator?.Dispose();
        }

        public static void SelfTest()
        {
            var eb = new EstimatorBench();
            eb.GlobalSetup();

            var tolerance = new AlmostEqualFloat(1e-6f);
            var expected = eb.TestCSharpProduct();

            static void Check(string name, float[] expected, float[] results, AlmostEqualFloat tolerance)
            {
                Helpers.PrintVector(name, results);
                Assert.Equal(expected, results, tolerance);
            }

            Check("product c#  ", expected, eb.TestCSharpProduct(), tolerance);
            Check("product rust", expected, eb.TestRustProduct(), tolerance);

            // Check softmax
            var expectedSoftmax = eb.TestCSharpSoftmax();

            // loose tolerance for approximated softmax
            var toleranceSoftmaxAccurate = new AlmostEqualFloat(1e-6f);
            var toleranceSoftmaxApprox = new AlmostEqualFloat(1e-4f);
            Check("softmax exact c#            ", expectedSoftmax, eb.TestCSharpSoftmax(), toleranceSoftmaxAccurate);
            Check("softmax rust not norm sleef ", expectedSoftmax, eb.TestRustSoftmaxNotNormalisedSleef(), toleranceSoftmaxAccurate);
            Check("softmax rust not norm approx", expectedSoftmax, eb.TestRustSoftmaxNotNormalisedApprox(), toleranceSoftmaxApprox);
            Check("softmax rust approx         ", expectedSoftmax, eb.TestRustSoftmax(), toleranceSoftmaxApprox);

            // test bench calls -- these will throw if there's anything amiss
            // (a lot easier to diagnose before the benchmark run starts)
            eb.BenchCSharpProduct();
            eb.BenchRustProduct();

            eb.BenchCSharpSoftmax();
            eb.BenchRustSoftmax();
        }

        public float[] TestCSharpProduct()
        {
            var x = _inputSets[0];
            var y = new float[NumOutputs];
            CSharpProduct(x, y);
            return y;
        }

        public float[] TestRustProduct()
        {
            var x = _inputSets[0];
            var y = new float[NumOutputs];
            RustProduct(x, y);
            return y;
        }

        public float[] TestCSharpSoftmax()
        {
            var x = _inputSets[0];
            var y = new float[NumOutputs];
            CSharpSoftmax(x, y);
            return y;
        }

        public float[] TestRustSoftmax()
        {
            var x = _inputSets[0];
            var y = new float[NumOutputs];
            RustSoftmax(x, y);
            return y;
        }

        private void Cumulative(Span<float> values)
        {
            float acc = 0;
            for (var i = 0; i < values.Length; ++i)
            {
                var value = values[i];
                acc += value;
                values[i] = acc;
            }
        }

        public float[] TestRustSoftmaxNotNormalisedApprox()
        {
            var x = _inputSets[0];
            var y = new float[NumOutputs];
            RustSoftmaxNotNormalisedApprox(x, y, out var sum);
            // for consistency in testing
            Cumulative(y);
            return y;
        }

        public float[] TestRustSoftmaxNotNormalisedSleef()
        {
            var x = _inputSets[0];
            var y = new float[NumOutputs];
            RustSoftmaxNotNormalisedSleef(x, y, out var sum);
            // for consistency in testing
            Cumulative(y);
            return y;
        }

        // simple loop
        public void CSharpProduct(ReadOnlySpan<float> x, Span<float> y)
        {
            for (int r = 0; r < NumOutputs; ++r)
            {
                var acc = _intercepts[r];
                for (int c = 0; c < NumInputs; ++c)
                {
                    acc += _coeff[c, r] * x[c];
                }
                y[r] = acc;
            }
        }

        // using Rust estimator
        public void RustProduct(ReadOnlySpan<float> x, Span<float> y)
        {
            _linearEstimator.EstimateLinear(x, y);
        }

        // simple loop
        public void CSharpSoftmax(ReadOnlySpan<float> x, Span<float> y)
        {
            var cumulativeSum = 0.0f;
            for (int r = 0; r < NumOutputs; ++r)
            {
                var acc = _intercepts[r];
                for (int c = 0; c < NumInputs; ++c)
                {
                    acc += _coeff[c, r] * x[c];
                }
                var exp = MathF.Exp(acc);
                cumulativeSum += exp;
                y[r] = cumulativeSum;
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public void RustSoftmax(ReadOnlySpan<float> x, Span<float> y)
        {
            _linearEstimator.EstimateSoftMaxCumulative(x, y);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public void RustSoftmaxNotNormalisedSleef(ReadOnlySpan<float> x, Span<float> y, out float sum)
        {
            _linearEstimator.EstimateSoftMaxNotNormalisedSleef(x, y, out sum);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public void RustSoftmaxNotNormalisedApprox(ReadOnlySpan<float> x, Span<float> y, out float sum)
        {
            _linearEstimator.EstimateSoftMaxNotNormalisedApprox(x, y, out sum);
        }

        private float[] GetRandomInputFloat() => GetRandomInputFloat(_rng);
        private float[] GetRandomInputFloat(Random rng) => _inputSets[rng.Next(NumInputs)];

        [Benchmark]
        public float BenchRustProduct()
        {
            var x = GetRandomInputFloat();
            Span<float> y = stackalloc float[NumOutputs];
            RustProduct(x, y);
            return y[0];
        }

        [Benchmark]
        public float BenchCSharpProduct()
        {
            var x = GetRandomInputFloat();
            Span<float> y = stackalloc float[NumOutputs];
            CSharpProduct(x, y);
            return y[0];
        }

        [Benchmark]
        public float BenchRustSoftmax()
        {
            var x = GetRandomInputFloat();
            Span<float> y = stackalloc float[NumOutputs];
            RustSoftmax(x, y);
            return y[0];
        }

        [Benchmark]
        public float BenchRustSoftmaxNotNormalisedApprox()
        {
            var x = GetRandomInputFloat();
            Span<float> y = stackalloc float[NumOutputs];
            RustSoftmaxNotNormalisedApprox(x, y, out var sum);
            return y[0];
        }

        [Benchmark]
        public float BenchRustSoftmaxNotNormalisedSleef()
        {
            var x = GetRandomInputFloat();
            Span<float> y = stackalloc float[NumOutputs];
            RustSoftmaxNotNormalisedSleef(x, y, out var sum);
            return y[0];
        }

        [Benchmark]
        public float BenchCSharpSoftmax()
        {
            var x = GetRandomInputFloat();
            Span<float> y = stackalloc float[NumOutputs];
            CSharpSoftmax(x, y);
            return y[0];
        }

        // simulate a large inference run (ignoring data copy etc)
        const int ITERATIONS = 1000;
        const int PER_ITERATION = 5000;
        public delegate void RunFunction(ReadOnlySpan<float> inputs, Span<float> outputs);
        public float LargeParallel(RunFunction func)
        {
            Parallel.For(0, ITERATIONS,
                () => new Random(),  // local state init
                (i, state, rng) =>
                {
                    var acc = 0f;
                    Span<float> output = stackalloc float[NumOutputs];
                    for (var iter = 0; iter < PER_ITERATION; ++iter)
                    {
                        var input = GetRandomInputFloat(rng);
                        func(input, output);
                        acc += output[0];
                    }
                    return rng;
                },
            rng =>
            {
                // do nothing with it
            }
            );
            return 0;
        }

        [Benchmark]
        public float LargeParallelCSharpSoftmax() => LargeParallel(CSharpSoftmax);

        [Benchmark]
        public float LargeParallelRustSoftmax() => LargeParallel(RustSoftmax);
        [Benchmark]
        public float LargeParallelRustSoftmaxNotNormalisedSleef() => LargeParallel((x, y) => RustSoftmaxNotNormalisedSleef(x, y, out var _));
        [Benchmark]
        public float LargeParallelRustSoftmaxNotNormalisedApprox() => LargeParallel((x, y) => RustSoftmaxNotNormalisedApprox(x, y, out var _));

    }
}

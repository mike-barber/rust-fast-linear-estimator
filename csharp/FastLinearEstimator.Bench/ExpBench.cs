using System;
using System.Linq;
using System.Numerics;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using BenchmarkDotNet.Attributes;

namespace FastLinearEstimator.Bench
{

    public class ExpBench
    {
        private const int _len = 8;
        private const int _reps = 100;
        private const int _numInputSets = 250;
        readonly private Random _rng = new Random();

        private float[][] _inputSets;
        private int _nextInput = 0;


        [GlobalSetup]
        public void GlobalSetup()
        {
            // setup input sets
            _inputSets = Enumerable.Range(0, 250).Select(_ =>
            {
                var inp = Enumerable.Range(0, _len).Select(_ => (float)_rng.NextDouble() * 10).ToArray();
                return inp;
            }).ToArray();
        }

        private float[] GetInput()
        {
            _nextInput = (_nextInput + 1) % _numInputSets;
            return _inputSets[_nextInput];
        }

        // Amortised cost of selecting input
        [Benchmark(OperationsPerInvoke = _len)]
        public float BaselineSelect()
        {
            return GetInput()[0];
        }

        [Benchmark(OperationsPerInvoke = _len)]
        public float ExpAccurateScalar()
        {
            var input = GetInput();
            Span<float> val = input.AsSpan(0, _len);
            Span<float> res = stackalloc float[input.Length];

            for (var r = 0; r < _reps; ++r)
            {
                res[0] = MathF.Exp(input[0]);
                res[1] = MathF.Exp(input[1]);
                res[2] = MathF.Exp(input[2]);
                res[3] = MathF.Exp(input[3]);
                res[4] = MathF.Exp(input[4]);
                res[5] = MathF.Exp(input[5]);
                res[6] = MathF.Exp(input[6]);
                res[7] = MathF.Exp(input[7]);
            }
            return res[0];
        }

        [Benchmark(OperationsPerInvoke = _len)]
        public float ExpApproxScalar()
        {
            var input = GetInput();
            Span<float> val = input.AsSpan(0, _len);
            Span<float> res = stackalloc float[input.Length];

            for (var r = 0; r < _reps; ++r)
            {
                res[0] = ExpApprox.ExpApproxScalar(input[0]);
                res[1] = ExpApprox.ExpApproxScalar(input[1]);
                res[2] = ExpApprox.ExpApproxScalar(input[2]);
                res[3] = ExpApprox.ExpApproxScalar(input[3]);
                res[4] = ExpApprox.ExpApproxScalar(input[4]);
                res[5] = ExpApprox.ExpApproxScalar(input[5]);
                res[6] = ExpApprox.ExpApproxScalar(input[6]);
                res[7] = ExpApprox.ExpApproxScalar(input[7]);
            }
            return res[0];
        }

        [Benchmark(OperationsPerInvoke = _len)]
        public float ExpAvx()
        {
            var input = GetInput();
            unsafe
            {
                var res = Vector256<float>.Zero;
                for (var r = 0; r < _reps; ++r)
                {
                    fixed (float* p = input)
                    {

                        var vec = Avx.LoadVector256(p);
                        res = ExpApprox.ExpApproxAvx(vec);
                    }
                }
                return res.GetElement(0);
            }
        }

        [Benchmark(OperationsPerInvoke = _len)]
        public float ExpVector()
        {
            var input = GetInput();
            var res = Vector<float>.Zero;
            for (var r = 0; r < _reps; ++r)
            {
                var vec = new Vector<float>(input);
                res = ExpApprox.ExpApproxVector(vec);
            }
            return res[0];
        }

        // not very efficient for short vectors
        [Benchmark(OperationsPerInvoke = _len)]
        public float ExpRustInterop()
        {
            var input = GetInput();
            Span<float> res = stackalloc float[input.Length];
            for (var r = 0; r < _reps; ++r)
            {
                input.CopyTo(res);
                RustSafe.ExpApproxInPlace(res);
            }
            return res[0];
        }
    }

}
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
        private const int _numInputSets = 250;
        readonly private Random _rng = new Random();

        private float[][] _inputSets;


        [GlobalSetup]
        public void GlobalSetup()
        {
            // setup input sets
            _inputSets = Enumerable.Range(0, 250).Select(_ =>
            {
                var inp = Enumerable.Range(0, 8).Select(_ => (float)_rng.NextDouble() * 10).ToArray();
                return inp;
            }).ToArray();
        }

        private float[] GetRandomInputFloat() => _inputSets[_rng.Next(_numInputSets)];

        [Benchmark]
        public float BaselineSelect()
        {
            return GetRandomInputFloat()[0];
        }

        [Benchmark]
        public float ExpScalar()
        {
            var input = GetRandomInputFloat();
            Span<float> res = stackalloc float[8];
            for (var i = 0; i < input.Length; ++i)
            {
                res[i] = MathF.Exp(input[i]);
            }
            return res[0];
        }

        [Benchmark]
        public float ExpAvx()
        {
            var input = GetRandomInputFloat();
            unsafe
            {
                fixed (float* p = input)
                {
                    var vec = Avx.LoadVector256(p);
                    var res = ExpApprox.ExpApproxAvx(vec);
                    return res.GetElement(0);
                }
            }
        }

        [Benchmark]
        public float ExpVector()
        {
            var input = GetRandomInputFloat();
            var vec = new Vector<float>(input);
            var res = ExpApprox.ExpApproxVector(vec);
            return res[0];
        }
    }

}
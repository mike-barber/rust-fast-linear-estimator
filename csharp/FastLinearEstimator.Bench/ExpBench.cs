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
        private const int _vectorWords = 8;
        private const int _arrayLen = 100; // some arbitrary length
        private const int _numInputSets = 250;

        readonly private Random _rng = new Random();

        private float[][] _inputSetsWords;
        private float[][] _inputSetArrays;
        private float[] _resultScratch;

        private int _nextInputWord = 0;
        private int _nextInputArray = 0;


        public static void SelfTest()
        {
            var bench = new ExpBench();
            bench.GlobalSetup();
            bench.ExpAccurateScalar1();
            bench.ExpAccurateScalar8();
            bench.ExpApproxScalar8();
            bench.ExpAvx();
            bench.ExpVector();
            bench.ExpRustVector();
            bench.ExpApproxArray();
            bench.ExpRustInteropArrayInPlace();
            bench.ExpRustInteropArray();
        }


        [GlobalSetup]
        public void GlobalSetup()
        {
            // setup input sets
            _inputSetsWords = Enumerable.Range(0, _numInputSets).Select(_ =>
            {
                var inp = Enumerable.Range(0, _vectorWords).Select(_ => (float)_rng.NextDouble() * 10).ToArray();
                return inp;
            }).ToArray();

            _inputSetArrays = Enumerable.Range(0, _numInputSets).Select(_ =>
            {
                var inp = Enumerable.Range(0, _arrayLen).Select(_ => (float)_rng.NextDouble() * 10).ToArray();
                return inp;
            }).ToArray();

            _resultScratch = new float[_arrayLen];
        }

        private float[] GetInputWord()
        {
            _nextInputWord = (_nextInputWord + 1) % _numInputSets;
            return _inputSetsWords[_nextInputWord];
        }

        private float[] GetInputArray()
        {
            _nextInputArray = (_nextInputArray + 1) % _numInputSets;
            return _inputSetArrays[_nextInputArray];
        }

        // cost of selecting input
        [Benchmark]
        public float BaselineSelect()
        {
            return GetInputWord()[0];
        }

        [Benchmark]
        public float ExpAccurateScalar1()
        {
            var input = GetInputWord();
            return MathF.Exp(input[0]);
        }

        [Benchmark]
        public float ExpAccurateScalar8()
        {
            var input = GetInputWord();
            Span<float> val = input.AsSpan(0, _vectorWords);
            Span<float> res = _resultScratch.AsSpan(0, _vectorWords);

            res[0] = MathF.Exp(input[0]);
            res[1] = MathF.Exp(input[1]);
            res[2] = MathF.Exp(input[2]);
            res[3] = MathF.Exp(input[3]);
            res[4] = MathF.Exp(input[4]);
            res[5] = MathF.Exp(input[5]);
            res[6] = MathF.Exp(input[6]);
            res[7] = MathF.Exp(input[7]);
            return res[0];
        }

        [Benchmark]
        public float ExpApproxScalar8()
        {
            var input = GetInputWord();
            Span<float> val = input.AsSpan(0, _vectorWords);
            Span<float> res = _resultScratch.AsSpan(0, _vectorWords);

            res[0] = ExpApprox.ExpApproxScalar(input[0]);
            res[1] = ExpApprox.ExpApproxScalar(input[1]);
            res[2] = ExpApprox.ExpApproxScalar(input[2]);
            res[3] = ExpApprox.ExpApproxScalar(input[3]);
            res[4] = ExpApprox.ExpApproxScalar(input[4]);
            res[5] = ExpApprox.ExpApproxScalar(input[5]);
            res[6] = ExpApprox.ExpApproxScalar(input[6]);
            res[7] = ExpApprox.ExpApproxScalar(input[7]);
            return res[0];
        }

        [Benchmark]
        public float ExpAvx()
        {
            var input = GetInputWord();
            unsafe
            {
                fixed (float* p = input, py = _resultScratch)
                {
                    var vec = Avx.LoadVector256(p);
                    var y = ExpApprox.ExpApproxAvx(vec);
                    Avx.Store(py,y);
                }
                return _resultScratch[0];
            }
        }

        [Benchmark]
        public float ExpVector()
        {
            var input = GetInputWord();
            var vec = new Vector<float>(input);
            var y = ExpApprox.ExpApproxVector(vec);
            y.CopyTo(_resultScratch);
            return _resultScratch[0];
        }

        // not very efficient for short vectors
        [Benchmark]
        public float ExpRustVectorInPlace()
        {
            var input = GetInputWord();
            Span<float> res = _resultScratch.AsSpan(0, _vectorWords);
            input.CopyTo(res);
            RustSafe.ExpApproxInPlace(res);
            return res[0];
        }

        // not very efficient for short vectors
        [Benchmark]
        public float ExpRustVector()
        {
            var input = GetInputWord();
            Span<float> res = _resultScratch.AsSpan(0, _vectorWords);
            RustSafe.ExpApprox(input, res);
            return res[0];
        }

        [Benchmark]
        public float ExpApproxArray()
        {
            var input = GetInputArray();
            Span<float> inp = input.AsSpan();
            Span<float> res = _resultScratch.AsSpan(0, _arrayLen);

            // whole words - round to nearest 8
            var chunkEnd = inp.Length >> 3 << 3;
            var i = 0;
            while (i < chunkEnd)
            {
                var x = new Vector<float>(inp.Slice(i, 8));

                var y = ExpApprox.ExpApproxVector(x);

                y.CopyTo(res.Slice(i, 8));
                i += 8;
            }

            // remainder
            if (i < inp.Length)
            {
                var lenRemaining = inp.Length - i;

                Span<float> temp = stackalloc float[8];
                inp.Slice(i, lenRemaining).CopyTo(temp);
                var x = new Vector<float>(temp);

                var y = ExpApprox.ExpApproxVector(x);

                y.CopyTo(temp);
                temp.Slice(0, lenRemaining).CopyTo(res.Slice(i, lenRemaining));
            }

            return res[0];
        }

        [Benchmark]
        public float ExpRustInteropArrayInPlace()
        {
            var input = GetInputArray();
            Span<float> res = _resultScratch.AsSpan(0, _arrayLen);
            input.CopyTo(res);
            RustSafe.ExpApproxInPlace(res);
            return res[0];
        }

        [Benchmark]
        public float ExpRustInteropArray()
        {
            var input = GetInputArray();
            Span<float> res = _resultScratch.AsSpan(0, _arrayLen);
            RustSafe.ExpApprox(input, res);
            return res[0];
        }
    }

}
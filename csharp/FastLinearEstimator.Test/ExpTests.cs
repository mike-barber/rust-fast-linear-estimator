using System;
using System.Linq;
using System.Numerics;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using Xunit;

namespace FastLinearEstimator.Test
{
    public class ExpTests
    {
        float[] _inputs = new float[] { -10, -5, -1, 0, 1, 2, 5, 10 };

        float _tolerance = 1e-4f;

        private float[] Expected()
        {
            return _inputs.Select(v => MathF.Exp(v)).ToArray();
        }

        [Fact]
        public void ExpApproxScalar()
        {
            var expected = Expected();
            var resArr = _inputs.Select(ExpApprox.ExpApproxScalar).ToArray();

            for (var i = 0; i < 8; ++i)
            {
                AssertAlmostEqual(expected[i], resArr[i], _tolerance);
            }
        }

        [Fact]
        public void ExpApproxRust() 
        {
            var expected = Expected();
            
            var vals = new float[8];
            _inputs.CopyTo(vals.AsSpan());

            RustSafe.ExpApproxInPlace(vals);

            for (var i = 0; i < 8; ++i)
            {
                AssertAlmostEqual(expected[i], vals[i], _tolerance);
            }
        }

        [Fact]
        public void ExpApproxAvx()
        {
            var expected = Expected();
            var val = new Vector<float>(_inputs).AsVector256();
            var res = ExpApprox.ExpApproxAvx(val);

            var resArr = new float[8];
            res.AsVector().CopyTo(resArr);

            for (var i = 0; i < 8; ++i)
            {
                AssertAlmostEqual(expected[i], resArr[i], _tolerance);
            }
        }

        [Fact]
        public void ExpApproxVector()
        {
            var expected = Expected();
            var val = new Vector<float>(_inputs);
            var res = ExpApprox.ExpApproxVector(val);

            var resArr = new float[8];
            res.CopyTo(resArr);

            for (var i = 0; i < 8; ++i)
            {
                AssertAlmostEqual(expected[i], resArr[i], _tolerance);
            }
        }

        private void AssertAlmostEqual(float expected, float actual, float tolerance)
        {
            var delta = Math.Abs(expected) * _tolerance;
            Assert.InRange(actual, expected - delta, expected + delta);
        }
    }
}

using System;
using System.Collections.Generic;

namespace FastLinearEstimator.Bench
{
    public class AlmostEqualDouble : IEqualityComparer<double>
    {
        readonly double _tolerance;

        public AlmostEqualDouble(double tolerance)
        {
            _tolerance = tolerance;
        }

        public bool Equals(double x, double y)
        {
            var size = Math.Max(Math.Abs(x), Math.Abs(y));
            size = Math.Max(size, 1e-9);
            return Math.Abs(x - y) / size < _tolerance;
        }

        public int GetHashCode(double x) => x.GetHashCode();
    }

    public class AlmostEqualFloat : IEqualityComparer<float>
    {
        readonly double _tolerance;

        public AlmostEqualFloat(float tolerance)
        {
            _tolerance = tolerance;
        }

        public bool Equals(float x, float y)
        {
            var size = Math.Max(Math.Abs(x), Math.Abs(y));
            size = Math.Max(size, 1e-9f);
            return Math.Abs(x - y) / size < _tolerance;
        }

        public int GetHashCode(float x) => x.GetHashCode();
    }
}
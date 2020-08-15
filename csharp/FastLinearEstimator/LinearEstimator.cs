using System;
using System.Collections.Generic;
using System.Text;

namespace FastLinearEstimator
{
    public class LinearEstimator : IDisposable
    {
        private bool _disposedValue;
        private IntPtr _matrixNative;

        public int NumInputs { get; }
        public int NumOutputs { get; }

        /// <summary>
        /// Construct a native LinearEstimator
        /// </summary>
        /// <param name="coefficients">Coefficients in the form: float[NumInputs, NumOutputs]</param>
        /// <param name="intercepts">Intercepts in the form: float[NumOutputs]</param>
        public LinearEstimator(float[,] coefficients, float[] intercepts)
        {
            NumInputs = coefficients.GetLength(0);
            NumOutputs = coefficients.GetLength(1);
            if (NumOutputs != intercepts.Length)
            {
                throw new ArgumentOutOfRangeException("Intercepts dimension (num outputs) does not match coefficients second dimension");
            }
            unsafe
            {
                fixed (float* coeff = coefficients, inter = intercepts)
                {
                    _matrixNative = RustInterop.MatrixF32Create(NumInputs, NumOutputs, coeff, inter);
                    if (_matrixNative == IntPtr.Zero)
                    {
                        throw new ArgumentException("Could not construct matrix");
                    }
                }
            }
        }

        public void EstimateLinear(ReadOnlySpan<float> features, Span<float> results)
        {
            unsafe
            {
                fixed (float* feat = features, res = results)
                {
                    if (!RustInterop.MatrixF32Product(_matrixNative, feat, features.Length, res, results.Length))
                    {
                        throw new ArgumentException("Matrix product failed. Check dimensions.");
                    }
                }
            }

        }

        public void EstimateSoftMaxCumulative(ReadOnlySpan<float> features, Span<float> results)
        {
            unsafe
            {
                fixed (float* feat = features, res = results)
                {
                    if (!RustInterop.MatrixF32CumulativeSoftmax(_matrixNative, feat, features.Length, res, results.Length))
                    {
                        throw new ArgumentException("Matrix softmax failed. Check dimensions.");
                    }
                }
            }
        }

        protected virtual void Dispose(bool disposing)
        {
            if (!_disposedValue)
            {
                if (disposing)
                {
                    // dispose managed state -- not required
                }

                // free unmanaged resources 
                RustInterop.MatrixF32Delete(_matrixNative);
                _matrixNative = IntPtr.Zero;
                _disposedValue = true;
            }
        }

        ~LinearEstimator()
        {
            Dispose(disposing: false);
        }

        public void Dispose()
        {
            Dispose(disposing: true);
            GC.SuppressFinalize(this);
        }
    }
}

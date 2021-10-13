using System;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

namespace FastLinearEstimator
{
    internal static class RustInterop
    {
        // TODO: add resolver for specific platform (assuming it's all amd64 for now)
        const string LibName = "fast_linear_estimator_interop";

        [DllImport(LibName, EntryPoint = "test_add", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true, SetLastError = false)]
        internal static extern unsafe int TestAdd(int a, int b);

        [DllImport(LibName, EntryPoint = "matrix_f32_create", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true, SetLastError = false)]
        internal static extern unsafe IntPtr MatrixF32Create(int numInputs, int numOutputs, float* coefficients, float* intercepts);

        [DllImport(LibName, EntryPoint = "matrix_avx_f32_delete", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true, SetLastError = false)]
        internal static extern unsafe void MatrixF32Delete(IntPtr avxF32MatrixColumnMajor);

        // straight product
        [DllImport(LibName, EntryPoint = "matrix_f32_product", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true, SetLastError = false)]
        internal static extern unsafe bool MatrixF32Product(IntPtr avxF32MatrixColumnMajor, float* values, int valuesLength, float* results, int resultsLength);

        // product => cumulative softmax (approximate) for logistic regression
        [DllImport(LibName, EntryPoint = "matrix_f32_softmax_cumulative", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true, SetLastError = false)]
        internal static extern unsafe bool MatrixF32CumulativeSoftmax(IntPtr avxF32MatrixColumnMajor, float* values, int valuesLength, float* results, int resultsLength);

        // exponential approximation (in place)
        [DllImport(LibName, EntryPoint = "exp_approx_in_place", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true, SetLastError = false)]
        internal static extern unsafe void ExpApproxInPlace(float* values, int valuesLength);

        // exponential approximation (to destination)
        [DllImport(LibName, EntryPoint = "exp_approx", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true, SetLastError = false)]
        internal static extern unsafe void ExpApproxInPlace(float* src, float* dest, int valuesLength);
    }

    // safe public interfaces
    public static class RustSafe
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void ExpApproxInPlace(Span<float> values)
        {
            unsafe
            {
                fixed (float* p = values)
                {
                    RustInterop.ExpApproxInPlace(p, values.Length);
                }
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void ExpApprox(Span<float> src, Span<float> dest)
        {
            if (dest.Length != src.Length) throw new ArgumentOutOfRangeException("Length mismatch");
            unsafe
            {
                fixed (float* s = src, d = dest)
                {
                    RustInterop.ExpApproxInPlace(s, d, src.Length);
                }
            }
        }
    }
}

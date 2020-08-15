using System;
using System.Runtime.InteropServices;

namespace FastLinearEstimator
{
    internal static class RustInterop
    {
        [DllImport("logistic_lib", EntryPoint = "test_add", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true, SetLastError = false)]
        internal static extern unsafe int TestAdd(int a, int b);

        [DllImport("logistic_lib", EntryPoint = "matrix_f32_create", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true, SetLastError = false)]
        internal static extern unsafe IntPtr MatrixF32Create(int numInputs, int numOutputs, float* coefficients);

        [DllImport("logistic_lib", EntryPoint = "matrix_avx_f32_delete", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true, SetLastError = false)]
        internal static extern unsafe void MatrixF32Delete(IntPtr avxF32MatrixColumnMajor);

        // straight product
        [DllImport("logistic_lib", EntryPoint = "matrix_f32_product", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true, SetLastError = false)]
        internal static extern unsafe bool MatrixF32Product(IntPtr avxF32MatrixColumnMajor, float* values, int valuesLength, float* results, int resultsLength);

        // product => cumulative softmax (approximate) for logistic regression
        [DllImport("logistic_lib", EntryPoint = "matrix_f32_softmax_cumulative", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true, SetLastError = false)]
        internal static extern unsafe bool MatrixF32CumulativeSoftmax(IntPtr avxF32MatrixColumnMajor, float* values, int valuesLength, float* results, int resultsLength);
    }
}

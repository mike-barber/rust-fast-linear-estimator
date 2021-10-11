using System;
using System.Diagnostics;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace FastLinearEstimator
{
    public static class ExpApprox
    {
        public const float LOG2_E = 1.44269504088896340735992468100189214f;
        public const int EXP_BIAS_32 = 127;
        public const float EXP_HI = 88.3762626647949f;
        public const float EXP_LO = -88.3762626647949f;
        // reduced negative limit because we're converting to signed integers
        public const float EXP_LO_AVX_SIGNED = -88.028f;
        // taken from inavec
        public const float C0 = 1.06906116358144185133e-04f;
        public const float C1 = 3.03543677780836240743e-01f;
        public const float C2 = -2.24339532327269441936e-01f;
        public const float C3 = -7.92041454535668681958e-02f;
        public const float S = (float)(1u << 23);
        public const float B = S * (float)EXP_BIAS_32;


        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Vector256<float> ExpApproxAvx(Vector256<float> vector)
        {
            var x = vector;
            x = Avx.Min(x, Vector256.Create(EXP_HI));
            x = Avx.Max(x, Vector256.Create(EXP_LO_AVX_SIGNED));

            x = Avx.Multiply(x, Vector256.Create(LOG2_E));
            var fl = Avx.Floor(x);
            var xf = Avx.Subtract(x, fl);

            var kn = Vector256.Create(C3);
            kn = Avx.Add(Avx.Multiply(xf, kn), Vector256.Create(C2));
            kn = Avx.Add(Avx.Multiply(xf, kn), Vector256.Create(C1));
            kn = Avx.Add(Avx.Multiply(xf, kn), Vector256.Create(C0));
            x = Avx.Subtract(x, kn);

            var xf32 = Avx.Add(
                Avx.Multiply(Vector256.Create(S), x),
                Vector256.Create(B)
            );
            var xul = Avx.ConvertToVector256Int32(xf32);
            var res = xul.AsSingle();
            return res;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Vector<float> ExpApproxVector(Vector<float> vector)
        {
            var x = vector;
            x = Vector.Min(x, new Vector<float>(EXP_HI));
            x = Vector.Max(x, new Vector<float>(EXP_LO_AVX_SIGNED));

            x = Vector.Multiply(x, new Vector<float>(LOG2_E));

            var fl = Vector.Floor(x);
            var xf = Vector.Subtract(x, fl);

            var kn = new Vector<float>(C3);
            kn = Vector.Add(Vector.Multiply(xf, kn), new Vector<float>(C2));
            kn = Vector.Add(Vector.Multiply(xf, kn), new Vector<float>(C1));
            kn = Vector.Add(Vector.Multiply(xf, kn), new Vector<float>(C0));
            x = Vector.Subtract(x, kn);

            var xf32 = Vector.Add(
                Vector.Multiply(new Vector<float>(S), x),
                new Vector<float>(B)
            );
            var xul = Vector.ConvertToInt32(xf32);
            var res = Vector.AsVectorSingle(xul);
            return res;
        }
    }
}
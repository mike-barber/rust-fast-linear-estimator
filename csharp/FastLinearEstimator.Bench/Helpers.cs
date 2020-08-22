using System;
using System.Collections.Generic;
using System.Runtime.Intrinsics;

namespace FastLinearEstimator.Bench
{
    public static class Helpers
    {
        public static void PrintVector<T>(string label, IEnumerable<T> vect)
        {
            Console.WriteLine($"{label,20}: {string.Join(",", vect)}");
        }

        public static IEnumerable<float> FloatsFromAvx256(this IEnumerable<Vector256<float>> avxFloats)
        {
            foreach (var v in avxFloats) 
            {
                for (var i=0; i<8; ++i) {
                    yield return v.GetElement(i);
                }
            }
        }
    }
}

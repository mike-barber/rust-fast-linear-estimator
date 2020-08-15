using System;
using System.Linq;
using Xunit;

namespace FastLinearEstimator.Test
{
    public class BasicOperations
    {
        [Fact]
        public void BasicOperation()
        {
            var coefficients = new float[,]
            {
                { 1,2,3 }, { 4, 5, 6 }
            };
            var intercepts = new float[] { 100, 200, 300 };

            using var est = new LinearEstimator(coefficients, intercepts);
            var features = new float[] { 1, 2 };
            var results = new float[3];
            est.EstimateLinear(features, results);

            Assert.Equal(new float[] { 109, 212, 315 }, results);
        }

        [Fact]
        public void AllocateAndFree()
        {
            var coefficients = new float[,]
            {
                { 1,2,3 }, { 4, 5, 6 }
            };
            var intercepts = new float[] { 100, 200, 300 };

            // create a large number of estimators
            LinearEstimator[] estimators = Enumerable.Range(0, 1_000_000)
                .Select(_ => new LinearEstimator(coefficients, intercepts))
                .ToArray();

            // then use and free them             
            foreach (var est in estimators)
            {
                var features = new float[] { 1, 2 };
                var results = new float[3];
                est.EstimateLinear(features, results);
                Assert.Equal(new float[] { 109, 212, 315 }, results);

                est.Dispose();
            }
        }
    }
}

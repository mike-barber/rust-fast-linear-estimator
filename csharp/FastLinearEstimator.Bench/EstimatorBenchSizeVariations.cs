using BenchmarkDotNet.Attributes;

namespace FastLinearEstimator.Bench
{
    public class EstimatorBenchSizeVariations
    {
        [Params(2,3,4,6,8,10,12,20,40)]
        public int NumInputs = 8;

        [Params(2,3,5,7,8,15,25,32,36,60,120)]
        public int NumOutputs = 36;

        private EstimatorBench _estimatorBench;


        [GlobalSetup]
        public void GlobalSetup()
        {
            _estimatorBench = new EstimatorBench
            {
                NumInputs = NumInputs,
                NumOutputs = NumOutputs
            };
            _estimatorBench.GlobalSetup();
        }

        [GlobalCleanup]
        public void GlobalCleanup()
        {
            _estimatorBench?.GlobalCleanup();
        }

        [Benchmark]
        public float BenchMultLoopSoftmaxCumulativeFloat() => _estimatorBench.BenchCSharpSoftmax();

        [Benchmark]
        public float BenchMatrixSoftmaxCumulativeFloat() => _estimatorBench.BenchRustSoftmax();
    }
}

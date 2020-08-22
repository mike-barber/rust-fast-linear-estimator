using BenchmarkDotNet.Attributes;

namespace FastLinearEstimator.Bench
{
    public class EstimatorBenchSizeVariations
    {
        [Params(2,3,4,6,8,10,15,20,30,50,100)]
        public int NumInputs = 1;

        [Params(2,3,4,6,8,10,15,20,30,50,100)]
        public int NumOutputs = 1;

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
        public float BenchCSharpSoftmax() => _estimatorBench.BenchCSharpSoftmax();

        [Benchmark]
        public float BenchRustSoftmax() => _estimatorBench.BenchRustSoftmax();
    }
}

using BenchmarkDotNet.Configs;
using BenchmarkDotNet.Jobs;
using BenchmarkDotNet.Running;
using Perfolizer.Horology;

namespace FastLinearEstimator.Bench
{
    class Program
    {
        static void Main(string[] args)
        {
            // tests
            EstimatorBench.SelfTest();

            // benchmarking
            var job = Job.Default
                .WithGcServer(true);

            // in-process (need this until I fix the problem of not copying the Rust library to the benchmark output folder)
            job = job.WithToolchain(BenchmarkDotNet.Toolchains.InProcess.NoEmit.InProcessNoEmitToolchain.Instance);

            // limited iterations
            job = job
                  .WithWarmupCount(5)
                  .WithIterationCount(10)
                  .WithIterationTime(TimeInterval.FromMilliseconds(500))
                  ;

            var config = DefaultConfig.Instance.AddJob(job);

            BenchmarkRunner.Run<EstimatorBench>(config);
            //BenchmarkRunner.Run<EstimatorBenchSizeVariations>(config);
        }
    }
}

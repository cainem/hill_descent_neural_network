# Benchmarking Genetic Training Performance

This directory contains performance benchmarks for the genetic algorithm training.

## Running the Benchmark

```powershell
# Run the benchmark (generates HTML report)
cargo bench --bench genetic_training

# View the report
# Open: target/criterion/genetic_training_784_16_10_1000gen/report/index.html
```

## Generating Flamegraphs

To see where time is being spent, generate a flamegraph:

### Prerequisites

1. Install `cargo-flamegraph`:
   ```powershell
   cargo install flamegraph
   ```

2. On Windows, you may need to install the Windows Performance Toolkit (WPT):
   - Download from: https://docs.microsoft.com/en-us/windows-hardware/test/wpt/
   - Or via Visual Studio installer

### Generate Flamegraph

```powershell
# Run with profiling (creates flamegraph.svg)
cargo flamegraph --bench genetic_training -- --bench
```

Alternatively, you can use `perf` on Linux or integrate with the benchmark:

```powershell
# Using criterion's profiling support
cargo bench --bench genetic_training -- --profile-time=60
```

## Using cargo-profiling with Criterion

For more detailed profiling:

```powershell
# Install profiling tools
cargo install cargo-profiler

# Profile the benchmark
cargo profiler callgrind --bench genetic_training

# View results with kcachegrind (Linux) or qcachegrind (Windows/Mac)
```

## Benchmark Configuration

- **Network**: 784-16-10 (12,730 parameters)
- **Generations**: 1000
- **Population**: 500
- **Subset size**: 1000 training examples
- **Sample size**: 10 iterations (configurable in benchmark code)

## Expected Output

The benchmark will show:
- Mean execution time per run
- Standard deviation
- Throughput (iterations per second)
- Comparison with previous runs (if available)

## Interpreting Results

Key metrics to watch:
- **Time per generation**: Total time / 1000
- **Fitness evaluations per second**: (1000 generations × 500 population) / total time
- **Parameter updates per second**: Evaluations per second × 12,730 parameters

## Notes

- Benchmarks run in release mode automatically
- First run may take longer (cold cache)
- Subsequent runs are compared against baseline
- Results saved in `target/criterion/`

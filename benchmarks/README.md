# Detection Benchmarks

Benchmark the same detector and representative telemetry on the Jetson at increasing fleet sizes (initial targets: 100, 1,000, 10,000, and 50,000 flights).

Record throughput, elapsed time, memory use, and configuration. Separate startup/data-loading time from steady-state processing where practical.

Only introduce GPU acceleration if profiling identifies a substantial batch-parallel workload and end-to-end measurements show a benefit.

# Acceleration Boundary

The first detector should process streaming per-flight updates on CPU. Rule evaluation and each aircraft's temporal history are stateful and relatively small; moving every update to a GPU may cost more in data transfer and launch overhead than it saves.

Potential future CUDA/GPU candidates, after a working CPU baseline exists:

- Batch feature calculations across many flights or historical samples.
- Large-scale aircraft-pair conflict candidate calculations, if profiling shows pairwise work dominates.
- Optional bulk model inference if an ML detector is later adopted.

Keep GPU-specific code behind a batch interface so the detector contract and CPU fallback remain unchanged. Benchmark end-to-end on the Orin Nano before keeping an accelerated path. Do not assume CUDA is automatically the best GPU interface for every workload.

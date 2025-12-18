# Profiling

This folder contains Nsight Compute and Nsight Systems artifacts.

## Key Metrics

-  sm\_\_pipe_tensor_active.avg.pct_of_peak_sustained_active
-  dram\_\_bytes_read / write
-  l2_tex\_\_throughput
-  sm\_\_warps_active.avg

## Methodology

-  Compare baseline PyTorch ops vs fused kernels
-  Analyze memory vs compute bottlenecks
-  Track Tensor Core utilization changes

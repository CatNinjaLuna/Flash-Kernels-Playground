# Nsight Compute captures to include

Open the generated .ncu-rep in Nsight Compute UI and export screenshots for:

1. Summary (kernel time breakdown)
2. GPU Speed Of Light / Roofline
3. Memory Workload Analysis
4. Compute Workload Analysis

Key fields to point at :

-  Tensor Core utilization / pipe activity
-  dram bytes read/write (HBM traffic)
-  occupancy / warps active
-  L2 throughput

'''
Purpose:
Analyzes GPU kernel performance to identify bottlenecks like memory bandwidth limits, 
low occupancy, or inefficient memory access patterns. 
Essential for optimizing CUDA/Triton kernels.
'''

#!/usr/bin/env bash
set -euo pipefail

# Example: profile the Triton attention benchmark
# Produces: profiling/reports/attn.ncu-rep

mkdir -p profiling/reports

ncu \
  --set full \
  --target-processes all \
  --kernel-name-base function \
  -o profiling/reports/attn \
  python benchmarks/attention_bench.py


# make this file executable:
# chmod +x profiling/run_ncu_attention.sh
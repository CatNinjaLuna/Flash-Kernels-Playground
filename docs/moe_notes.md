# MoE Routing Notes

## What routing does
- Compute gating scores per token across experts
- Pick top-k experts
- Regroup tokens by expert for contiguous memory and better GEMM efficiency

## Why it’s tricky
- Load imbalance (some experts get more tokens)
- Regrouping is memory-bound and introduces overhead
- Padding vs bucketing tradeoff

## What to profile
- dram bytes read/write
- warp divergence / occupancy
- sorting/regroup cost vs expert GEMM speedup

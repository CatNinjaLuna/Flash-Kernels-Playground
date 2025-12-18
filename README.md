# Flash Kernels Playground
**Optimizing Attention & MoE Inference on NVIDIA GPUs**

## Overview
This project is a deep dive into **kernel-level performance optimization**
for large language model (LLM) inference, focusing on:

- Fused attention (QKᵀ → softmax → V)
- Mixture-of-Experts (MoE) routing
- Precision tradeoffs (FP16 / BF16 / FP8 / INT8)
- Decode vs prefill performance
- CUDA Graphs for launch overhead reduction

The goal is to approach **Tensor Core peak utilization** while minimizing
HBM traffic and kernel launch overhead.

## Tech Stack
- CUDA C++
- Triton
- CUTLASS
- PyTorch (custom ops)
- Nsight Compute / Nsight Systems

## Key Results
- ~65–70% Tensor Core peak utilization (measured via Nsight Compute)
- ~1.8× faster decode at 4k sequence length
- ~40% HBM traffic reduction via kernel fusion
- Static-shape CUDA Graph capture for decode loops

## Repo Structure
kernels/ # CUDA / Triton / CUTLASS kernels
benchmarks/ # Attention & MoE benchmarks
profiling/ # Nsight configs and reports
scripts/ # Build and run scripts
results/ # Logged performance numbers
docs/ # Design notes and diagrams

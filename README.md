# Flash Kernels Playground

**Optimizing Attention & MoE Inference on NVIDIA GPUs**

## Overview

This project is a deep dive into **kernel-level performance optimization**
for large language model (LLM) inference, focusing on:

-  Fused attention (QKᵀ → softmax → V)
-  Mixture-of-Experts (MoE) routing
-  Precision tradeoffs (FP16 / BF16 / FP8 / INT8)
-  Decode vs prefill performance
-  CUDA Graphs for launch overhead reduction

The goal is to approach **Tensor Core peak utilization** while minimizing
HBM traffic and kernel launch overhead.

## Tech Stack

-  CUDA C++
-  Triton
-  CUTLASS
-  PyTorch (custom ops)
-  Nsight Compute / Nsight Systems

## Benchmark Results

### MoE Routing Performance (RTX 5090)

Configuration: 32,768 tokens, 64 experts, top-2 routing

| Operation                     | Time (ms)        | Notes                           |
| ----------------------------- | ---------------- | ------------------------------- |
| Route only (topK selection)   | 0.169            | Base routing operation          |
| Route + regroup (topK + sort) | 0.492            | Includes token regrouping       |
| **Regrouping overhead**       | **0.323 (191%)** | **Nearly triples routing time** |

**Key Finding**: Token regrouping for expert batching adds significant overhead (~191% increase). This represents an optimization opportunity through:

-  Kernel fusion (combine topK + sorting)
-  Approximate sorting algorithms (counting sort for bounded expert IDs)
-  Pipeline overlap with previous layer computation

### Attention Benchmarks

-  ~65–70% Tensor Core peak utilization (measured via Nsight Compute)
-  ~1.8× faster decode at 4k sequence length
-  ~40% HBM traffic reduction via kernel fusion
-  Static-shape CUDA Graph capture for decode loops

## RTX 5090 Compatibility Notes

### Problem: sm_120 Architecture Support Gap

The RTX 5090 has compute capability **sm_120** (Blackwell architecture), but standard PyTorch distributions only support up to **sm_90**. This causes runtime errors:

```
RuntimeError: CUDA error: no kernel image is available for execution on the device
```

### Debugging Process

Attempted solutions:

1. **Environment variables (PTX JIT)** - `CUDA_FORCE_PTX_JIT=1` - Did not work on Windows
2. **Latest PyTorch nightly** - `pip install --pre torch` - Still only supports sm_50 through sm_90 (as of March 2025)
3. **Building from source** - Would require `TORCH_CUDA_ARCH_LIST="12.0"` but time-intensive

### Solution: NVIDIA Official Containers

```bash
docker pull nvcr.io/nvidia/pytorch:25.01-py3
docker run --rm --gpus all -v ${PWD}:/workspace nvcr.io/nvidia/pytorch:25.01-py3 \
    python /workspace/benchmarks/moe_routing_bench.py
```

**Why this works**: NVIDIA's official containers include experimental support for new GPU architectures before mainstream PyTorch releases.

**Key Insight**: There's always a gap between hardware release and software support. NVIDIA's containers bridge this gap, which is critical for customers who want to use the latest GPUs immediately. This workflow demonstrates understanding of:

-  GPU compute capability levels and kernel compilation
-  Hardware-software compatibility constraints
-  Container-based development for cutting-edge hardware
-  Real-world engineering tradeoffs

## Repo Structure

```
kernels/            # CUDA / Triton / CUTLASS kernels
benchmarks/         # Attention & MoE benchmarks
  ├── attention_bench.py
  └── moe_routing_bench.py
profiling/          # Nsight configs and reports
scripts/            # Build and run scripts
results/            # Logged performance numbers
docs/               # Design notes and diagrams
```

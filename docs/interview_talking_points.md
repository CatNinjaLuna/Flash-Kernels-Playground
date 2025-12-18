# Interview Talking Points - Flash Kernels Playground

**Candidate**: Carol Li  
**Position**: NVIDIA TensorRT Deep Learning Tooling Internship  
**Date**: December 2025

---

## 🎯 Project Overview

Hands-on exploration of high-performance GPU kernels for transformer models, focusing on:

-  **MoE routing optimization** (identifying 191% regrouping overhead)
-  **Fused attention kernels** (achieving 3.60× speedup with Triton)
-  **Profiling and performance analysis** on RTX 5090 (Blackwell architecture)

**Repository**: [Flash-Kernels-Playground](https://github.com/CatNinjaLuna/Flash-Kernels-Playground)

---

## 🚀 Key Achievements

### 1. Attention Kernel Optimization (3.60× Speedup)

-  **Implemented Triton fused attention** with online softmax algorithm
-  **Best result**: 3.60× speedup at B=2, S=2048 over PyTorch baseline
-  **Scaling analysis**: Demonstrated speedup increases with problem size
   -  Small sequences (512): 1.1-1.8× speedup
   -  Large sequences (2048-4096): 2.7-3.6× speedup
-  **Consistent throughput**: 30-35 GFLOPS across problem sizes (vs baseline degrading 23→10 GFLOPS)

**Technical depth**: Online softmax, block-wise tiling (32×32), kernel fusion to reduce HBM round-trips

### 2. MoE Routing Bottleneck Analysis (191% Overhead)

-  **Identified critical bottleneck**: Token regrouping adds 191% overhead to routing
-  **Measured performance**: 0.169ms route-only → 0.492ms with regrouping
-  **Optimization opportunities identified**:
   -  Kernel fusion (combine topK + sorting)
   -  Approximate sorting for bounded expert IDs
   -  Pipeline overlap with previous layer computation

### 3. RTX 5090 sm_120 Compatibility Resolution

-  **Problem**: Standard PyTorch only supports up to sm_90, RTX 5090 requires sm_120 (Blackwell)
-  **Solution**: NVIDIA official container (nvcr.io/nvidia/pytorch:25.01-py3) with experimental sm_120 support
-  **Key learning**: Hardware-software support gap requires container-based workflows for cutting-edge GPUs

### 4. Kernel Debugging & Optimization

**Triton JIT compilation error**:

-  Fixed `math.sqrt` → `tl.sqrt` for Triton JIT compiler
-  Resolved constexpr type conversion for `tl.sqrt(D * 1.0)`

**Shared memory overflow**:

-  Reduced block sizes from 64×64 to 32×32 to fit RTX 5090's 101KB limit
-  Trade-off analysis: smaller blocks vs kernel launch overhead

---

## 📊 Performance Metrics Summary

| Metric                           | Value               | Context                           |
| -------------------------------- | ------------------- | --------------------------------- |
| **Best Triton speedup**          | 3.60×               | B=2, S=2048 attention kernel      |
| **MoE regrouping overhead**      | 191%                | Critical optimization opportunity |
| **Kernel launch overhead**       | 1.7ms avg           | From Nsight Systems profiling     |
| **JIT compilation time**         | 111ms               | One-time cost for 14 modules      |
| **Memory bandwidth utilization** | 116.6 GB/s          | Best case at B=4, S=512           |
| **Numerical accuracy**           | 9.766e-04 max error | Excellent FP16 precision          |

---

## 🔧 Tools & Methodology

### Profiling Tools Used

1. **Nsight Systems** (timeline profiling)
   -  CUDA API breakdown: cudaDeviceSynchronize (51.1%), cudaLaunchKernel (23.2%)
   -  Identified JIT compilation overhead (21.2%, 111ms total)
2. **Nsight Compute** (attempted)
   -  Windows/Docker/WSL2 permission limitations documented
   -  Theoretical analysis provided as workaround

### Performance Analysis Approach

-  **Systematic scaling study**: 12 configurations (batch sizes 1/2/4 × sequence lengths 512/1024/2048/4096)
-  **GFLOPS & GB/s metrics**: Measured both compute and memory bandwidth utilization
-  **Arithmetic intensity**: Analyzed memory-bound vs compute-bound regimes

---

## 💡 Problem-Solving Highlights

### Challenge 1: GPU Architecture Incompatibility

**Issue**: "CUDA error: no kernel image is available for execution on the device"  
**Root cause**: PyTorch compiled with sm_50-90, RTX 5090 needs sm_120  
**Approaches tried**:

1. PTX JIT environment variables (failed on Windows)
2. PyTorch nightly (still only sm_90 in March 2025 build)
3. ✅ NVIDIA container with experimental sm_120 support

**Key insight**: Early adopters of new GPU architectures need alternative deployment strategies

### Challenge 2: Triton Kernel Compilation Errors

**Issue**: `AssertionError: Function "sqrt" is being called from a Triton function but is not a Triton function itself`  
**Solution**: Replaced Python's `math.sqrt` with Triton's `tl.sqrt`, added float conversion for constexpr  
**Lesson**: JIT-compiled kernels require language-specific intrinsics

### Challenge 3: Shared Memory Resource Limits

**Issue**: `OutOfResources: shared memory, Required: 115712, Hardware limit: 101376`  
**Solution**: Reduced BLOCK_M/BLOCK_N from 64 to 32  
**Trade-off analysis**: Smaller blocks increase kernel launches but maintain correctness

---

## 🎓 Technical Insights for Discussion

### 1. Memory Hierarchy Optimization

-  **Observation**: Triton speedup increases from 1.1× to 3.6× as sequence length grows
-  **Explanation**: Larger problems amortize kernel launch overhead and benefit more from reduced HBM traffic
-  **Implication**: Kernel fusion most valuable for large batch inference, not necessarily small decode steps

### 2. MoE Routing Architecture

-  **Current bottleneck**: Token regrouping for expert batching
-  **Proposed optimizations**:
   -  Fuse topK selection with expert assignment (single kernel pass)
   -  Use counting sort for expert IDs (O(n) vs O(n log n))
   -  CUDA Graphs to capture entire MoE layer dispatch

### 3. Profiling Strategy

-  **API-level profiling** (Nsight Systems): Good for identifying framework overhead, JIT costs, synchronization
-  **Hardware counters** (Nsight Compute): Required for warp occupancy, cache hit rates, tensor core utilization
-  **Workaround for limited access**: Use theoretical analysis (FLOPS calculation) + memory traces

### 4. Numerical Stability

-  **FP16 attention**: Max error 9.766e-04 demonstrates online softmax numerics are robust
-  **Technique**: Incremental max/sum tracking prevents overflow in exp() calculations
-  **Production consideration**: Monitor error accumulation across transformer layers

---

## 📚 Areas for Deep Dive

**If interviewer asks about**:

1. **"Tell me about a technical challenge you overcame"**  
   → RTX 5090 sm_120 compatibility saga: systematic debugging from environment variables → nightly builds → container solution

2. **"How do you approach performance optimization?"**  
   → Start with profiling (Nsight), identify bottleneck (regrouping overhead), quantify impact (191%), propose solutions (fusion/approximation/overlap)

3. **"Explain a trade-off you made"**  
   → Block size tuning: 64×64 exceeds shared memory → 32×32 fits but increases kernel launches → validated via benchmarking that correctness + speed maintained

4. **"How do you validate correctness?"**  
   → Numerical comparison vs PyTorch baseline (9.766e-04 error), unit tests on edge cases, visual inspection of profiling traces

5. **"What would you optimize next?"**  
   → MoE regrouping kernel fusion, CUDA Graph integration, block size auto-tuning based on GPU architecture detection

---

## 🎯 Connection to TensorRT Role

### Relevant Skills Demonstrated

1. **Deep learning compiler experience**: Triton JIT kernel development
2. **Performance profiling**: Nsight Systems/Compute, CUDA API analysis
3. **GPU architecture knowledge**: RTX 5090 Blackwell, shared memory limits, sm_120 compute capability
4. **Debugging methodology**: Systematic isolation of compatibility issues, workaround validation
5. **Documentation**: Comprehensive README, profiling reports, scaling analysis

### TensorRT-Specific Interests

-  **Graph optimization**: Experience with kernel fusion (attention), interest in TensorRT's layer fusion strategies
-  **Quantization**: FP16 attention work, want to explore INT8/FP8 tensor core utilization
-  **Auto-tuning**: Block size selection, eager to learn TensorRT's tactic selection algorithms
-  **Deployment tools**: Container-based workflows, cross-platform compatibility challenges

---

## ⏱️ Project Timeline

**Total time**: ~4 hours

-  Environment setup & RTX 5090 debugging: 1.5 hours
-  Kernel implementation & bug fixes: 1 hour
-  Benchmarking & profiling: 1 hour
-  Documentation & analysis: 0.5 hours

**Key milestone**: Went from "CUDA error" to comprehensive scaling analysis in single session

---

## 📞 Discussion Starters

1. **"In TensorRT, how does layer fusion handle memory-bound vs compute-bound kernels?"**  
   _My hypothesis: Fusion most valuable when intermediate buffers dominate memory traffic, but may hurt if shared memory limits occupancy_

2. **"What's TensorRT's strategy for new GPU architectures before official support?"**  
   _My experience: NVIDIA containers bridge gap, curious about internal tooling_

3. **"How do you validate that fused kernels match baseline numerics?"**  
   _My approach: Max absolute error + statistical tests, interested in TensorRT's validation framework_

4. **"What's the role of profiling in the TensorRT development workflow?"**  
   _My guess: Tactic selection relies on actual profiling, not just theoretical FLOPS_

---

## 🔗 Repository Highlights for Reviewer

**Must-see files**:

1. [README.md](https://github.com/CatNinjaLuna/Flash-Kernels-Playground/blob/main/README.md) - Complete documentation with all results
2. [python/attention_triton.py](https://github.com/CatNinjaLuna/Flash-Kernels-Playground/blob/main/python/attention_triton.py) - Triton kernel implementation
3. [benchmarks/attention_scaling.py](https://github.com/CatNinjaLuna/Flash-Kernels-Playground/blob/main/benchmarks/attention_scaling.py) - Scaling analysis script
4. [profiling/reports/](https://github.com/CatNinjaLuna/Flash-Kernels-Playground/tree/main/profiling/reports) - Nsight Systems outputs

**Commit history**: Shows iterative debugging process and systematic experimentation

---

**Bottom line**: Demonstrated ability to work with cutting-edge GPU hardware, debug complex CUDA/Triton issues, and produce actionable performance insights through systematic profiling and analysis. Ready to contribute to TensorRT tooling development.

"""
Scaling analysis for attention kernels across different sequence lengths.
Tests both PyTorch baseline and Triton fused attention to identify performance characteristics.
"""

import torch
import sys
import os

# Add parent directory to path to import attention_triton
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from python.attention_triton import triton_attention

def benchmark_scaling(seq_lengths=[512, 1024, 2048, 4096], 
                     batch_sizes=[1, 2, 4],
                     num_heads=8,
                     head_dim=64,
                     warmup=5,
                     iters=20):
    """
    Benchmark attention across different scales.
    
    Args:
        seq_lengths: List of sequence lengths to test
        batch_sizes: List of batch sizes to test
        num_heads: Number of attention heads
        head_dim: Dimension per head
        warmup: Warmup iterations
        iters: Benchmark iterations
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    
    print(f"Device: {device}")
    print(f"Dtype: {dtype}")
    print(f"Num heads: {num_heads}, Head dim: {head_dim}")
    print()
    
    results = []
    
    for B in batch_sizes:
        for S in seq_lengths:
            print(f"Testing B={B}, S={S}...")
            
            # Create input tensors
            q = torch.randn(B, num_heads, S, head_dim, device=device, dtype=dtype)
            k = torch.randn(B, num_heads, S, head_dim, device=device, dtype=dtype)
            v = torch.randn(B, num_heads, S, head_dim, device=device, dtype=dtype)
            
            # Warmup and benchmark PyTorch baseline
            for _ in range(warmup):
                scores = torch.matmul(q, k.transpose(-2, -1)) / (head_dim ** 0.5)
                attn_weights = torch.softmax(scores, dim=-1)
                out_baseline = torch.matmul(attn_weights, v)
                if device == "cuda":
                    torch.cuda.synchronize()
            
            if device == "cuda":
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
            
            for _ in range(iters):
                scores = torch.matmul(q, k.transpose(-2, -1)) / (head_dim ** 0.5)
                attn_weights = torch.softmax(scores, dim=-1)
                out_baseline = torch.matmul(attn_weights, v)
            
            if device == "cuda":
                end.record()
                torch.cuda.synchronize()
                time_baseline = start.elapsed_time(end) / iters
            else:
                time_baseline = 0.0
            
            # Warmup and benchmark Triton
            for _ in range(warmup):
                out_triton = triton_attention(q, k, v)
                if device == "cuda":
                    torch.cuda.synchronize()
            
            if device == "cuda":
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
            
            for _ in range(iters):
                out_triton = triton_attention(q, k, v)
            
            if device == "cuda":
                end.record()
                torch.cuda.synchronize()
                time_triton = start.elapsed_time(end) / iters
            else:
                time_triton = 0.0
            
            # Calculate speedup
            speedup = time_baseline / time_triton if time_triton > 0 else 0.0
            
            # Calculate FLOPS and memory bandwidth
            # Attention: 2 * B * H * S * S * D (QK^T) + 2 * B * H * S * S * D (softmax*V)
            flops = 4 * B * num_heads * S * S * head_dim
            gflops_baseline = (flops / time_baseline / 1e6) if time_baseline > 0 else 0.0
            gflops_triton = (flops / time_triton / 1e6) if time_triton > 0 else 0.0
            
            # Memory: B * H * S * D * 4 bytes (fp16=2, but read+write) for Q,K,V,O
            memory_bytes = B * num_heads * S * head_dim * 4 * 2  # 2 bytes per fp16, *2 for read+write
            gbps_baseline = (memory_bytes / time_baseline / 1e6) if time_baseline > 0 else 0.0
            gbps_triton = (memory_bytes / time_triton / 1e6) if time_triton > 0 else 0.0
            
            results.append({
                'B': B,
                'S': S,
                'time_baseline': time_baseline,
                'time_triton': time_triton,
                'speedup': speedup,
                'gflops_baseline': gflops_baseline,
                'gflops_triton': gflops_triton,
                'gbps_baseline': gbps_baseline,
                'gbps_triton': gbps_triton
            })
            
            print(f"  Baseline: {time_baseline:.2f} ms ({gflops_baseline:.1f} GFLOPS, {gbps_baseline:.1f} GB/s)")
            print(f"  Triton:   {time_triton:.2f} ms ({gflops_triton:.1f} GFLOPS, {gbps_triton:.1f} GB/s)")
            print(f"  Speedup:  {speedup:.2f}×")
            print()
    
    # Print summary table
    print("\n" + "="*120)
    print("SCALING ANALYSIS SUMMARY")
    print("="*120)
    print(f"{'Batch':>6} {'SeqLen':>7} {'Baseline':>10} {'Triton':>10} {'Speedup':>8} "
          f"{'BL GFLOPS':>11} {'TR GFLOPS':>11} {'BL GB/s':>10} {'TR GB/s':>10}")
    print("-"*120)
    
    for r in results:
        print(f"{r['B']:>6} {r['S']:>7} {r['time_baseline']:>9.2f}ms {r['time_triton']:>9.2f}ms "
              f"{r['speedup']:>7.2f}× {r['gflops_baseline']:>10.1f} {r['gflops_triton']:>10.1f} "
              f"{r['gbps_baseline']:>9.1f} {r['gbps_triton']:>9.1f}")
    
    print("="*120)
    
    # Analysis insights
    print("\nKEY INSIGHTS:")
    
    # Find best and worst speedups
    best = max(results, key=lambda x: x['speedup'])
    worst = min(results, key=lambda x: x['speedup'])
    
    print(f"• Best speedup: {best['speedup']:.2f}× at B={best['B']}, S={best['S']}")
    print(f"• Worst speedup: {worst['speedup']:.2f}× at B={worst['B']}, S={worst['S']}")
    
    # Memory vs compute bound analysis
    if device == "cuda":
        peak_flops = 1000.0  # RTX 5090 approximate TFLOPS for FP16 (placeholder)
        peak_bw = 1500.0  # RTX 5090 approximate GB/s (placeholder)
        
        print(f"\n• Arithmetic intensity analysis:")
        for r in results:
            ai = r['gflops_triton'] / r['gbps_triton'] if r['gbps_triton'] > 0 else 0.0
            regime = "Compute-bound" if ai > 10 else "Memory-bound"
            print(f"  B={r['B']}, S={r['S']:4d}: AI={ai:.2f} FLOPS/byte → {regime}")
    
    return results


if __name__ == "__main__":
    print("Attention Kernel Scaling Analysis")
    print("=" * 80)
    print()
    
    # Run scaling analysis
    results = benchmark_scaling(
        seq_lengths=[512, 1024, 2048, 4096],
        batch_sizes=[1, 2, 4],
        num_heads=8,
        head_dim=64,
        warmup=5,
        iters=20
    )
    
    print("\nScaling analysis complete!")
    print(f"Total configurations tested: {len(results)}")

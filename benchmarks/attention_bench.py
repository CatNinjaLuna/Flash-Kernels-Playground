import torch
import time
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from python.attention_triton import triton_attention

def benchmark(fn, warmup=10, iters=50, device="cuda"):
    for _ in range(warmup):
        fn()
    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(iters):
        fn()
    if device == "cuda":
        torch.cuda.synchronize()
    return (time.time() - t0) / iters

if __name__ == "__main__":
    torch.manual_seed(0)
    
    # RTX 5090 (sm_120) now supported via NVIDIA container
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    B, H, S, D = 1, 32, 2048, 128  # Increased S for GPU testing
    dtype = torch.float16 if device == "cuda" else torch.float32
    q = torch.randn(B, H, S, D, device=device, dtype=dtype)
    k = torch.randn(B, H, S, D, device=device, dtype=dtype)
    v = torch.randn(B, H, S, D, device=device, dtype=dtype)

    def baseline():
        attn = torch.softmax((q @ k.transpose(-1, -2)) / (D ** 0.5), dim=-1)
        return attn @ v

    tb = benchmark(baseline, iters=20, device=device)
    print(f"Baseline: {tb*1e3:.2f} ms")
    
    if device == "cuda":
        def triton_fn():
            return triton_attention(q, k, v)
        tt = benchmark(triton_fn, iters=50, device=device)
        print(f"Triton  : {tt*1e3:.2f} ms")
    else:
        print("Triton  : skipped (requires CUDA)")

    # quick correctness check on smaller S to avoid huge memory
    if device == "cuda":
        S2 = 512
        q2 = q[:, :, :S2, :].contiguous()
        k2 = k[:, :, :S2, :].contiguous()
        v2 = v[:, :, :S2, :].contiguous()
        ref = torch.softmax((q2 @ k2.transpose(-1, -2)) / (D ** 0.5), dim=-1) @ v2
        out = triton_attention(q2, k2, v2)
        max_err = (ref - out).abs().max().item()
        print(f"Max abs error (S=512): {max_err:.3e}")

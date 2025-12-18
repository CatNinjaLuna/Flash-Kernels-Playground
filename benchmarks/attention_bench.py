import torch
import time
from python.attention_triton import triton_attention

def benchmark(fn, warmup=10, iters=50):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.time() - t0) / iters

if __name__ == "__main__":
    torch.manual_seed(0)
    assert torch.cuda.is_available()

    B, H, S, D = 1, 32, 4096, 128
    q = torch.randn(B, H, S, D, device="cuda", dtype=torch.float16)
    k = torch.randn(B, H, S, D, device="cuda", dtype=torch.float16)
    v = torch.randn(B, H, S, D, device="cuda", dtype=torch.float16)

    def baseline():
        attn = torch.softmax((q @ k.transpose(-1, -2)) / (D ** 0.5), dim=-1)
        return attn @ v

    def triton_fn():
        return triton_attention(q, k, v)

    tb = benchmark(baseline, iters=20)
    tt = benchmark(triton_fn, iters=50)

    print(f"Baseline: {tb*1e3:.2f} ms")
    print(f"Triton  : {tt*1e3:.2f} ms")

    # quick correctness check on smaller S to avoid huge memory
    S2 = 512
    q2 = q[:, :, :S2, :].contiguous()
    k2 = k[:, :, :S2, :].contiguous()
    v2 = v[:, :, :S2, :].contiguous()
    ref = torch.softmax((q2 @ k2.transpose(-1, -2)) / (D ** 0.5), dim=-1) @ v2
    out = triton_attention(q2, k2, v2)
    max_err = (ref - out).abs().max().item()
    print(f"Max abs error (S=512): {max_err:.3e}")

import os
# Force PTX JIT compilation for unsupported architectures (RTX 5090 sm_120)
os.environ['CUDA_FORCE_PTX_JIT'] = '1'

import torch
import time

def bench(fn, warmup=20, iters=100, device="cuda"):
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

def topk_route(scores, k=2):
    # scores: [T, E] (tokens, experts)
    topv, topi = torch.topk(scores, k=k, dim=-1)  # [T,k]
    # Flatten token->(expert, weight)
    return topi.reshape(-1), topv.reshape(-1)

def regroup_tokens(expert_ids):
    # expert_ids: [T*k]
    # returns perm that groups by expert (stable enough for demo)
    return torch.argsort(expert_ids)

if __name__ == "__main__":
    # Try CUDA first, fall back to CPU if RTX 5090 sm_120 not supported
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        # Test if CUDA actually works with a small tensor
        if device == "cuda":
            test = torch.randn(10, 10, device="cuda", dtype=torch.float16)
            del test
        print(f"Using device: {device}")
    except RuntimeError as e:
        print(f"CUDA not supported (RTX 5090 needs sm_120 support): {e}")
        print("Falling back to CPU mode (timing not meaningful for GPU benchmarking)")
        device = "cpu"
    
    torch.manual_seed(0)

    T = 32768      # tokens
    E = 64         # experts
    K = 2          # top-k

    dtype = torch.float16 if device == "cuda" else torch.float32
    scores = torch.randn(T, E, device=device, dtype=dtype)

    def route_only():
        topk_route(scores, k=K)

    def route_and_regroup():
        eids, w = topk_route(scores, k=K)
        perm = regroup_tokens(eids)
        eids2 = eids[perm]
        w2 = w[perm]
        return eids2, w2

    t1 = bench(route_only, device=device)
    t2 = bench(route_and_regroup, device=device)

    print(f"\nResults (T={T}, E={E}, K={K}):")
    print(f"Route only        : {t1*1e3:.3f} ms")
    print(f"Route + regroup   : {t2*1e3:.3f} ms")
    print(f"Overhead          : {(t2-t1)*1e3:.3f} ms ({(t2/t1-1)*100:.1f}%)")

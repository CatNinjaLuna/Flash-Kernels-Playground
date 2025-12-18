import torch
import time

def bench(fn, warmup=20, iters=100):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(iters):
        fn()
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
    assert torch.cuda.is_available()
    torch.manual_seed(0)

    T = 32768      # tokens
    E = 64         # experts
    K = 2          # top-k

    scores = torch.randn(T, E, device="cuda", dtype=torch.float16)

    def route_only():
        topk_route(scores, k=K)

    def route_and_regroup():
        eids, w = topk_route(scores, k=K)
        perm = regroup_tokens(eids)
        eids2 = eids[perm]
        w2 = w[perm]
        return eids2, w2

    t1 = bench(route_only)
    t2 = bench(route_and_regroup)

    print(f"Route only        : {t1*1e3:.3f} ms")
    print(f"Route + regroup   : {t2*1e3:.3f} ms")

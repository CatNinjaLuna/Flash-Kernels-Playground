import torch
import time

def benchmark(fn, warmup=10, iters=50):
    for _ in range(warmup):
        fn()

    torch.cuda.synchronize()
    start = time.time()

    for _ in range(iters):
        fn()

    torch.cuda.synchronize()
    return (time.time() - start) / iters

if __name__ == "__main__":
    B, H, S, D = 1, 32, 4096, 128
    q = torch.randn(B, H, S, D, device="cuda", dtype=torch.float16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)

    def baseline():
        attn = torch.softmax(q @ k.transpose(-1, -2) / (D ** 0.5), dim=-1)
        return attn @ v

    t = benchmark(baseline)
    print(f"Baseline attention: {t*1e3:.2f} ms")


'''
torch.cuda.synchronize()

CUDA operations are asynchronous - When you call a PyTorch operation on the GPU, 
the CPU immediately continues without waiting for the GPU to finish.

Without synchronization, time.time() would measure how long it takes to queue operations, 
not how long they actually take to execute on the GPU.

With synchronization, you ensure all GPU work is done before recording timestamps, 
giving you the true execution time


'''
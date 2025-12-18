import math
import torch
import triton
import triton.language as tl


@triton.jit
def attn_fwd_kernel(Q, K, V, Out,
                    stride_qb, stride_qh, stride_qm, stride_qk,
                    stride_kb, stride_kh, stride_kn, stride_kk,
                    stride_vb, stride_vh, stride_vn, stride_vk,
                    stride_ob, stride_oh, stride_om, stride_ok,
                    B: tl.constexpr, H: tl.constexpr,
                    M: tl.constexpr, N: tl.constexpr, D: tl.constexpr,
                    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    """
    Computes Out[b,h,m,:] = softmax(Q[b,h,m,:] @ K[b,h,:,:]^T) @ V[b,h,:,:]
    Shapes:
      Q: [B,H,M,D], K: [B,H,N,D], V: [B,H,N,D], Out: [B,H,M,D]
    """
    pid_m = tl.program_id(0)  # along M
    pid_bh = tl.program_id(1)  # along B*H

    b = pid_bh // H
    h = pid_bh % H

    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    d_offsets = tl.arange(0, D)

    # Q tile pointer: [BLOCK_M, D]
    q_ptr = Q + b * stride_qb + h * stride_qh + m_offsets[:, None] * stride_qm + d_offsets[None, :] * stride_qk
    q = tl.load(q_ptr, mask=(m_offsets[:, None] < M), other=0.0).to(tl.float32)

    # Online softmax stats per row
    m_i = tl.full((BLOCK_M,), -float("inf"), tl.float32)
    l_i = tl.zeros((BLOCK_M,), tl.float32)

    # Accumulator for output: [BLOCK_M, D]
    acc = tl.zeros((BLOCK_M, D), tl.float32)

    # Loop over K/V tiles along N
    for n0 in range(0, N, BLOCK_N):
        n_offsets = n0 + tl.arange(0, BLOCK_N)

        k_ptr = K + b * stride_kb + h * stride_kh + n_offsets[None, :] * stride_kn + d_offsets[:, None] * stride_kk
        v_ptr = V + b * stride_vb + h * stride_vh + n_offsets[:, None] * stride_vn + d_offsets[None, :] * stride_vk

        k = tl.load(k_ptr, mask=(n_offsets[None, :] < N), other=0.0).to(tl.float32)  # [D, BN]
        v = tl.load(v_ptr, mask=(n_offsets[:, None] < N), other=0.0).to(tl.float32)  # [BN, D]

        # scores = q @ k : [BM, BN]
        scale = 1.0 / tl.sqrt(D * 1.0)
        scores = tl.dot(q, k) * scale

        # online softmax update
        tile_max = tl.max(scores, axis=1)
        m_new = tl.maximum(m_i, tile_max)
        p = tl.exp(scores - m_new[:, None])

        l_i = l_i * tl.exp(m_i - m_new) + tl.sum(p, axis=1)
        # rescale acc to new max
        acc = acc * tl.exp(m_i - m_new)[:, None] + tl.dot(p, v)

        m_i = m_new

    out = acc / l_i[:, None]
    out_ptr = Out + b * stride_ob + h * stride_oh + m_offsets[:, None] * stride_om + d_offsets[None, :] * stride_ok
    tl.store(out_ptr, out.to(tl.float16), mask=(m_offsets[:, None] < M))


def triton_attention(q, k, v, block_m=32, block_n=32):
    assert q.is_cuda and k.is_cuda and v.is_cuda
    assert q.dtype in (torch.float16, torch.bfloat16)
    B, H, M, D = q.shape
    _, _, N, _ = k.shape
    out = torch.empty((B, H, M, D), device=q.device, dtype=torch.float16)

    grid = (triton.cdiv(M, block_m), B * H)
    attn_fwd_kernel[grid](
        q, k, v, out,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        B=B, H=H, M=M, N=N, D=D,
        BLOCK_M=block_m, BLOCK_N=block_n,
        num_warps=4,
    )
    return out

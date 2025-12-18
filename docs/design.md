# Kernel Design Notes

## Fused Attention

Baseline attention materializes the full QKᵀ matrix, causing:

-  Excessive HBM traffic
-  Multiple kernel launches

Note: The naive implementation explicitly stores the attention score matrix, which is quadratic in sequence length and causes high HBM traffic.

We fuse:
QKᵀ → scale → mask → softmax → V

### Benefits

-  Avoids materializing attention matrix
-  Keeps partial sums in registers
-  Uses shared memory for K/V tiles

### Numerical Stability

-  Online softmax (log-sum-exp)
-  FP16/BF16 compute with FP32 accumulation

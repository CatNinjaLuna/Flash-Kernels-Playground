#include <cuda.h>
#include <cuda_runtime.h>

__global__ void fused_attention_kernel(
    const half* Q,
    const half* K,
    const half* V,
    half* O,
    int seq_len,
    int head_dim
) {
    // TODO:
    // - Tile Q/K/V
    // - Use shared memory for K/V
    // - Warp-level MMA for QK^T
    // - Online softmax
}

void launch_fused_attention(...) {
    // kernel launch config
}

#include <cuda_runtime.h>

__global__ void pv_dot_product_v1(const float* P, const float* V, float* output, int seq_len, int d_k) {
    // Computes Output[i, j] = P[i] dot V[:, j]
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < seq_len && col < d_k) {
        float out_val = 0.0f;
        for (int k = 0; k < seq_len; ++k) {
            out_val += P[row * seq_len + k] * V[k * d_k + col];
        }
        output[row * d_k + col] = out_val;
    }
}

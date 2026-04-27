#include <cuda_runtime.h>
#include <math.h>

__global__ void qk_dot_product_v1(const float* Q, const float* K, float* S, int seq_len, int d_k) {
    // Computes S[i, j] = (Q[i] dot K[j]) / sqrt(d_k)
    int row = blockIdx.y * blockDim.y + threadIdx.y; // i: Query token
    int col = blockIdx.x * blockDim.x + threadIdx.x; // j: Key token

    if (row < seq_len && col < seq_len) {
        float dot_product = 0.0f;
        for (int d = 0; d < d_k; ++d) {
            dot_product += Q[row * d_k + d] * K[col * d_k + d];
        }
        S[row * seq_len + col] = dot_product / sqrtf((float)d_k);
    }
}

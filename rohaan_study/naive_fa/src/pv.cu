#include "attention.cuh"
#include <cuda_runtime.h>

__global__ void pv_naive(const float* P, const float* V, float* O, int seq_len, int d_k) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < seq_len && col < d_k) {
        float sum = 0.0f;
        for (int k = 0; k < seq_len; ++k) {
            sum += P[row * seq_len + k] * V[k * d_k + col];
        }
        O[row * d_k + col] = sum;
    }
}

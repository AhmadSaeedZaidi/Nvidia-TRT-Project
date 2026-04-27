#include "attention.cuh"
#include <cuda_runtime.h>
#include <math.h>

#define TILE 16

__global__ void qk_naive(const float* Q, const float* K, float* S, int seq_len, int d_k) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < seq_len && col < seq_len) {
        float dot = 0.0f;
        for (int d = 0; d < d_k; ++d) {
            dot += Q[row * d_k + d] * K[col * d_k + d];
        }
        S[row * seq_len + col] = dot / sqrtf((float)d_k);
    }
}

__global__ void qk_tiled(const float* Q, const float* K, float* S, int seq_len, int d_k) {
    __shared__ float Qs[TILE][TILE];
    __shared__ float Ks[TILE][TILE];

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;
    float sum = 0.0f;

    int tiles = (d_k + TILE - 1) / TILE;
    for (int t = 0; t < tiles; ++t) {
        int d1 = t * TILE + threadIdx.x;
        if (row < seq_len && d1 < d_k) Qs[threadIdx.y][threadIdx.x] = Q[row * d_k + d1];
        else Qs[threadIdx.y][threadIdx.x] = 0.0f;

        int d2 = t * TILE + threadIdx.y;
        if (col < seq_len && d2 < d_k) Ks[threadIdx.x][threadIdx.y] = K[col * d_k + d2];
        else Ks[threadIdx.x][threadIdx.y] = 0.0f;

        __syncthreads();
        for (int k = 0; k < TILE; ++k) {
            sum += Qs[threadIdx.y][k] * Ks[threadIdx.x][k];
        }
        __syncthreads();
    }

    if (row < seq_len && col < seq_len) S[row * seq_len + col] = sum / sqrtf((float)d_k);
}

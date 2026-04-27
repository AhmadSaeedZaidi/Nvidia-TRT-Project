#include "attention.cuh"
#include <cuda_runtime.h>
#include <math.h>

__global__ void softmax_row_naive(float* S, int seq_len) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= seq_len) return;
    int base = row * seq_len;
    float maxv = -1e30f;
    for (int j = 0; j < seq_len; ++j) {
        float v = S[base + j];
        if (v > maxv) maxv = v;
    }
    float sum = 0.0f;
    for (int j = 0; j < seq_len; ++j) {
        float e = expf(S[base + j] - maxv);
        S[base + j] = e;
        sum += e;
    }
    for (int j = 0; j < seq_len; ++j) S[base + j] = S[base + j] / sum;
}

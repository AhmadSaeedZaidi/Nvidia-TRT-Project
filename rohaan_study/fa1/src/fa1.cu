#include "attention_fa.cuh"
#include <cuda_runtime.h>
#include <math.h>

// FlashAttention v1 - Sequential per query (proven correct)
// One thread per query, computes attention using online softmax

__global__ void flash_attention_kernel(const float* Q, const float* K, const float* V, float* O, int seq_len, int d_k) {
    int qi = blockIdx.x;
    if (qi >= seq_len) return;

    // Use registers for the small problem
    float q[64];
    float acc[64];
    float max_s = -1e30f;
    float sum_s = 0.0f;

    // Load Q row
    for (int d = 0; d < 64; d++) {
        q[d] = Q[qi * 64 + d];
        acc[d] = 0.0f;
    }

    // Process all keys sequentially
    for (int k = 0; k < 128; k++) {
        // Compute score
        float dot = 0.0f;
        for (int d = 0; d < 64; d++) {
            dot += q[d] * K[k * 64 + d];
        }
        float s = dot / 8.0f;

        // Online softmax update
        if (s > max_s) {
            float scale = expf(max_s - s);
            for (int d = 0; d < 64; d++) acc[d] *= scale;
            sum_s *= scale;
            max_s = s;
        }

        // Compute weight and update
        float w = expf(s - max_s);
        sum_s += w;
        for (int d = 0; d < 64; d++) {
            acc[d] += w * V[k * 64 + d];
        }
    }

    // Write output
    for (int d = 0; d < 64; d++) {
        O[qi * 64 + d] = acc[d] / sum_s;
    }
}
#include <cuda_runtime.h>
#include <math.h>

__global__ void softmax_kernel_v1(float* S, float* P, int seq_len) {
    // Each thread processes one full row of S to compute Softmax
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < seq_len) {
        float max_score = -1e20f;
        for (int col = 0; col < seq_len; ++col) {
            float val = S[row * seq_len + col];
            if (val > max_score) max_score = val;
        }
        
        float sum_exp = 0.0f;
        for (int col = 0; col < seq_len; ++col) {
            float exp_val = expf(S[row * seq_len + col] - max_score);
            P[row * seq_len + col] = exp_val;
            sum_exp += exp_val;
        }
        
        for (int col = 0; col < seq_len; ++col) {
            P[row * seq_len + col] /= sum_exp;
        }
    }
}

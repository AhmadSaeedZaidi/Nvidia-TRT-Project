#include <cuda_runtime.h>
#include <math.h>

#define SOFTMAX_BLOCK_SIZE 256

// Block-level reduction for Softmax
// Each block processes exactly one row.
__global__ void softmax_kernel_v2(const float* S, float* P, int seq_len) {
    int row = blockIdx.x;
    if (row >= seq_len) return;

    int tid = threadIdx.x;
    
    // 1. Thread-local max
    float local_max = -1e20f;
    for (int col = tid; col < seq_len; col += blockDim.x) {
        float val = S[row * seq_len + col];
        if (val > local_max) local_max = val;
    }

    // Block-level max reduction in shared memory
    __shared__ float shared_max[32]; // Max 1024 threads / 32 warps = 32
    int lane = tid % 32;
    int warp_id = tid / 32;

    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        local_max = fmaxf(local_max, __shfl_down_sync(0xffffffff, local_max, offset));
    }
    if (lane == 0) shared_max[warp_id] = local_max;
    __syncthreads();

    // Reduce warps
    local_max = (tid < (blockDim.x / 32)) ? shared_max[lane] : -1e20f;
    if (warp_id == 0) {
        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            local_max = fmaxf(local_max, __shfl_down_sync(0xffffffff, local_max, offset));
        }
        if (tid == 0) shared_max[0] = local_max; // Block max
    }
    __syncthreads();
    
    float block_max_val = shared_max[0];

    // 2. Thread-local sum of exp
    float local_sum = 0.0f;
    for (int col = tid; col < seq_len; col += blockDim.x) {
        float val = S[row * seq_len + col];
        float exp_val = expf(val - block_max_val);
        P[row * seq_len + col] = exp_val;
        local_sum += exp_val;
    }

    // Block-level sum reduction
    __shared__ float shared_sum[32];
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        local_sum += __shfl_down_sync(0xffffffff, local_sum, offset);
    }
    if (lane == 0) shared_sum[warp_id] = local_sum;
    __syncthreads();

    local_sum = (tid < (blockDim.x / 32)) ? shared_sum[lane] : 0.0f;
    if (warp_id == 0) {
        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            local_sum += __shfl_down_sync(0xffffffff, local_sum, offset);
        }
        if (tid == 0) shared_sum[0] = local_sum;
    }
    __syncthreads();

    float block_sum_val = shared_sum[0];

    // 3. Normalize
    for (int col = tid; col < seq_len; col += blockDim.x) {
        P[row * seq_len + col] /= block_sum_val;
    }
}

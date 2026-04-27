#include <cuda_runtime.h>

#define TILE_SIZE 32
#define COARSE_FACTOR 4

__global__ void pv_tiled_coarsened(const float* P, const float* V, float* output, int seq_len, int d_k) {
    // Computes Output = P * V
    // P is (seq_len, seq_len)
    // V is (seq_len, d_k)
    // output is (seq_len, d_k)
    
    __shared__ float P_ds[TILE_SIZE][TILE_SIZE];
    __shared__ float V_ds[TILE_SIZE][TILE_SIZE];

    int tx = threadIdx.x; // 0 to 7
    int ty = threadIdx.y; // 0 to 31

    int row = blockIdx.y * TILE_SIZE + ty;
    int col_start = blockIdx.x * TILE_SIZE + tx * COARSE_FACTOR; // The contiguous block of 4 elements for this thread

    float sum[COARSE_FACTOR] = {0.0f};

    // Inner loop over seq_len (the K dimension in GEMM)
    for (int t = 0; t < (seq_len + TILE_SIZE - 1) / TILE_SIZE; ++t) {
        
        // 1. Load P tile into Shared Memory
        // Using stride of 8 (since blockDim.x = 8) to load 32 elements per row
        for (int c = 0; c < COARSE_FACTOR; ++c) {
            int p_col = t * TILE_SIZE + tx + c * 8;
            if (row < seq_len && p_col < seq_len) {
                P_ds[ty][tx + c * 8] = P[row * seq_len + p_col];
            } else {
                P_ds[ty][tx + c * 8] = 0.0f;
            }
        }

        // 2. Load V tile into Shared Memory
        // Using stride of 8 (since blockDim.x = 8) to load 32 elements per row
        int v_row = t * TILE_SIZE + ty;
        for (int c = 0; c < COARSE_FACTOR; ++c) {
            int v_col = blockIdx.x * TILE_SIZE + tx + c * 8;
            if (v_row < seq_len && v_col < d_k) {
                V_ds[ty][tx + c * 8] = V[v_row * d_k + v_col];
            } else {
                V_ds[ty][tx + c * 8] = 0.0f;
            }
        }
        __syncthreads();

        // 3. Compute partial dot products
        // Thread calculates a 1 x COARSE_FACTOR chunk of the output
        for (int step = 0; step < TILE_SIZE; ++step) {
            float p_val = P_ds[ty][step]; // Loaded once into register
            
            #pragma unroll
            for (int c = 0; c < COARSE_FACTOR; ++c) {
                // We access V_ds using the contiguous mapping for computation
                sum[c] += p_val * V_ds[step][tx * COARSE_FACTOR + c];
            }
        }
        __syncthreads();
    }

    // Write results
    if (row < seq_len) {
        #pragma unroll
        for (int c = 0; c < COARSE_FACTOR; ++c) {
            int col = col_start + c;
            if (col < d_k) {
                output[row * d_k + col] = sum[c];
            }
        }
    }
}

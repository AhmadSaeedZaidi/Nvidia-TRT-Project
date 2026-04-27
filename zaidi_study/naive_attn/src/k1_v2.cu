#include <cuda_runtime.h>
#include <math.h>

// Tiled and Coarsened Q * K^T
// Q is (seq_len, d_k)
// K is (seq_len, d_k)
// We want S = Q * K^T, which is (seq_len, seq_len)
// S[i, j] = dot(Q[i, :], K[j, :]) / sqrt(d_k)

#define TILE_SIZE 32
#define COARSE_FACTOR 4

__global__ void qk_tiled_coarsened(const float* Q, const float* K, float* S, int seq_len, int d_k) {
    // Thread block computes a TILE_SIZE x TILE_SIZE block of the output S
    // Each thread computes a 1 x COARSE_FACTOR row-segment of that block
    
    // Shared memor
    __shared__ float Q_ds[TILE_SIZE][TILE_SIZE];
    __shared__ float K_ds[TILE_SIZE][TILE_SIZE];

    // indices
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * TILE_SIZE + ty;
    int col_start = blockIdx.x * TILE_SIZE + tx * COARSE_FACTOR;

    // Registers for thread coarsening (each thread holds COARSE_FACTOR output elements)
    float sum[COARSE_FACTOR] = {0.0f};


    for (int t = 0; t < (d_k + TILE_SIZE - 1) / TILE_SIZE; ++t) {
        
        if (row < seq_len && t * TILE_SIZE + tx < d_k) {
            Q_ds[ty][tx] = Q[row * d_k + t * TILE_SIZE + tx];
            for (int c = 1; c < COARSE_FACTOR; ++c) {
                if (t * TILE_SIZE + tx + c * (TILE_SIZE/COARSE_FACTOR) < d_k) {
                    Q_ds[ty][tx + c * (TILE_SIZE/COARSE_FACTOR)] = Q[row * d_k + t * TILE_SIZE + tx + c * (TILE_SIZE/COARSE_FACTOR)];
                } else {
                     Q_ds[ty][tx + c * (TILE_SIZE/COARSE_FACTOR)] = 0.0f;
                }
            }
        } else {
             Q_ds[ty][tx] = 0.0f;
             for (int c = 1; c < COARSE_FACTOR; ++c) {
                 Q_ds[ty][tx + c * (TILE_SIZE/COARSE_FACTOR)] = 0.0f;
             }
        }
        
        int k_row = blockIdx.x * TILE_SIZE + ty; 
        if (k_row < seq_len && t * TILE_SIZE + tx < d_k) {
            K_ds[ty][tx] = K[k_row * d_k + t * TILE_SIZE + tx];
            for (int c = 1; c < COARSE_FACTOR; ++c) {
                if (t * TILE_SIZE + tx + c * (TILE_SIZE/COARSE_FACTOR) < d_k) {
                    K_ds[ty][tx + c * (TILE_SIZE/COARSE_FACTOR)] = K[k_row * d_k + t * TILE_SIZE + tx + c * (TILE_SIZE/COARSE_FACTOR)];
                } else {
                     K_ds[ty][tx + c * (TILE_SIZE/COARSE_FACTOR)] = 0.0f;
                }
            }
        } else {
            K_ds[ty][tx] = 0.0f;
            for (int c = 1; c < COARSE_FACTOR; ++c) {
                K_ds[ty][tx + c * (TILE_SIZE/COARSE_FACTOR)] = 0.0f;
            }
        }

        __syncthreads();

        // 2. Compute partial dot products for the current tile
        // Thread calculates a 1 x COARSE_FACTOR chunk
        for (int step = 0; step < TILE_SIZE; ++step) {
            float q_val = Q_ds[ty][step]; // Loaded once into register
            
            #pragma unroll
            for (int c = 0; c < COARSE_FACTOR; ++c) {
                float k_val = K_ds[tx * COARSE_FACTOR + c][step]; // K is transposed in memory access
                sum[c] += q_val * k_val;
            }
        }

        __syncthreads();
    }

    // 3. Write results back to global memory (scaled by sqrt(d_k))
    float scale = 1.0f / sqrtf((float)d_k);
    if (row < seq_len) {
        #pragma unroll
        for (int c = 0; c < COARSE_FACTOR; ++c) {
            int col = col_start + c;
            if (col < seq_len) {
                S[row * seq_len + col] = sum[c] * scale;
            }
        }
    }
}

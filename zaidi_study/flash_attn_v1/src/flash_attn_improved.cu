#include <cuda_runtime.h>
#include <math.h>

#define BR 32
#define BC 32
#define D_K 64 // Fixed for this test scope

// Hardware-Optimized Flash Attention v1
// Uses 2D thread blocks (32x32 = 1024 threads) to eliminate register spills via Shared Memory
// and leverages warp-level reductions (__shfl_down_sync) for Max and Sum computation.
__global__ void flash_attn_1_fwd_f32_improved_kernel(const float* Q, const float* K, const float* V, float* O, int seq_len, int d_k) {
    int tx = threadIdx.x; // 0 to 31 (lane ID)
    int ty = threadIdx.y; // 0 to 31 (row in block)
    int bx = blockIdx.x;
    
    int row = bx * BR + ty;
    
    // 32 KB Shared Memory (No register spilling for Q/O!)
    __shared__ float Q_s[BR][D_K];
    __shared__ float K_s[BC][D_K];
    __shared__ float V_s[BC][D_K];
    __shared__ float O_s[BR][D_K];
    
    __shared__ float m_i[BR];
    __shared__ float l_i[BR];
    
    // Initialize scalars
    if (tx == 0) {
        m_i[ty] = -1e20f;
        l_i[ty] = 0.0f;
    }
    
    // 1. Cooperative load Q and initialize O
    if (row < seq_len) {
        Q_s[ty][tx] = Q[row * d_k + tx];
        Q_s[ty][tx + 32] = Q[row * d_k + tx + 32];
        O_s[ty][tx] = 0.0f;
        O_s[ty][tx + 32] = 0.0f;
    } else {
        Q_s[ty][tx] = 0.0f; Q_s[ty][tx + 32] = 0.0f;
        O_s[ty][tx] = 0.0f; O_s[ty][tx + 32] = 0.0f;
    }
    __syncthreads();
    
    int num_blocks_c = (seq_len + BC - 1) / BC;
    float scale = 1.0f / sqrtf((float)d_k);
    
    for (int bc = 0; bc < num_blocks_c; ++bc) {
        // 2. Cooperative load K and V tiles
        int k_row = bc * BC + ty;
        if (k_row < seq_len) {
            K_s[ty][tx] = K[k_row * d_k + tx];
            K_s[ty][tx + 32] = K[k_row * d_k + tx + 32];
            V_s[ty][tx] = V[k_row * d_k + tx];
            V_s[ty][tx + 32] = V[k_row * d_k + tx + 32];
        } else {
            K_s[ty][tx] = 0.0f; K_s[ty][tx + 32] = 0.0f;
            V_s[ty][tx] = 0.0f; V_s[ty][tx + 32] = 0.0f;
        }
        __syncthreads();
        
        // 3. Compute S_ij = Q_i * K_j^T
        float s_ij = 0.0f;
        for (int d = 0; d < D_K; ++d) {
            s_ij += Q_s[ty][d] * K_s[tx][d];
        }
        s_ij *= scale;
        
        if (bc * BC + tx >= seq_len) {
            s_ij = -1e20f; // bounds masking
        }
        
        // 4. Warp-level Max reduction across columns (tx)
        float m_ij_local = s_ij;
        for (int offset = 16; offset > 0; offset /= 2) {
            m_ij_local = fmaxf(m_ij_local, __shfl_down_sync(0xffffffff, m_ij_local, offset));
        }
        m_ij_local = __shfl_sync(0xffffffff, m_ij_local, 0); // Broadcast from lane 0 to whole warp
        
        float m_i_old = m_i[ty];
        float m_i_new = fmaxf(m_i_old, m_ij_local);
        
        // 5. Compute P_ij
        float p_ij = expf(s_ij - m_i_new);
        if (bc * BC + tx >= seq_len) {
            p_ij = 0.0f;
        }
        
        // 6. Warp-level Sum reduction across columns (tx)
        float l_ij_local = p_ij;
        for (int offset = 16; offset > 0; offset /= 2) {
            l_ij_local += __shfl_down_sync(0xffffffff, l_ij_local, offset);
        }
        l_ij_local = __shfl_sync(0xffffffff, l_ij_local, 0);
        
        float exp_diff = expf(m_i_old - m_i_new);
        float l_i_new = exp_diff * l_i[ty] + l_ij_local;
        
        if (tx == 0) {
            m_i[ty] = m_i_new;
            l_i[ty] = l_i_new;
        }
        
        // 7. Compute PV Update
        float pv_dot_0 = 0.0f;
        float pv_dot_1 = 0.0f;
        
        for (int j = 0; j < BC; ++j) {
            float p_val = __shfl_sync(0xffffffff, p_ij, j); // Fetch P_ij from lane j
            pv_dot_0 += p_val * V_s[j][tx];
            pv_dot_1 += p_val * V_s[j][tx + 32];
        }
        
        O_s[ty][tx] = exp_diff * O_s[ty][tx] + pv_dot_0;
        O_s[ty][tx + 32] = exp_diff * O_s[ty][tx + 32] + pv_dot_1;
        
        __syncthreads();
    }
    
    // 8. Write final normalized Output to Global Memory
    if (row < seq_len) {
        float final_l_i = l_i[ty];
        O[row * d_k + tx] = O_s[ty][tx] / final_l_i;
        O[row * d_k + tx + 32] = O_s[ty][tx + 32] / final_l_i;
    }
}

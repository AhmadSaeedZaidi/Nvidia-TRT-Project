#include <cuda_runtime.h>
#include <math.h>

#define BR 32
#define BC 32
#define D_K 64 // Fixed for this test scope

// Simplified Flash Attention 1 (Dao et al. 2022) implementation
__global__ void flash_attn_1_fwd_f32_kernel(const float* Q, const float* K, const float* V, float* O, int seq_len, int d_k) {
    // Each block processes a tile of BR rows of the Output matrix.
    // Thread block is 1D: BR threads. Each thread computes 1 row of O.
    
    int bx = blockIdx.x;
    int tx = threadIdx.x; // 0 to BR - 1

    int row = bx * BR + tx;
    if (row >= seq_len) return;

    // Registers for Q_i and O_i state
    float q_i[D_K];
    float o_i[D_K] = {0.0f};
    
    // Register initialization
    for (int d = 0; d < D_K; ++d) {
        q_i[d] = Q[row * d_k + d];
    }
    
    float m_i = -1e20f;
    float l_i = 0.0f;
    float scale = 1.0f / sqrtf((float)d_k);

    // Shared Memory for KV tiles
    __shared__ float K_s[BC][D_K];
    __shared__ float V_s[BC][D_K];

    int num_blocks_c = (seq_len + BC - 1) / BC;
    
    // Inner loop over column blocks of K and V
    for (int bc = 0; bc < num_blocks_c; ++bc) {
        
        // 1. Load K and V tiles cooperatively
        // We have BR=32 threads. We need to load BC x D_K = 32 x 64 elements.
        // Each thread (tx) loads row `tx` of the K/V tile.
        for (int d = 0; d < D_K; ++d) {
            int k_row = bc * BC + tx;
            if (k_row < seq_len) {
                K_s[tx][d] = K[k_row * d_k + d];
                V_s[tx][d] = V[k_row * d_k + d];
            } else {
                K_s[tx][d] = 0.0f;
                V_s[tx][d] = 0.0f;
            }
        }
        __syncthreads();

        // 2. Compute S_ij = Q_i * K_s^T
        float s_ij[BC];
        float m_ij_local = -1e20f;
        for (int j = 0; j < BC; ++j) {
            float dot = 0.0f;
            for (int d = 0; d < D_K; ++d) {
                dot += q_i[d] * K_s[j][d];
            }
            dot *= scale; // Scale standard attention
            s_ij[j] = dot;
            if (dot > m_ij_local) {
                m_ij_local = dot;
            }
        }

        // 3. Update running max
        float m_i_new = fmaxf(m_i, m_ij_local);

        // 4. Compute P_ij = exp(S_ij - m_i_new) and local sum
        float P_ij[BC];
        float l_ij_local = 0.0f;
        for (int j = 0; j < BC; ++j) {
            int k_col = bc * BC + j;
            if (k_col < seq_len) {
                P_ij[j] = expf(s_ij[j] - m_i_new);
                l_ij_local += P_ij[j];
            } else {
                P_ij[j] = 0.0f; // Masking out-of-bounds keys
            }
        }

        // 5. Update running partition sums and logic
        float exp_diff = expf(m_i - m_i_new);
        float l_i_new = exp_diff * l_i + l_ij_local;

        // 6. Update O_i representation
        for (int d = 0; d < D_K; ++d) {
            float pv_dot = 0.0f;
            for (int j = 0; j < BC; ++j) {
                pv_dot += P_ij[j] * V_s[j][d];
            }
            o_i[d] = exp_diff * o_i[d] + pv_dot;
        }

        // Save states for next loop iteration
        m_i = m_i_new;
        l_i = l_i_new;
        __syncthreads(); // Wait before next tile overrides shared memory
    }

    // 7. Write final normalized O_i to Global Memory
    for (int d = 0; d < D_K; ++d) {
        O[row * d_k + d] = o_i[d] / l_i;
    }
}

#pragma once

// Flash Attention v1
__global__ void flash_attn_1_fwd_f32_kernel(const float* Q, const float* K, const float* V, float* O, int seq_len, int d_k);

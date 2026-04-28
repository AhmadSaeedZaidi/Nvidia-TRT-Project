#pragma once

// Flash Attention v1 (Naive 1-thread-per-row)
__global__ void flash_attn_1_fwd_f32_kernel(const float* Q, const float* K, const float* V, float* O, int seq_len, int d_k);

// Flash Attention v1 (Hardware Optimized 2D-Block)
__global__ void flash_attn_1_fwd_f32_improved_kernel(const float* Q, const float* K, const float* V, float* O, int seq_len, int d_k);

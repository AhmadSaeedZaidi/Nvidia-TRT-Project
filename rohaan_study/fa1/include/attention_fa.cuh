#pragma once
#include <cuda_runtime.h>

// FlashAttention kernel declaration
// Each block handles one query, all threads in block collaborate
__global__ void flash_attention_kernel(const float* Q, const float* K, const float* V, float* O, int seq_len, int d_k);
#pragma once

// Kernel 1: Q * K^T
__global__ void qk_dot_product_v1(const float* Q, const float* K, float* S, int seq_len, int d_k);
__global__ void qk_tiled_coarsened(const float* Q, const float* K, float* S, int seq_len, int d_k);

// Kernel 2: Softmax
__global__ void softmax_kernel_v1(float* S, float* P, int seq_len);
__global__ void softmax_kernel_v2(const float* S, float* P, int seq_len);

// Kernel 3: P * V
__global__ void pv_dot_product_v1(const float* P, const float* V, float* output, int seq_len, int d_k);
__global__ void pv_tiled_coarsened(const float* P, const float* V, float* output, int seq_len, int d_k);

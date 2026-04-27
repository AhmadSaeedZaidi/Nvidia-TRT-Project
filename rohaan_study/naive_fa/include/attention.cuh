#pragma once
#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

__global__ void qk_naive(const float* Q, const float* K, float* S, int seq_len, int d_k);
__global__ void qk_tiled(const float* Q, const float* K, float* S, int seq_len, int d_k);

__global__ void softmax_row_naive(float* S, int seq_len);

__global__ void pv_naive(const float* P, const float* V, float* O, int seq_len, int d_k);

#ifdef __cplusplus
}
#endif

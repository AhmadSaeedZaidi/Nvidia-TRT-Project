#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <cmath>
#include "kernels.cuh"

void checkCuda(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error: " << msg << " (" << cudaGetErrorString(err) << ")" << std::endl;
        exit(EXIT_FAILURE);
    }
}

int main() {
    int seq_len = 128;
    int d_k = 64;

    size_t qkv_size = seq_len * d_k * sizeof(float);
    size_t s_size = seq_len * seq_len * sizeof(float);

    // Host memory
    std::vector<float> h_Q(seq_len * d_k), h_K(seq_len * d_k), h_V(seq_len * d_k);
    std::vector<float> h_S_v1(seq_len * seq_len), h_S_v2(seq_len * seq_len);
    std::vector<float> h_P(seq_len * seq_len);
    std::vector<float> h_Out(seq_len * d_k);

    // Initialize with random numbers
    for (int i = 0; i < seq_len * d_k; ++i) {
        h_Q[i] = static_cast<float>(rand()) / RAND_MAX;
        h_K[i] = static_cast<float>(rand()) / RAND_MAX;
        h_V[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    // Device memory
    float *d_Q, *d_K, *d_V, *d_S, *d_S_v2, *d_P, *d_Out;
    checkCuda(cudaMalloc(&d_Q, qkv_size), "Malloc Q");
    checkCuda(cudaMalloc(&d_K, qkv_size), "Malloc K");
    checkCuda(cudaMalloc(&d_V, qkv_size), "Malloc V");
    checkCuda(cudaMalloc(&d_S, s_size), "Malloc S");
    checkCuda(cudaMalloc(&d_S_v2, s_size), "Malloc S v2");
    checkCuda(cudaMalloc(&d_P, s_size), "Malloc P");
    checkCuda(cudaMalloc(&d_Out, qkv_size), "Malloc Out");

    // Copy to device
    cudaMemcpy(d_Q, h_Q.data(), qkv_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_K, h_K.data(), qkv_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, h_V.data(), qkv_size, cudaMemcpyHostToDevice);

    // Run Kernel 1 (v1)
    dim3 threads1(16, 16);
    dim3 blocks1((seq_len + threads1.x - 1) / threads1.x, (seq_len + threads1.y - 1) / threads1.y);
    qk_dot_product_v1<<<blocks1, threads1>>>(d_Q, d_K, d_S, seq_len, d_k);
    cudaDeviceSynchronize();

    // Run Kernel 1 (v2 - Tiled)
    dim3 threads_t(8, 32); // specific block logic from k2_v2.cu (tx goes 0 to TILE_SIZE/COARSE=8)
    dim3 blocks_t((seq_len + 31) / 32, (seq_len + 31) / 32);
    qk_tiled_coarsened<<<blocks_t, threads_t>>>(d_Q, d_K, d_S_v2, seq_len, d_k);
    cudaDeviceSynchronize();

    // Check v1 vs v2
    cudaMemcpy(h_S_v1.data(), d_S, s_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_S_v2.data(), d_S_v2, s_size, cudaMemcpyDeviceToHost);

    float max_diff = 0.0f;
    for (int i = 0; i < seq_len * seq_len; ++i) {
        float diff = std::abs(h_S_v1[i] - h_S_v2[i]);
        if (diff > max_diff) max_diff = diff;
    }
    std::cout << "K1 (QK^T) Max Difference b/w v1 and v2: " << max_diff << "\n";

    // Run Kernel 2 (v1)
    dim3 blocks2((seq_len + 255) / 256);
    dim3 threads2(256);
    softmax_kernel_v1<<<blocks2, threads2>>>(d_S, d_P, seq_len);
    cudaDeviceSynchronize();
    
    // Similarly one could run softmax_kernel_v2
    
    // Run Kernel 3 (v1)
    dim3 threads3(16, 16);
    dim3 blocks3((d_k + threads3.x - 1) / threads3.x, (seq_len + threads3.y - 1) / threads3.y);
    pv_dot_product_v1<<<blocks3, threads3>>>(d_P, d_V, d_Out, seq_len, d_k);
    cudaDeviceSynchronize();

    std::cout << "Attention computation completed successfully!\n";

    cudaFree(d_Q); cudaFree(d_K); cudaFree(d_V);
    cudaFree(d_S); cudaFree(d_S_v2); cudaFree(d_P); cudaFree(d_Out);

    return 0;
}

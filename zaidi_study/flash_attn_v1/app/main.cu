#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <cmath>
#include <iomanip>
#include "kernels.cuh"
#include "../naive_attn/include/kernels.cuh"

// Clean exit on CUDA errors
void checkCuda(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error: " << msg << " (" << cudaGetErrorString(err) << ")\n";
        exit(EXIT_FAILURE);
    }
}

// Utility to find maximum difference between two arrays
float getMaxDiff(const std::vector<float>& arr1, const std::vector<float>& arr2) {
    float max_diff = 0.0f;
    for (size_t i = 0; i < arr1.size(); ++i) {
        max_diff = std::max(max_diff, std::abs(arr1[i] - arr2[i]));
    }
    return max_diff;
}

int main() {
    int seq_len = 2048; // increase to 2048
    int d_k = 64;

    size_t qkv_size = seq_len * d_k * sizeof(float);
    size_t s_size = seq_len * seq_len * sizeof(float);

    // Host memory
    std::vector<float> h_Q(seq_len * d_k), h_K(seq_len * d_k), h_V(seq_len * d_k);
    std::vector<float> h_Out_v1(seq_len * d_k), h_Out_v2(seq_len * d_k), h_Out_flash(seq_len * d_k);

    for (int i = 0; i < seq_len * d_k; ++i) {
        h_Q[i] = static_cast<float>(rand()) / RAND_MAX;
        h_K[i] = static_cast<float>(rand()) / RAND_MAX;
        h_V[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    // Device memory
    float *d_Q, *d_K, *d_V;
    float *d_S_v1, *d_P_v1, *d_Out_v1;
    float *d_S_v2, *d_P_v2, *d_Out_v2;
    float *d_Out_flash;

    checkCuda(cudaMalloc(&d_Q, qkv_size), "Malloc Q");
    checkCuda(cudaMalloc(&d_K, qkv_size), "Malloc K");
    checkCuda(cudaMalloc(&d_V, qkv_size), "Malloc V");
    
    checkCuda(cudaMalloc(&d_S_v1, s_size), "Malloc S v1");
    checkCuda(cudaMalloc(&d_P_v1, s_size), "Malloc P v1");
    checkCuda(cudaMalloc(&d_Out_v1, qkv_size), "Malloc Out v1");

    checkCuda(cudaMalloc(&d_S_v2, s_size), "Malloc S v2");
    checkCuda(cudaMalloc(&d_P_v2, s_size), "Malloc P v2");
    checkCuda(cudaMalloc(&d_Out_v2, qkv_size), "Malloc Out v2");

    checkCuda(cudaMalloc(&d_Out_flash, qkv_size), "Malloc Out flash");

    cudaMemcpy(d_Q, h_Q.data(), qkv_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_K, h_K.data(), qkv_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, h_V.data(), qkv_size, cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float t_v1_k1, t_v1_k2, t_v1_k3;
    float t_v2_k1, t_v2_k2, t_v2_k3;
    float t_flash, t_flash_improved;

    // =========================================================
    // 1. RUN NAIVE BASELINE (V1)
    // =========================================================
    
    // K1
    dim3 t1_v1(16, 16);
    dim3 b1_v1((seq_len + 15) / 16, (seq_len + 15) / 16);
    cudaEventRecord(start);
    qk_dot_product_v1<<<b1_v1, t1_v1>>>(d_Q, d_K, d_S_v1, seq_len, d_k);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&t_v1_k1, start, stop);

    // K2
    dim3 t2_v1(256);
    dim3 b2_v1((seq_len + 255) / 256);
    cudaEventRecord(start);
    softmax_kernel_v1<<<b2_v1, t2_v1>>>(d_S_v1, d_P_v1, seq_len);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&t_v1_k2, start, stop);

    // K3
    dim3 t3_v1(16, 16);
    dim3 b3_v1((d_k + 15) / 16, (seq_len + 15) / 16);
    cudaEventRecord(start);
    pv_dot_product_v1<<<b3_v1, t3_v1>>>(d_P_v1, d_V, d_Out_v1, seq_len, d_k);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&t_v1_k3, start, stop);

    // =========================================================
    // 2. RUN OPTIMIZED PIPELINE (V2)
    // =========================================================

    // K1
    dim3 t1_v2(8, 32); 
    dim3 b1_v2((seq_len + 31) / 32, (seq_len + 31) / 32);
    cudaEventRecord(start);
    qk_tiled_coarsened<<<b1_v2, t1_v2>>>(d_Q, d_K, d_S_v2, seq_len, d_k);
    checkCuda(cudaGetLastError(), "qk_tiled_coarsened launch");
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&t_v2_k1, start, stop);

    // K2
    dim3 t2_v2(256);
    dim3 b2_v2(seq_len); // 1 block per row
    cudaEventRecord(start);
    softmax_kernel_v2<<<b2_v2, t2_v2>>>(d_S_v2, d_P_v2, seq_len);
    checkCuda(cudaGetLastError(), "softmax_kernel_v2 launch");
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&t_v2_k2, start, stop);

    // K3
    dim3 t3_v2(8, 32);
    dim3 b3_v2((d_k + 31) / 32, (seq_len + 31) / 32);
    cudaEventRecord(start);
    pv_tiled_coarsened<<<b3_v2, t3_v2>>>(d_P_v2, d_V, d_Out_v2, seq_len, d_k);
    checkCuda(cudaGetLastError(), "pv_tiled_coarsened launch");
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&t_v2_k3, start, stop);


    // =========================================================
    // 3. RUN FLASH ATTENTION (V1)
    // =========================================================
    
    // BR = 32 blocks
    int BR = 32;
    dim3 t_flash_block(BR); // 32 threads, 1 per row for Naive FA1
    dim3 b_flash_grid((seq_len + BR - 1) / BR);
    
    cudaEventRecord(start);
    flash_attn_1_fwd_f32_kernel<<<b_flash_grid, t_flash_block>>>(d_Q, d_K, d_V, d_Out_flash, seq_len, d_k);
    checkCuda(cudaGetLastError(), "flash_attn_1 launch");
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&t_flash, start, stop);

    // =========================================================
    // 4. RUN FLASH ATTENTION (V1) IMPROVED
    // =========================================================
    
    float *d_Out_flash_improved;
    checkCuda(cudaMalloc(&d_Out_flash_improved, qkv_size), "Malloc Out flash improved");
    
    dim3 t_flash_improved_block(32, 32); // 2D block: 32 cols (lane ID), 32 rows
    dim3 b_flash_improved_grid((seq_len + BR - 1) / BR);
    
    cudaEventRecord(start);
    flash_attn_1_fwd_f32_improved_kernel<<<b_flash_improved_grid, t_flash_improved_block>>>(d_Q, d_K, d_V, d_Out_flash_improved, seq_len, d_k);
    checkCuda(cudaGetLastError(), "flash_attn_improved launch");
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&t_flash_improved, start, stop);


    // =========================================================
    // COMPARISON
    // =========================================================
    cudaMemcpy(h_Out_v1.data(), d_Out_v1, qkv_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_Out_v2.data(), d_Out_v2, qkv_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_Out_flash.data(), d_Out_flash, qkv_size, cudaMemcpyDeviceToHost);
    
    std::vector<float> h_Out_flash_improved(seq_len * d_k);
    cudaMemcpy(h_Out_flash_improved.data(), d_Out_flash_improved, qkv_size, cudaMemcpyDeviceToHost);

    std::cout << "========= CORRECTNESS =========\n";
    std::cout << "Max diff (V1 vs V2):    " << std::scientific << getMaxDiff(h_Out_v1, h_Out_v2) << "\n";
    std::cout << "Max diff (V1 vs FA_1):  " << std::scientific << getMaxDiff(h_Out_v1, h_Out_flash) << "\n";
    std::cout << "Max diff (V1 vs FA_2):  " << std::scientific << getMaxDiff(h_Out_v1, h_Out_flash_improved) << "\n\n";

    std::cout << "========= PERFORMANCE (ms) =========\n";
    
    float total_v1 = t_v1_k1 + t_v1_k2 + t_v1_k3;
    float total_v2 = t_v2_k1 + t_v2_k2 + t_v2_k3;

    std::cout << std::left << std::setw(25) << "Naive Pipeline (V1)" << ": " << std::fixed << std::setprecision(4) << total_v1 << " ms\n";
    std::cout << std::left << std::setw(25) << "Optimized Pipeline (V2)" << ": " << total_v2 << " ms\n";
    std::cout << std::left << std::setw(25) << "Flash Attention v1" << ": " << t_flash << " ms\n";
    std::cout << std::left << std::setw(25) << "Flash Attn (Improved)" << ": " << t_flash_improved << " ms\n\n";
    
    std::cout << "Speedup (FA Imp vs V1): " << (total_v1 / t_flash_improved) << "x\n";
    std::cout << "Speedup (FA Imp vs V2): " << (total_v2 / t_flash_improved) << "x\n";
    std::cout << "Speedup (FA Imp vs FA1): " << (t_flash / t_flash_improved) << "x\n";

    // Cleanup
    cudaFree(d_Q); cudaFree(d_K); cudaFree(d_V);
    cudaFree(d_S_v1); cudaFree(d_P_v1); cudaFree(d_Out_v1);
    cudaFree(d_S_v2); cudaFree(d_P_v2); cudaFree(d_Out_v2);
    cudaFree(d_Out_flash);
    cudaFree(d_Out_flash_improved);
    
    return 0;
}

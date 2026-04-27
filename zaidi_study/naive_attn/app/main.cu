#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <cmath>
#include <iomanip>
#include "kernels.cuh"

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
    int seq_len = 128;
    int d_k = 64;

    size_t qkv_size = seq_len * d_k * sizeof(float);
    size_t s_size = seq_len * seq_len * sizeof(float);

    // Host memory
    std::vector<float> h_Q(seq_len * d_k), h_K(seq_len * d_k), h_V(seq_len * d_k);
    std::vector<float> h_S_v1(seq_len * seq_len), h_S_v2(seq_len * seq_len);
    std::vector<float> h_P_v1(seq_len * seq_len), h_P_v2(seq_len * seq_len);
    std::vector<float> h_Out_v1(seq_len * d_k), h_Out_v2(seq_len * d_k);

    for (int i = 0; i < seq_len * d_k; ++i) {
        h_Q[i] = static_cast<float>(rand()) / RAND_MAX;
        h_K[i] = static_cast<float>(rand()) / RAND_MAX;
        h_V[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    // Device memory
    float *d_Q, *d_K, *d_V, *d_S_v1, *d_S_v2, *d_P_v1, *d_P_v2, *d_Out_v1, *d_Out_v2;
    checkCuda(cudaMalloc(&d_Q, qkv_size), "Malloc Q");
    checkCuda(cudaMalloc(&d_K, qkv_size), "Malloc K");
    checkCuda(cudaMalloc(&d_V, qkv_size), "Malloc V");
    checkCuda(cudaMalloc(&d_S_v1, s_size), "Malloc S v1");
    checkCuda(cudaMalloc(&d_S_v2, s_size), "Malloc S v2");
    checkCuda(cudaMalloc(&d_P_v1, s_size), "Malloc P v1");
    checkCuda(cudaMalloc(&d_P_v2, s_size), "Malloc P v2");
    checkCuda(cudaMalloc(&d_Out_v1, qkv_size), "Malloc Out v1");
    checkCuda(cudaMalloc(&d_Out_v2, qkv_size), "Malloc Out v2");

    cudaMemcpy(d_Q, h_Q.data(), qkv_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_K, h_K.data(), qkv_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, h_V.data(), qkv_size, cudaMemcpyHostToDevice);

    // Setup timing metrics
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float t_k1_v1 = 0, t_k1_v2 = 0;
    float t_k2_v1 = 0, t_k2_v2 = 0;
    float t_k3_v1 = 0, t_k3_v2 = 0;

    // -------------------------------------------------------------
    // KERNEL 1: Q * K^T
    // -------------------------------------------------------------
    dim3 t1_v1(16, 16);
    dim3 b1_v1((seq_len + 15) / 16, (seq_len + 15) / 16);
    cudaEventRecord(start);
    qk_dot_product_v1<<<b1_v1, t1_v1>>>(d_Q, d_K, d_S_v1, seq_len, d_k);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&t_k1_v1, start, stop);

    dim3 t1_v2(8, 32); 
    dim3 b1_v2((seq_len + 31) / 32, (seq_len + 31) / 32);
    cudaEventRecord(start);
    qk_tiled_coarsened<<<b1_v2, t1_v2>>>(d_Q, d_K, d_S_v2, seq_len, d_k);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&t_k1_v2, start, stop);

    cudaMemcpy(h_S_v1.data(), d_S_v1, s_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_S_v2.data(), d_S_v2, s_size, cudaMemcpyDeviceToHost);

    // -------------------------------------------------------------
    // KERNEL 2: Softmax
    // -------------------------------------------------------------
    dim3 t2_v1(256);
    dim3 b2_v1((seq_len + 255) / 256);
    cudaEventRecord(start);
    softmax_kernel_v1<<<b2_v1, t2_v1>>>(d_S_v2, d_P_v1, seq_len); // using v2 output as input to be fair
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&t_k2_v1, start, stop);

    dim3 t2_v2(256);
    dim3 b2_v2(seq_len); // 1 block per row
    cudaEventRecord(start);
    softmax_kernel_v2<<<b2_v2, t2_v2>>>(d_S_v2, d_P_v2, seq_len);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&t_k2_v2, start, stop);

    cudaMemcpy(h_P_v1.data(), d_P_v1, s_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_P_v2.data(), d_P_v2, s_size, cudaMemcpyDeviceToHost);

    // -------------------------------------------------------------
    // KERNEL 3: P * V
    // -------------------------------------------------------------
    dim3 t3_v1(16, 16);
    dim3 b3_v1((d_k + 15) / 16, (seq_len + 15) / 16);
    cudaEventRecord(start);
    pv_dot_product_v1<<<b3_v1, t3_v1>>>(d_P_v2, d_V, d_Out_v1, seq_len, d_k);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&t_k3_v1, start, stop);

    dim3 t3_v2(8, 32);
    dim3 b3_v2((d_k + 31) / 32, (seq_len + 31) / 32);
    cudaEventRecord(start);
    pv_tiled_coarsened<<<b3_v2, t3_v2>>>(d_P_v2, d_V, d_Out_v2, seq_len, d_k);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&t_k3_v2, start, stop);

    cudaMemcpy(h_Out_v1.data(), d_Out_v1, qkv_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_Out_v2.data(), d_Out_v2, qkv_size, cudaMemcpyDeviceToHost);

    // -------------------------------------------------------------
    // RESULTS & COMPARISONS
    // -------------------------------------------------------------
    std::cout << "========= CORRECTNESS =========\n";
    std::cout << "K1 (QK^T) Max Difference   : " << getMaxDiff(h_S_v1, h_S_v2) << "\n";
    std::cout << "K2 (Softmax) Max Difference: " << getMaxDiff(h_P_v1, h_P_v2) << "\n";
    std::cout << "K3 (P*V) Max Difference    : " << getMaxDiff(h_Out_v1, h_Out_v2) << "\n\n";

    std::cout << "========= PERFORMANCE (ms) =========\n";
    auto printStats = [](const char* name, float t1, float t2) {
        float speedup = t1 / (t2 + 1e-9); // robust against 0 div
        std::cout << std::left << std::setw(15) << name 
                  << ": v1 = " << std::fixed << std::setprecision(4) << t1 << " ms"
                  << " | v2 = " << t2 << " ms"
                  << " | Speedup = " << speedup << "x\n";
    };

    printStats("K1 (Q*K^T)", t_k1_v1, t_k1_v2);
    printStats("K2 (Softmax)", t_k2_v1, t_k2_v2);
    printStats("K3 (P*V)", t_k3_v1, t_k3_v2);

    float total_v1 = t_k1_v1 + t_k2_v1 + t_k3_v1;
    float total_v2 = t_k1_v2 + t_k2_v2 + t_k3_v2;
    std::cout << "\n------------------------------------\n";
    std::cout << "Total Pipeline (v1): " << total_v1 << " ms\n";
    std::cout << "Total Pipeline (v2): " << total_v2 << " ms\n";
    std::cout << "Overall Speedup    : " << (total_v1 / (total_v2 + 1e-9)) << "x\n";

    // Cleanup
    cudaFree(d_Q); cudaFree(d_K); cudaFree(d_V);
    cudaFree(d_S_v1); cudaFree(d_S_v2);
    cudaFree(d_P_v1); cudaFree(d_P_v2);
    cudaFree(d_Out_v1); cudaFree(d_Out_v2);
    
    return 0;
}

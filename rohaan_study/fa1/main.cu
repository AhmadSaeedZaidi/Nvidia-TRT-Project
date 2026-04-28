#include <iostream>
#include <vector>
#include <fstream>
#include <cmath>
#include <cuda_runtime.h>
#include "attention_fa.cuh"

#define CUDA_CHECK(call) do { \
    cudaError_t _err = (call); \
    if (_err != cudaSuccess) { \
        std::cerr << "CUDA error: " << cudaGetErrorString(_err) << std::endl; \
        return -1; \
    } \
} while(0)

int main() {
    const int N = 128;
    const int D = 64;

    std::vector<float> hQ(N * D), hK(N * D), hV(N * D);
    for (int i = 0; i < N * D; ++i) {
        hQ[i] = sinf((float)i) * 0.12345f;
        hK[i] = cosf((float)(i * 3)) * 0.2345f;
        hV[i] = sinf((float)(i * 5)) * 0.3456f;
    }

    float *dQ = nullptr, *dK = nullptr, *dV = nullptr, *dO = nullptr;
    CUDA_CHECK(cudaMalloc(&dQ, N * D * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dK, N * D * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dV, N * D * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dO, N * D * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(dQ, hQ.data(), N * D * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dK, hK.data(), N * D * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dV, hV.data(), N * D * sizeof(float), cudaMemcpyHostToDevice));

    // timing
    cudaEvent_t start, end;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&end));

    // Launch kernel: 128 blocks (one per query), 1 thread per block
    CUDA_CHECK(cudaEventRecord(start));
    flash_attention_kernel<<<N, 1>>>(dQ, dK, dV, dO, N, D);
    CUDA_CHECK(cudaEventRecord(end));
    CUDA_CHECK(cudaEventSynchronize(end));
    float t_fa1 = 0.0f; CUDA_CHECK(cudaEventElapsedTime(&t_fa1, start, end));

    std::vector<float> hO(N * D);
    CUDA_CHECK(cudaMemcpy(hO.data(), dO, N * D * sizeof(float), cudaMemcpyDeviceToHost));

    // write output to text file
    std::ofstream fout("fa1_out.txt");
    for (size_t i = 0; i < hO.size(); ++i) {
        if (i) fout << ' ';
        fout << hO[i];
    }
    fout << '\n';
    fout.close();

    std::cout << "FA1_total_ms: " << t_fa1 << std::endl;

    cudaEventDestroy(start); cudaEventDestroy(end);
    cudaFree(dQ); cudaFree(dK); cudaFree(dV); cudaFree(dO);
    return 0;
}
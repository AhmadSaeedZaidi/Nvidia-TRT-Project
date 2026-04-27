#include <iostream>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>
#include "attention.cuh"

#define CUDA_CHECK(call) do { \
	cudaError_t e = (call); \
	if (e != cudaSuccess) { \
		std::cerr << "CUDA error: " << cudaGetErrorString(e) << std::endl; \
		return -1; \
	} \
} while(0)

int main() {
	const int N = 8;
	const int D = 16;
	std::vector<float> hQ(N * D), hK(N * D), hV(N * D);
	for (int i = 0; i < N * D; ++i) {
		hQ[i] = (float)(i % 7) / 7.0f;
		hK[i] = (float)((i * 3) % 11) / 11.0f;
		hV[i] = (float)((i * 5) % 13) / 13.0f;
	}

	float *dQ = nullptr, *dK = nullptr, *dV = nullptr, *dS = nullptr, *dO = nullptr;
	CUDA_CHECK(cudaMalloc(&dQ, N * D * sizeof(float)));
	CUDA_CHECK(cudaMalloc(&dK, N * D * sizeof(float)));
	CUDA_CHECK(cudaMalloc(&dV, N * D * sizeof(float)));
	CUDA_CHECK(cudaMalloc(&dS, N * N * sizeof(float)));
	CUDA_CHECK(cudaMalloc(&dO, N * D * sizeof(float)));

	CUDA_CHECK(cudaMemcpy(dQ, hQ.data(), N * D * sizeof(float), cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy(dK, hK.data(), N * D * sizeof(float), cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy(dV, hV.data(), N * D * sizeof(float), cudaMemcpyHostToDevice));

	dim3 b1(16, 16);
	dim3 g1((N + b1.x - 1) / b1.x, (N + b1.y - 1) / b1.y);
	qk_naive<<<g1, b1>>>(dQ, dK, dS, N, D);
	CUDA_CHECK(cudaGetLastError());

	int threads = 128;
	int blocks = (N + threads - 1) / threads;
	softmax_row_naive<<<blocks, threads>>>(dS, N);
	CUDA_CHECK(cudaGetLastError());

	dim3 b2(16, 16);
	dim3 g2((D + b2.x - 1) / b2.x, (N + b2.y - 1) / b2.y);
	pv_naive<<<g2, b2>>>(dS, dV, dO, N, D);
	CUDA_CHECK(cudaGetLastError());

	std::vector<float> hO(N * D);
	CUDA_CHECK(cudaMemcpy(hO.data(), dO, N * D * sizeof(float), cudaMemcpyDeviceToHost));

	std::cout << "output[0..7]: ";
	for (int i = 0; i < std::min(8, N * D); ++i) std::cout << hO[i] << " ";
	std::cout << std::endl;

	cudaFree(dQ);
	cudaFree(dK);
	cudaFree(dV);
	cudaFree(dS);
	cudaFree(dO);

	return 0;
}


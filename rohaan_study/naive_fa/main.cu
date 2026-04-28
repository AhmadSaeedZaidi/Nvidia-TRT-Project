#include <iostream>
#include <vector>
#include <fstream>
#include <cmath>
#include <cuda_runtime.h>
#include "attention.cuh"

#define CUDA_CHECK(call) do { \
	cudaError_t _err = (call); \
	if (_err != cudaSuccess) { \
		std::cerr << "CUDA error: " << cudaGetErrorString(_err) << std::endl; \
		return -1; \
	} \
} while(0)

int main() {
	// Small/medium test size; adjust as needed
	const int N = 128;
	const int D = 64;
	std::vector<float> hQ(N * D), hK(N * D), hV(N * D);
	for (int i = 0; i < N * D; ++i) {
		hQ[i] = sinf((float)i) * 0.12345f;
		hK[i] = cosf((float)(i * 3)) * 0.2345f;
		hV[i] = sinf((float)(i * 5)) * 0.3456f;
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

	// Events for timing
	cudaEvent_t start, end;
	CUDA_CHECK(cudaEventCreate(&start));
	CUDA_CHECK(cudaEventCreate(&end));

	// K1: Q*K^T
	dim3 b1(16, 16);
	dim3 g1((N + b1.x - 1) / b1.x, (N + b1.y - 1) / b1.y);
	CUDA_CHECK(cudaEventRecord(start));
	qk_naive<<<g1, b1>>>(dQ, dK, dS, N, D);
	CUDA_CHECK(cudaEventRecord(end));
	CUDA_CHECK(cudaEventSynchronize(end));
	float t_qk = 0.0f; CUDA_CHECK(cudaEventElapsedTime(&t_qk, start, end));

	// K2: softmax
	int threads = 128;
	int blocks = (N + threads - 1) / threads;
	CUDA_CHECK(cudaEventRecord(start));
	softmax_row_naive<<<blocks, threads>>>(dS, N);
	CUDA_CHECK(cudaEventRecord(end));
	CUDA_CHECK(cudaEventSynchronize(end));
	float t_soft = 0.0f; CUDA_CHECK(cudaEventElapsedTime(&t_soft, start, end));

	// K3: P*V
	dim3 b2(16, 16);
	dim3 g2((D + b2.x - 1) / b2.x, (N + b2.y - 1) / b2.y);
	CUDA_CHECK(cudaEventRecord(start));
	pv_naive<<<g2, b2>>>(dS, dV, dO, N, D);
	CUDA_CHECK(cudaEventRecord(end));
	CUDA_CHECK(cudaEventSynchronize(end));
	float t_pv = 0.0f; CUDA_CHECK(cudaEventElapsedTime(&t_pv, start, end));

	// Copy output
	std::vector<float> hO(N * D);
	CUDA_CHECK(cudaMemcpy(hO.data(), dO, N * D * sizeof(float), cudaMemcpyDeviceToHost));

	// Write output to text file for comparison
	std::ofstream fout("naive_out.txt");
	for (size_t i = 0; i < hO.size(); ++i) {
		if (i) fout << ' ';
		fout << hO[i];
	}
	fout << '\n';
	fout.close();

	// Print timings
	std::cout << "K1_qk_ms: " << t_qk << std::endl;
	std::cout << "K2_softmax_ms: " << t_soft << std::endl;
	std::cout << "K3_pv_ms: " << t_pv << std::endl;
	std::cout << "TOTAL_naive_ms: " << (t_qk + t_soft + t_pv) << std::endl;

	// cleanup
	cudaEventDestroy(start); cudaEventDestroy(end);
	cudaFree(dQ); cudaFree(dK); cudaFree(dV); cudaFree(dS); cudaFree(dO);

	return 0;
}

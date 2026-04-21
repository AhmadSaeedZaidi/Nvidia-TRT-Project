# Nvidia-TRT-Project
Capstone Project for GPU-Programing course, closing real pull request at NVIDIA/TensorRT-LLM

# FP4/E2M1 Fused Multi-Head Attention Sandbox

This repository serves as an isolated CUDA sandbox for developing and profiling missing FP4 (E2M1) Fused Multi-Head Attention (FMHA) kernels for NVIDIA's Blackwell architecture. 

The primary objective is to implement the `OE2m1 ForGen VarSeqQ32/64/128` kernels. This work directly addresses performance regressions caused by multi-CTA fallback paths in downstream inference servers.

**Tracking Issues:**
* Upstream: [NVIDIA/TensorRT-LLM#11620](https://github.com/NVIDIA/TensorRT-LLM/issues/11620)
* Downstream: [vllm-project/vllm#34988](https://github.com/vllm-project/vllm/issues/34988)

## Hardware & Architecture Requirements

This codebase natively targets the NVIDIA Blackwell architecture. 

* **Compute Capability:** `sm_120` (GeForce RTX 50-series) or `sm_100` (Datacenter B200/B300).
* **Memory Constraints:** Kernels must be tuned for `sm_120` shared memory limits (~99 KB per SM) during local testing before being scaled to `sm_100` limits (~227 KB per SM) for enterprise deployment.
* **Requirements:** CUDA Toolkit 12.x+, CMake 3.18+.

## Project Structure

* `src/`: Core CUDA implementations of the `OE2m1` kernels.
* `include/`: Dispatcher logic and shared memory tiling macros.
* `tests/`: High-precision (FP16/BF16) unfused baseline implementations for numerical equivalence testing.
* `main.cu`: The primary test harness generating synthetic $Q$, $K$, and $V$ tensors and calculating output tolerance (MSE/Cosine Similarity).

## Build Instructions

This project uses CMake for rapid compilation independent of the larger TensorRT-LLM build system. The default configuration targets `sm_120`.

```bash
# Clone the sandbox
git clone [https://github.com/AhmadSaeedZaidi/nvidia-trt-project.git](https://github.com/AhmadSaeedZaidi/nvidia-trt-project.git)
cd nvidia-trt-project

# Build the test harness
mkdir build
cd build
cmake .. 
make

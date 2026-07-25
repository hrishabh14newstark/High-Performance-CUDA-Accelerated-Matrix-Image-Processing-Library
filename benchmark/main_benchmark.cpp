#include <iostream>
#include <vector>
#include <chrono>
#include <cuda_runtime.h>
#include "matrix_gemm.cuh"
#include "image_filter.cuh"
#include "reduction.cuh"
#include "cuda_common.h"

int main() {
    int N = 2048;
    size_t bytes = N * N * sizeof(float);

    std::cout << "========================================================\n";
    std::cout << " CUDA Accelerator & GEMM Benchmark Suite (N = " << N << ")\n";
    std::cout << "========================================================\n";

    std::vector<float> h_A(N * N, 1.0f);
    std::vector<float> h_B(N * N, 2.0f);
    std::vector<float> h_C(N * N, 0.0f);

    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, bytes));
    CUDA_CHECK(cudaMalloc(&d_B, bytes));
    CUDA_CHECK(cudaMalloc(&d_C, bytes));

    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), bytes, cudaMemcpyHostToDevice));

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // Benchmark Tiled GEMM
    CUDA_CHECK(cudaEventRecord(start));
    launch_tiled_gemm(d_A, d_B, d_C, N);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    std::cout << "[+] Tiled GEMM Execution Time: " << ms << " ms\n";

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return 0;
}

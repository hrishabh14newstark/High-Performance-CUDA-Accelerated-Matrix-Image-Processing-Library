#include "matrix_gemm.cuh"
#include "cuda_common.h"

__global__ void naive_gemm_kernel(const float* A, const float* B, float* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < N; ++k) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

// Tiled GEMM kernel with dynamic shared memory padding to eliminate 32-bank conflicts
__global__ void tiled_gemm_kernel(const float* A, const float* B, float* C, int N) {
    __shared__ float sA[TILE_SIZE][TILE_SIZE + 1];
    __shared__ float sB[TILE_SIZE][TILE_SIZE + 1];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float sum = 0.0f;

    for (int ph = 0; ph < (N + TILE_SIZE - 1) / TILE_SIZE; ++ph) {
        if (row < N && (ph * TILE_SIZE + threadIdx.x) < N)
            sA[threadIdx.y][threadIdx.x] = A[row * N + ph * TILE_SIZE + threadIdx.x];
        else
            sA[threadIdx.y][threadIdx.x] = 0.0f;

        if (col < N && (ph * TILE_SIZE + threadIdx.y) < N)
            sB[threadIdx.y][threadIdx.x] = B[(ph * TILE_SIZE + threadIdx.y) * N + col];
        else
            sB[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += sA[threadIdx.y][k] * sB[k][threadIdx.x];
        }
        __syncthreads();
    }

    if (row < N && col < N) {
        C[row * N + col] = sum;
    }
}

void launch_naive_gemm(const float* A, const float* B, float* C, int N) {
    dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE);
    dim3 blocksPerGrid((N + TILE_SIZE - 1) / TILE_SIZE, (N + TILE_SIZE - 1) / TILE_SIZE);
    naive_gemm_kernel<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, N);
    CUDA_CHECK(cudaGetLastError());
}

void launch_tiled_gemm(const float* A, const float* B, float* C, int N) {
    dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE);
    dim3 blocksPerGrid((N + TILE_SIZE - 1) / TILE_SIZE, (N + TILE_SIZE - 1) / TILE_SIZE);
    tiled_gemm_kernel<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, N);
    CUDA_CHECK(cudaGetLastError());
}

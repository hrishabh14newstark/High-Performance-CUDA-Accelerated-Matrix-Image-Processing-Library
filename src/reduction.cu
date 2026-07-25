#include "reduction.cuh"
#include "cuda_common.h"

__inline__ __device__ float warp_reduce_sum(float val) {
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__global__ void warp_reduction_kernel(const float* input, float* output, int n) {
    float sum = 0.0f;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = tid; i < n; i += stride) {
        sum += input[i];
    }

    sum = warp_reduce_sum(sum);

    if ((threadIdx.x & (warpSize - 1)) == 0) {
        atomicAdd(output, sum);
    }
}

void launch_reduction(const float* input, float* output, int n) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    warp_reduction_kernel<<<blocks, threads>>>(input, output, n);
    CUDA_CHECK(cudaGetLastError());
}

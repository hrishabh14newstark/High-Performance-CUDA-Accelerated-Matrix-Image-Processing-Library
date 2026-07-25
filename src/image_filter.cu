#include "image_filter.cuh"
#include "cuda_common.h"

__global__ void image_filter_kernel(const float* input, float* output, const float* __restrict__ filter, int width, int height) {
    __shared__ float sData[BLOCK_DIM + 2 * RADIUS][BLOCK_DIM + 2 * RADIUS];

    int gRow = blockIdx.y * BLOCK_DIM + threadIdx.y;
    int gCol = blockIdx.x * BLOCK_DIM + threadIdx.x;

    int lRow = threadIdx.y + RADIUS;
    int lCol = threadIdx.x + RADIUS;

    // Load main tile element
    if (gRow < height && gCol < width)
        sData[lRow][lCol] = input[gRow * width + gCol];
    else
        sData[lRow][lCol] = 0.0f;

    // Load halo regions
    if (threadIdx.y < RADIUS) {
        // Top Halo
        sData[lRow - RADIUS][lCol] = (gRow >= RADIUS) ? input[(gRow - RADIUS) * width + gCol] : 0.0f;
        // Bottom Halo
        sData[lRow + BLOCK_DIM][lCol] = (gRow + BLOCK_DIM < height) ? input[(gRow + BLOCK_DIM) * width + gCol] : 0.0f;
    }
    if (threadIdx.x < RADIUS) {
        // Left Halo
        sData[lRow][lCol - RADIUS] = (gCol >= RADIUS) ? input[gRow * width + (gCol - RADIUS)] : 0.0f;
        // Right Halo
        sData[lRow][lCol + BLOCK_DIM] = (gCol + BLOCK_DIM < width) ? input[gRow * width + (gCol + BLOCK_DIM)] : 0.0f;
    }

    __syncthreads();

    if (gRow < height && gCol < width) {
        float val = 0.0f;
        #pragma unroll
        for (int fRow = -RADIUS; fRow <= RADIUS; ++fRow) {
            #pragma unroll
            for (int fCol = -RADIUS; fCol <= RADIUS; ++fCol) {
                val += sData[lRow + fRow][lCol + fCol] * filter[(fRow + RADIUS) * FILTER_SIZE + (fCol + RADIUS)];
            }
        }
        output[gRow * width + gCol] = val;
    }
}

void launch_image_filter(const float* input, float* output, const float* filter, int width, int height) {
    dim3 threadsPerBlock(BLOCK_DIM, BLOCK_DIM);
    dim3 blocksPerGrid((width + BLOCK_DIM - 1) / BLOCK_DIM, (height + BLOCK_DIM - 1) / BLOCK_DIM);
    image_filter_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, filter, width, height);
    CUDA_CHECK(cudaGetLastError());
}

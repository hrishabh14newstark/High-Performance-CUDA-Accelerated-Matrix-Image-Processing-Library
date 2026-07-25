#ifndef MATRIX_GEMM_CUH
#define MATRIX_GEMM_CUH

#define TILE_SIZE 16

void launch_naive_gemm(const float* A, const float* B, float* C, int N);
void launch_tiled_gemm(const float* A, const float* B, float* C, int N);

#endif // MATRIX_GEMM_CUH

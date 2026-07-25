#include <omp.h>

void cpu_filter_omp(const float* input, float* output, const float* filter, int width, int height, int kSize) {
    int radius = kSize / 2;
    #pragma omp parallel for collapse(2)
    for (int r = 0; r < height; ++r) {
        for (int c = 0; c < width; ++c) {
            float sum = 0.0f;
            for (int fr = -radius; fr <= radius; ++fr) {
                for (int fc = -radius; fc <= radius; ++fc) {
                    int inR = r + fr;
                    int inC = c + fc;
                    if (inR >= 0 && inR < height && inC >= 0 && inC < width) {
                        sum += input[inR * width + inC] * filter[(fr + radius) * kSize + (fc + radius)];
                    }
                }
            }
            output[r * width + c] = sum;
        }
    }
}

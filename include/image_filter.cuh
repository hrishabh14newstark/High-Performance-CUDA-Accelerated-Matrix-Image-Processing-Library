#ifndef IMAGE_FILTER_CUH
#define IMAGE_FILTER_CUH

#define FILTER_SIZE 5
#define RADIUS (FILTER_SIZE / 2)
#define BLOCK_DIM 16

void launch_image_filter(const float* input, float* output, const float* filter, int width, int height);

#endif // IMAGE_FILTER_CUH

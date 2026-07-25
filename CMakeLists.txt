cmake_minimum_required(VERSION 3.18)
project(HighPerformanceCudaLibrary LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# Target NVIDIA Architectures (sm_70, sm_80, sm_86, sm_90)
set(CMAKE_CUDA_ARCHITECTURES 70 80 86 90)

find_package(OpenMP REQUIRED)

include_directories(include)

add_library(cuda_compute_lib
    src/matrix_gemm.cu
    src/image_filter.cu
    src/reduction.cu
)

add_library(cpu_baseline_lib
    cpu_baseline/cpu_gemm.cpp
    cpu_baseline/cpu_filter.cpp
)
target_link_libraries(cpu_baseline_lib PRIVATE OpenMP::OpenMP_CXX)

add_executable(cuda_perf_benchmark benchmark/main_benchmark.cpp)
target_link_libraries(cuda_perf_benchmark PRIVATE cuda_compute_lib cpu_baseline_lib)

set_target_properties(cuda_perf_benchmark PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

#include "add_kernel.h"
#include <cuda_runtime.h>
#include <cstdio>

__global__ void add_kernel_impl(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

void launch_add_kernel(const float* a, const float* b, float* c, int n) {
    int threads_per_block = 256;
    int blocks_per_grid = (n + threads_per_block - 1) / threads_per_block;
    add_kernel_impl<<<blocks_per_grid, threads_per_block>>>(a, b, c, n);

    // It's good practice to check for errors after launching a kernel,
    // especially during development.
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        // In a real application, you might want to throw an exception here.
        fprintf(stderr, "CUDA kernel launch failed: %s\n", cudaGetErrorString(err));
    }
} 
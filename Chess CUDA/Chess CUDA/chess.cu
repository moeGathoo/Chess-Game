#include <iostream>
#include <chrono>
#include <math.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

// Kernel function to add the elements of two arrays
__global__ void add(int n, float* x, float* y)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < n)
        y[index] = x[index] * 10;
}

int main(void)
{
    int N = 20;
    float* x, * y;
    float* host = new float[N * sizeof(float)];
    float* out = new float[N * sizeof(float)];

    // Allocate Unified Memory – accessible from CPU or GPU
    cudaMalloc(& x, N * sizeof(float));
    cudaMalloc(& y, N * sizeof(float));
    std::cout << N << std::endl;

    // initialize x and y arrays on the host
    for (int i = 0; i < N; i++) {
        host[i] = i;
    }

    cudaMemcpy(&x, &host, N * sizeof(float), cudaMemcpyHostToDevice);

    // Run kernel on 1M elements on the GPU
    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;
    cudaEvent_t begin, end;
    cudaEventCreate(&begin);
    cudaEventCreate(&end);

    cudaEventRecord(begin, 0);
    add << <numBlocks, blockSize>> > (N, x, y);
    cudaEventRecord(end, 0);
    cudaEventSynchronize(end);
    cudaMemcpy(&out, &y, N * sizeof(float), cudaMemcpyDeviceToHost);
    float time = 0;
    cudaEventElapsedTime(&time, begin, end);
    for (int i = 0; i < N; i++)
        std::cout << out[i] << std::endl;
    std::cout << "Kernel run time: " << time << std::endl;

    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < N; i++) {
        out[i] = host[i] * 10;
    }
    auto stop = std::chrono::high_resolution_clock::now();
    long long timeTaken = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count();
    std::cout << "CPU Time: " << timeTaken << std::endl;

    // Wait for GPU to finish before accessing on host
    cudaDeviceSynchronize();

    // Check for errors (all values should be 3.0f)
    float maxError = 0.0f;
    std::cout << "added" << std::endl;
    //std::cout << "Max error: " << maxError << std::endl;

    // Free memory
    cudaFree(x);
    cudaFree(y);
    free(host);

    return 0;
}
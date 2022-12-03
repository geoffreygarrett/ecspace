/// Introduction
/// - https://developer.nvidia.com/blog/easy-introduction-cuda-c-and-c/
///
#include "helper.cuh"
#include <iostream>
#include <math.h>
#include <stdio.h>


/// \brief CUDA Kernel function to add the elements of two arrays on the GPU
/// \notes
/// 1. __global__ functions are known as `kernels`, and code that runs on the device is known as `device code`.
/// 2. The __global__ keyword tells the compiler that this function will be called from the host, but executed on the device.
__global__
void add(int n, float *x, float *y)
{
    for (int i = 0; i < n; i++)
        y[i] = x[i] + y[i];
}

/// Memory Allocation in CUDA
/// \long_description
/// To compute on the GPU, I need to allocate memory accessible by the GPU.
/// Unified Memory in CUDA makes this easy by providing a single memory space
/// accessible by all GPUs and CPUs in your system. To allocate data in unified
/// memory, call cudaMallocManaged(), which returns a pointer that you can access
/// from host (CPU) code or device (GPU) code. To free the data, just pass
/// the pointer to cudaFree().
///
/// I just need to replace the calls to new in the code above with calls
/// to cudaMallocManaged(), and replace calls to delete [] with calls to cudaFree.


int main(void)
{

    int N = 1<<20;
    float *x, *y;

    // Prints helper device info stuff
    device_info();

    // Allocate Unified Memory – accessible from CPU or GPU
    cudaMallocManaged(&x, N*sizeof(float));
    cudaMallocManaged(&y, N*sizeof(float));

    // initialize x and y arrays on the host
    for (int i = 0; i < N; i++) {
        x[i] = 1.0f;
        y[i] = 2.0f;
    }

    cudaError_t errSync  = cudaGetLastError();
    cudaError_t errAsync = cudaDeviceSynchronize();
    if (errSync != cudaSuccess)
        printf("[1] Sync kernel error: %s\n", cudaGetErrorString(errSync));
    if (errAsync != cudaSuccess)
        printf("[1] Async kernel error: %s\n", cudaGetErrorString(errAsync));

    // Run kernel on 1M elements on the GPU
    add<<<1, 1>>>(N, x, y);

    /// \error_checking
    /// This code checks for both synchronous and asynchronous errors.
    /// Invalid execution configuration parameters, e.g. too many threads
    /// per thread block, are reflected in the value of errSync returned
    /// by cudaGetLastError(). Asynchronous errors that occur on the device
    /// after control is returned to the host, such as out-of-bounds memory
    /// accesses, require a synchronization mechanism such as
    /// cudaDeviceSynchronize(), which blocks the host thread until all
    /// previously issued commands have completed. Any asynchronous error
    /// is returned by cudaDeviceSynchronize(). We can also check for
    /// asynchronous errors and reset the runtime error state by modifying
    /// the last statement to call cudaGetLastError().
    cudaError_t errSync2  = cudaGetLastError();
    cudaError_t errAsync2 = cudaDeviceSynchronize();
    if (errSync2 != cudaSuccess)
        printf("[2] Sync kernel error: %s\n", cudaGetErrorString(errSync2));
    if (errAsync2 != cudaSuccess)
        printf("[2] Async kernel error: %s\n", cudaGetErrorString(errAsync2));

    /// \note
    /// Device synchronization is expensive, because it causes the entire
    /// device to wait, destroying any potential for concurrency at that
    /// point in your program. So use it with care. Typically, I use
    /// preprocessor macros to insert asynchronous error checking only
    /// in debug builds of my code, and not in release builds.

    // Wait for GPU to finish before accessing on host
    cudaDeviceSynchronize();

    // Check for errors (all values should be 3.0f)
    float maxError = 0.0f;
    for (int i = 0; i < N; i++)
        maxError = fmax(maxError, fabs(y[i]-3.0f));
    std::cout << "Max error: " << maxError << std::endl;

    // Free memory
    cudaFree(x);
    cudaFree(y);

    return 0;
}
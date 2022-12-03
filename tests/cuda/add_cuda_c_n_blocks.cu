#include <iostream>
#include <math.h>

/// \brief CUDA Kernel function to add the elements of two arrays on the GPU
/// \notes
/// 1. __global__ functions are known as `kernels`, and code that runs on the device is known as `device code`.
/// 2. The __global__ keyword tells the compiler that this function will be called from the host, but executed on the device.
__global__
        void add(int n, float *x, float *y)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < n; i += stride)
        y[i] = x[i] + y[i];
}

/// \brief Memory Allocation in CUDA
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
    // check if GPU is available
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount == 0) {
        std::cerr << "Error: no devices supporting CUDA.\n";
        exit(EXIT_FAILURE);
    }
    int N = 1<<20;
    float *x, *y;

    // Allocate Unified Memory – accessible from CPU or GPU
    cudaMallocManaged(&x, N*sizeof(float));
    cudaMallocManaged(&y, N*sizeof(float));

    // initialize x and y arrays on the host
    for (int i = 0; i < N; i++) {
        x[i] = 1.0f;
        y[i] = 2.0f;
    }

    // Run kernel on 1M elements on the GPU
    add<<<1, 256>>>(N, x, y);

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
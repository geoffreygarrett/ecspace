//#include "timer.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "test.cuh"

#define BLOCK_SIZE 256

__global__ void cuda_hello(){
    printf("Hello World from GPU!\n");
}

void wrapper (){
    cuda_hello<<<1,1>>>();
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
}
// Prints helper device info stuff
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

void device_info(){
    int nDevices;
    cudaGetDeviceCount(&nDevices);
    for (int i = 0; i < nDevices; i++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        printf("Device Number: %d\n", i);
        printf("  Device name: %s\n", prop.name);
        printf("  Memory Clock Rate (KHz): %d\n",
               prop.memoryClockRate);
        printf("  Memory Bus Width (bits): %d\n",
               prop.memoryBusWidth);
        printf("  Peak Memory Bandwidth (GB/s): %f\n\n",
               2.0*prop.memoryClockRate*(prop.memoryBusWidth/8)/1.0e6);
    }

    cudaError_t err = cudaGetDeviceCount(&nDevices);
    if (err != cudaSuccess) printf("%s\n", cudaGetErrorString(err));
}


//__host__ __device__ void gravitational_attraction(float4 *p, float dt, int n) {
//    int i = blockDim.x * blockIdx.x + threadIdx.x;
//    if (i < n) {
//        float Fx = 0.0f;
//        float Fy = 0.0f;
//        float Fz = 0.0f;
//
//        for (int j = 0; j < n; j++) {
//            float dx = p[j].x - p[i].x;
//            float dy = p[j].y - p[i].y;
//            float dz = p[j].z - p[i].z;
//            float distSqr = dx * dx + dy * dy + dz * dz + SOFTENING;
//            float invDist = rsqrtf(distSqr);
//            float invDist3 = invDist * invDist * invDist;
//
//            Fx += dx * invDist3;
//            Fy += dy * invDist3;
//            Fz += dz * invDist3;
//        }
//
//        v[i].x += dt * Fx;
//        v[i].y += dt * Fy;
//        v[i].z += dt * Fz;
//    }
//}

//__global__ void bodyForce(float4 *p, float4 *v, float dt, int n) {
//    int i = blockDim.x * blockIdx.x + threadIdx.x;
//    if (i < n) {
//        float Fx = 0.0f;
//        float Fy = 0.0f;
//        float Fz = 0.0f;
//
//        for (int j = 0; j < n; j++) {
//            float dx = p[j].x - p[i].x;
//            float dy = p[j].y - p[i].y;
//            float dz = p[j].z - p[i].z;
//            float distSqr = dx * dx + dy * dy + dz * dz + SOFTENING;
//            float invDist = rsqrtf(distSqr);
//            float invDist3 = invDist * invDist * invDist;
//
//            Fx += dx * invDist3;
//            Fy += dy * invDist3;
//            Fz += dz * invDist3;
//        }
//
//        v[i].x += dt * Fx;
//        v[i].y += dt * Fy;
//        v[i].z += dt * Fz;
//    }
//}

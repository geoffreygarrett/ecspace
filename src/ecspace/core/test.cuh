
#ifndef HELPER_CUH
#define HELPER_CUH

#include <iostream>
#include <math.h>
#include <stdio.h>

/// Error Handling in CUDA
/// - https://developer.nvidia.com/blog/how-query-device-properties-and-handle-errors-cuda-cc/
/// - https://leimao.github.io/blog/Proper-CUDA-Error-Checking/
/// - https://stackoverflow.com/questions/14038589/what-is-the-canonical-way-to-check-for-errors-using-the-cuda-runtime-api
///
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true);

void device_info();
void wrapper ();


#endif // HELPER_CUH
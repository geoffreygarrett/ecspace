# Where do these tests originate from?

The following tests are from a blogpost by [NVIDIA](https://devblogs.nvidia.com/parallelforall/using-cuda-warp-level-primitives/).
- [`add_cpu.cpp`](add_cpu.cpp)
- [`add_cuda_a_1_thread.cu`](add_cuda_a_1_thread.cu)
- [`add_cuda_b_1_block.cu`](add_cuda_b_1_block.cu)
- [`add_cuda_c_n_blocks.cu`](add_cuda_c_n_blocks.cu)

- [helper.cuh](helper.cuh) is a helper file that contains some CUDA helper functions.


nbody: https://github.com/harrism/mini-nbody
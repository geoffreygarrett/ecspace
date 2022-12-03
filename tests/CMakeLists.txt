
# add tests under cuda/
add_executable(test_add_cpu ${CMAKE_CURRENT_SOURCE_DIR}/cuda/add_cpu.cpp)
add_test(NAME test_add_cpu COMMAND test_add_cpu)

add_executable(test_nbody_cpu ${CMAKE_CURRENT_SOURCE_DIR}/cuda/nbody_cpu.cpp)
add_test(NAME test_nbody_cpu COMMAND test_nbody_cpu)

if (ECSPACE_CUDA)
    set(CUDA_TESTS
            add_cuda_a_1_thread
            add_cuda_b_1_block
            add_cuda_c_n_blocks
            nbody_cuda_a_soa
            )
    foreach (test ${CUDA_TESTS})
        add_executable("test_${test}" ${CMAKE_CURRENT_SOURCE_DIR}/cuda/${test}.cu)
        set_target_properties(test_${test} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
        add_test(NAME ${test} COMMAND ${test})
    endforeach ()
endif ()
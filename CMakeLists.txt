cmake_minimum_required(VERSION 3.23)

project(ECSpace VERSION 0.1.0 LANGUAGES CXX)

option(ECSPACE_BUILD_TESTS "Build tests" ON)
option(ECSPACE_BUILD_EXAMPLES "Build examples" ON)
option(ECSPACE_CUDA "Build with CUDA support" ON)
option(ECSPACE_VERBOSE_MAKEFILE "Verbose makefile" OFF)
option(ECSPACE_MATPLOTLIB "Support matplotlib-cpp" ON)

add_definitions(-DECSPACE_MATPLOTLIB=1)

# temporary sticking this here, should be optional, but for now it's required
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if (ECSPACE_VERBOSE_MAKEFILE)
    set(CMAKE_VERBOSE_MAKEFILE ON)
endif()

if (ECSPACE_CUDA)
    # temporary sticking this here, should be optional, but for now it's required
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)

    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        #  found this in https://github.com/stephen-sorley/cuda-cmake
        # there's more knowledge there for configuring cuda.
        set(CMAKE_CUDA_ARCHITECTURES
                50 # Maxwell
                60 # Pascal
                70 # Volta
                75 # Turing
                )
    endif()


    find_package(CUDAToolkit REQUIRED)
    message(STATUS "CUDAToolkit_FOUND: ${CUDAToolkit_FOUND}")
    message(STATUS "CUDAToolkit version: ${CUDAToolkit_VERSION}")
    message(STATUS "CUDAToolkit_BIN_DIR: ${CUDAToolkit_BIN_DIR}")
    message(STATUS "CUDAToolkit_INCLUDE_DIRS: ${CUDAToolkit_INCLUDE_DIRS}")
    message(STATUS "CUDAToolkit_LIBRARY_DIR: ${CUDAToolkit_LIBRARY_DIR}")
    message(STATUS "CUDAToolkit_LIBRARY_ROOT: ${CUDAToolkit_LIBRARY_ROOT}")
    message(STATUS "CUDAToolkit_TARGET_DIR: ${CUDAToolkit_TARGET_DIR}")
    message(STATUS "CUDAToolkit_NVCC_EXECUTABLE: ${CUDAToolkit_NVCC_EXECUTABLE}")
    message(STATUS "CUDAToolkit_ROOT: ${CUDAToolkit_ROOT}")
    set(CMAKE_CUDA_COMPILER ${CUDAToolkit_BIN_DIR}/nvcc)
    execute_process(COMMAND ${CMAKE_CUDA_COMPILER} --version OUTPUT_VARIABLE CUDA_COMPILER_VERSION)

    include(CheckLanguage)
    check_language(CUDA)
    enable_language(CUDA)

    # if debug
    if (CMAKE_BUILD_TYPE STREQUAL "Debug" OR CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")
        set(DEVICE_DEBUGGING -G0) # CUDA_DEVICE_DEBUGGING
    endif()
endif()

if (ECSPACE_MATPLOTLIB)
    set(CMAKE_PREFIX_PATH "/home/ggarrett/miniconda3/envs/tudat-entt")
    # get python
    find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
    # pythonlibs
    find_package(PythonLibs REQUIRED)
    # include
    include_directories(${Python3_INCLUDE_DIRS})
    include_directories(${PYTHON_INCLUDE_DIRS})


    # get numpy
    execute_process(COMMAND ${Python3_EXECUTABLE} -c "import numpy; print(numpy.get_include())"
            OUTPUT_VARIABLE NUMPY_INCLUDE_DIR
            OUTPUT_STRIP_TRAILING_WHITESPACE)

    # include
    include_directories(${NUMPY_INCLUDE_DIR})
endif()

if (ECSPACE_BUILD_TESTS)
#
#     use google tests
#     https://github.com/google/googletest.git
    enable_testing()
    add_subdirectory(tests)
endif ()

# add directory
#add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/vendor/entt)
#add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/vendor/eigen)
add_library(ecspace
        src/ecspace/core/accelerations.h
        src/ecspace/core/components.h
        src/ecspace/core/systems.h
        src/ecspace/core/consts.h
        src/ecspace/core/termination.h
#        .graveyard/systems/animation.h
        src/ecspace/core/conditions.h
        )

# set target properties
set_target_properties(ecspace PROPERTIES
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED ON
        CXX_EXTENSIONS OFF
        LINKER_LANGUAGE CXX
        )

# ADD tbb
find_package(TBB REQUIRED)
include_directories(${TBB_INCLUDE_DIRS})
target_link_libraries(ecspace TBB::tbb)

# important setting for accessing the eigen library
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")
find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})

# find and link boost ijn
#find_package(Boost REQUIRED)
#include_directories(${Boost_INCLUDE_DIRS})
#target_link_libraries(ecspace ${Boost_LIBRARIES})

# add opencv
#find_package(OpenCV REQUIRED)
#include_directories(${OpenCV_INCLUDE_DIRS})
#target_link_libraries(ecspace ${OpenCV_LIBS})

target_include_directories(ecspace PUBLIC ${Boost_INCLUDE_DIRS})
target_include_directories(ecspace INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/vendor/entt/src)
target_include_directories(ecspace INTERFACE Eigen3::Eigen)
target_include_directories(ecspace INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/vendor/matplotlib-cpp)
#target_include_directories(ecspace INTERFACE /home/ggarrett/CLionProjects/ecspace/vendor/cv-plot/CvPlot/inc)
target_link_libraries(ecspace ${PYTHON_LIBRARIES})


set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_RELEASE} -g")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp") # compile for multiple cores

# NOTE: `-03` Absolutely needed for Eigen. Orders of magnitude faster.
#       Eigen is a heavily templated library, and the compiler needs to see all the
#       code in order to optimise it.
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")

add_executable(ecspace_earth_asteroids examples/earth_asteroids.cpp)
target_link_libraries(ecspace_earth_asteroids ecspace)

add_executable(tudat_entt_test_1 .graveyard/test_1.cpp)
target_link_libraries(tudat_entt_test_1 ecspace)
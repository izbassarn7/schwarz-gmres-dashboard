cmake_minimum_required(VERSION 3.24)
project(schwarz_gmres LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)
find_package(CUDAToolkit REQUIRED)

find_library(METIS_LIBRARY metis)
find_path(METIS_INCLUDE_DIR metis.h PATH_SUFFIXES metis)
if(NOT METIS_LIBRARY OR NOT METIS_INCLUDE_DIR)
    message(FATAL_ERROR "METIS not found. Install with: apt install libmetis-dev")
endif()

find_package(LAPACK REQUIRED)

option(USE_PETSC "Build PETSc comparison baselines" OFF)
option(USE_TRILINOS "Build Trilinos comparison baselines" OFF)

if(USE_PETSC)
    include(cmake/FindPETSc.cmake)
endif()
if(USE_TRILINOS)
    include(cmake/FindTrilinos.cmake)
endif()

add_subdirectory(src)

option(BUILD_TESTS "Build unit tests" ON)
if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

option(BUILD_BENCH "Build benchmarks" ON)
if(BUILD_BENCH)
    add_subdirectory(bench)
endif()

add_library(schwarz_core STATIC
    core/gmres.cpp
    core/fgmres.cpp
    gpu/local_solve_kernel.cu
    gpu/gmres_gpu.cu
)
target_include_directories(schwarz_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(schwarz_core PUBLIC
    MPI::MPI_CXX
    OpenMP::OpenMP_CXX
    CUDA::cudart
    CUDA::cusparse
    CUDA::cublas
    CUDA::cusolver
    ${METIS_LIBRARY}
    ${LAPACK_LIBRARIES}
)
target_include_directories(schwarz_core PUBLIC ${METIS_INCLUDE_DIR})

# Activate #ifdef USE_CUDA guards in source files when CUDA is available
target_compile_definitions(schwarz_core PUBLIC USE_CUDA)

set_target_properties(schwarz_core PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
)

add_executable(schwarz_solve main.cpp)
target_link_libraries(schwarz_solve PRIVATE schwarz_core)

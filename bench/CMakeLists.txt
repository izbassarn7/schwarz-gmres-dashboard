add_executable(strong_scaling strong_scaling.cpp)
target_link_libraries(strong_scaling PRIVATE schwarz_core)

add_executable(weak_scaling weak_scaling.cpp)
target_link_libraries(weak_scaling PRIVATE schwarz_core)

if(USE_CUDA)
    add_executable(gpu_scaling gpu_scaling.cpp)
    target_link_libraries(gpu_scaling PRIVATE schwarz_core CUDA::cudart CUDA::cublas CUDA::cusparse)
endif()

if(USE_PETSC)
    add_executable(petsc_baseline petsc_baseline.cpp)
    target_link_libraries(petsc_baseline PRIVATE schwarz_core PETSc::PETSc)
endif()

if(USE_TRILINOS)
    add_executable(trilinos_baseline trilinos_baseline.cpp)
    target_link_libraries(trilinos_baseline PRIVATE schwarz_core
        ${Trilinos_LIBRARIES})
    target_include_directories(trilinos_baseline PRIVATE
        ${Trilinos_INCLUDE_DIRS})
endif()

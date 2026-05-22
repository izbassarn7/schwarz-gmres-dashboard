add_executable(test_gmres test_gmres.cpp)
target_link_libraries(test_gmres PRIVATE schwarz_core)
add_test(NAME test_gmres COMMAND test_gmres)

add_executable(test_schwarz test_schwarz.cpp)
target_link_libraries(test_schwarz PRIVATE schwarz_core)
add_test(NAME test_schwarz COMMAND test_schwarz)

add_executable(test_two_level test_two_level.cpp)
target_link_libraries(test_two_level PRIVATE schwarz_core)
add_test(NAME test_two_level COMMAND test_two_level)

add_executable(test_gpu_kernels test_gpu_kernels.cu)
target_link_libraries(test_gpu_kernels PRIVATE schwarz_core)
set_target_properties(test_gpu_kernels PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
add_test(NAME test_gpu_kernels COMMAND test_gpu_kernels)

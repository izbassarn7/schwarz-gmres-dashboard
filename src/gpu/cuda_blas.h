#pragma once

#include "cuda_common.h"

namespace schwarz {
namespace gpu {

// cuBLAS wrappers for common BLAS-1 operations on device vectors

inline double cublas_dot(cublasHandle_t handle, int n,
                         const double* d_x, const double* d_y) {
    double result;
    CUBLAS_CHECK(cublasDdot(handle, n, d_x, 1, d_y, 1, &result));
    return result;
}

inline double cublas_nrm2(cublasHandle_t handle, int n, const double* d_x) {
    double result;
    CUBLAS_CHECK(cublasDnrm2(handle, n, d_x, 1, &result));
    return result;
}

// y = alpha * x + y
inline void cublas_axpy(cublasHandle_t handle, int n,
                        double alpha, const double* d_x, double* d_y) {
    CUBLAS_CHECK(cublasDaxpy(handle, n, &alpha, d_x, 1, d_y, 1));
}

// x = alpha * x
inline void cublas_scal(cublasHandle_t handle, int n,
                        double alpha, double* d_x) {
    CUBLAS_CHECK(cublasDscal(handle, n, &alpha, d_x, 1));
}

// y = x
inline void cublas_copy(cublasHandle_t handle, int n,
                        const double* d_x, double* d_y) {
    CUBLAS_CHECK(cublasDcopy(handle, n, d_x, 1, d_y, 1));
}

}  // namespace gpu
}  // namespace schwarz

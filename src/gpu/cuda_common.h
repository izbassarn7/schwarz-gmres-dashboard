#pragma once

#include <cuda_runtime.h>
#include <cusparse.h>
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <stdexcept>
#include <string>

namespace schwarz {
namespace gpu {

#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) \
        throw std::runtime_error(std::string("CUDA error: ") + \
            cudaGetErrorString(err) + " at " + __FILE__ + ":" + \
            std::to_string(__LINE__)); \
} while(0)

#define CUSPARSE_CHECK(call) do { \
    cusparseStatus_t err = (call); \
    if (err != CUSPARSE_STATUS_SUCCESS) \
        throw std::runtime_error(std::string("cuSPARSE error at ") + \
            __FILE__ + ":" + std::to_string(__LINE__)); \
} while(0)

#define CUBLAS_CHECK(call) do { \
    cublasStatus_t err = (call); \
    if (err != CUBLAS_STATUS_SUCCESS) \
        throw std::runtime_error(std::string("cuBLAS error at ") + \
            __FILE__ + ":" + std::to_string(__LINE__)); \
} while(0)

#define CUSOLVER_CHECK(call) do { \
    cusolverStatus_t err = (call); \
    if (err != CUSOLVER_STATUS_SUCCESS) \
        throw std::runtime_error(std::string("cuSOLVER error at ") + \
            __FILE__ + ":" + std::to_string(__LINE__)); \
} while(0)

struct CudaHandles {
    cusparseHandle_t cusparse = nullptr;
    cublasHandle_t cublas = nullptr;
    cusolverDnHandle_t cusolver = nullptr;
    cudaStream_t stream = nullptr;

    void init() {
        CUDA_CHECK(cudaStreamCreate(&stream));
        CUSPARSE_CHECK(cusparseCreate(&cusparse));
        CUSPARSE_CHECK(cusparseSetStream(cusparse, stream));
        CUBLAS_CHECK(cublasCreate(&cublas));
        CUBLAS_CHECK(cublasSetStream(cublas, stream));
        CUSOLVER_CHECK(cusolverDnCreate(&cusolver));
        CUSOLVER_CHECK(cusolverDnSetStream(cusolver, stream));
    }

    void destroy() {
        if (cusolver) { cusolverDnDestroy(cusolver); cusolver = nullptr; }
        if (cublas) { cublasDestroy(cublas); cublas = nullptr; }
        if (cusparse) { cusparseDestroy(cusparse); cusparse = nullptr; }
        if (stream) { cudaStreamDestroy(stream); stream = nullptr; }
    }

    ~CudaHandles() { destroy(); }
};

// Device memory RAII wrapper
template<typename T>
struct DeviceBuffer {
    T* ptr = nullptr;
    size_t count = 0;

    DeviceBuffer() = default;
    explicit DeviceBuffer(size_t n) : count(n) {
        CUDA_CHECK(cudaMalloc(&ptr, n * sizeof(T)));
    }
    ~DeviceBuffer() { if (ptr) cudaFree(ptr); }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    DeviceBuffer(DeviceBuffer&& o) noexcept : ptr(o.ptr), count(o.count) {
        o.ptr = nullptr; o.count = 0;
    }
    DeviceBuffer& operator=(DeviceBuffer&& o) noexcept {
        if (ptr) cudaFree(ptr);
        ptr = o.ptr; count = o.count;
        o.ptr = nullptr; o.count = 0;
        return *this;
    }

    void allocate(size_t n) {
        if (ptr) cudaFree(ptr);
        count = n;
        CUDA_CHECK(cudaMalloc(&ptr, n * sizeof(T)));
    }

    void upload(const T* host_data, size_t n, cudaStream_t s = nullptr) {
        CUDA_CHECK(cudaMemcpyAsync(ptr, host_data, n * sizeof(T),
                                   cudaMemcpyHostToDevice, s));
    }

    void download(T* host_data, size_t n, cudaStream_t s = nullptr) const {
        CUDA_CHECK(cudaMemcpyAsync(host_data, ptr, n * sizeof(T),
                                   cudaMemcpyDeviceToHost, s));
    }
};

}  // namespace gpu
}  // namespace schwarz

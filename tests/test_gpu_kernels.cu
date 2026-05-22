#include "../src/core/sparse_matrix.h"
#include "../src/core/gmres.h"
#include "../src/core/vector.h"
#include "../src/mesh/poisson.h"
#include "../src/precond/ilu0.h"
#include "../src/gpu/cuda_common.h"
#include "../src/gpu/cuda_sparse.h"
#include "../src/gpu/cuda_blas.h"
#include "../src/gpu/asm_gpu.h"

#include <iostream>
#include <cmath>
#include <vector>

using namespace schwarz;

static int test_count = 0;
static int pass_count = 0;

#define CHECK(cond, msg) do { \
    test_count++; \
    if (!(cond)) { \
        std::cerr << "FAIL: " << msg << " (line " << __LINE__ << ")\n"; \
    } else { \
        pass_count++; \
        std::cout << "PASS: " << msg << "\n"; \
    } \
} while(0)

static void test_device_spmv() {
    CSRMatrix A = make_poisson_2d(10, 10);
    int n = A.nrows;

    gpu::CudaHandles handles;
    handles.init();

    gpu::DeviceCSRMatrix dA;
    dA.upload(A, handles);

    std::vector<double> h_x(n, 1.0), h_y_cpu(n), h_y_gpu(n);
    A.spmv(h_x, h_y_cpu);

    gpu::DeviceBuffer<double> d_x(n), d_y(n);
    d_x.upload(h_x.data(), n, handles.stream);

    dA.spmv(handles, 1.0, d_x.ptr, 0.0, d_y.ptr);

    d_y.download(h_y_gpu.data(), n, handles.stream);
    cudaStreamSynchronize(handles.stream);

    double err = 0;
    for (int i = 0; i < n; ++i)
        err += std::abs(h_y_cpu[i] - h_y_gpu[i]);

    CHECK(err < 1e-10, "GPU SpMV matches CPU SpMV");
}

static void test_cublas_ops() {
    int n = 100;
    gpu::CudaHandles handles;
    handles.init();

    std::vector<double> h_a(n), h_b(n);
    for (int i = 0; i < n; ++i) { h_a[i] = i + 1.0; h_b[i] = 1.0; }

    gpu::DeviceBuffer<double> d_a(n), d_b(n);
    d_a.upload(h_a.data(), n, handles.stream);
    d_b.upload(h_b.data(), n, handles.stream);
    cudaStreamSynchronize(handles.stream);

    double d = gpu::cublas_dot(handles.cublas, n, d_a.ptr, d_b.ptr);
    double expected = n * (n + 1.0) / 2.0;
    CHECK(std::abs(d - expected) < 1e-6, "cuBLAS dot product correct");

    double nrm = gpu::cublas_nrm2(handles.cublas, n, d_b.ptr);
    CHECK(std::abs(nrm - std::sqrt((double)n)) < 1e-6, "cuBLAS nrm2 correct");
}

static void test_asm_gpu_precond() {
    CSRMatrix A = make_poisson_2d(10, 10);
    int n = A.nrows;

    gpu::CudaHandles handles;
    handles.init();

    gpu::ASMGPUPrecond pc(4, 1, false);
    pc.setup(A, handles);

    std::vector<double> x(n, 1.0);
    std::vector<double> y(n, 0.0);
    pc.apply(x.data(), y.data(), n);

    double nrm = 0;
    for (int i = 0; i < n; ++i) nrm += y[i] * y[i];
    nrm = std::sqrt(nrm);

    CHECK(nrm > 0, "GPU ASM preconditioner produces nonzero output");
}

int main() {
    std::cout << "=== GPU Kernel Tests ===\n";

    int device_count = 0;
    cudaGetDeviceCount(&device_count);
    if (device_count == 0) {
        std::cout << "No CUDA devices found. Skipping GPU tests.\n";
        return 0;
    }
    std::cout << "Found " << device_count << " CUDA device(s)\n";

    test_device_spmv();
    test_cublas_ops();
    test_asm_gpu_precond();

    std::cout << "\n" << pass_count << "/" << test_count << " tests passed.\n";
    return (pass_count == test_count) ? 0 : 1;
}

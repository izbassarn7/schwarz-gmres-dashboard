#pragma once

#include "cuda_common.h"
#include "cuda_sparse.h"
#include "cuda_blas.h"
#include "asm_gpu.h"
#include "../core/sparse_matrix.h"

#include <functional>
#include <vector>

namespace schwarz {
namespace gpu {

struct GPUGMRESResult {
    int iterations = 0;
    double residual_norm = 0.0;
    bool converged = false;
    double time_solve_s = 0.0;
};

using PrecondApplyFn = std::function<void(const double* d_x, double* d_y, int n)>;

// Declared in gmres_gpu.cu
GPUGMRESResult gmres_gpu(
    CudaHandles& handles,
    DeviceCSRMatrix& d_A,
    const double* d_b,
    double* d_x,
    int n,
    PrecondApplyFn precond_apply,
    int restart = 30,
    int max_iter = 1000,
    double tol = 1e-10);

// Convenience: solve Ax=b entirely on GPU with ASM/RAS preconditioner.
// A is on host; uploaded internally. Solution returned in h_x.
inline GPUGMRESResult solve_gpu(
    const CSRMatrix& A,
    const std::vector<double>& b,
    std::vector<double>& x,
    int nparts = 4,
    int overlap = 1,
    bool restricted = true,
    int restart = 30,
    int max_iter = 1000,
    double tol = 1e-10)
{
    int n = A.nrows;
    CudaHandles handles;
    handles.init();

    DeviceCSRMatrix d_A;
    d_A.upload(A, handles);

    DeviceBuffer<double> d_b_buf(n), d_x_buf(n);
    d_b_buf.upload(b.data(), n, handles.stream);
    d_x_buf.upload(x.data(), n, handles.stream);

    ASMGPUPrecond pc(nparts, overlap, restricted);
    pc.setup(A, handles);
    auto precond_fn = pc.make_precond_fn();

    auto result = gmres_gpu(handles, d_A, d_b_buf.ptr, d_x_buf.ptr, n,
                            precond_fn, restart, max_iter, tol);

    x.resize(n);
    d_x_buf.download(x.data(), n, handles.stream);
    CUDA_CHECK(cudaStreamSynchronize(handles.stream));

    return result;
}

}  // namespace gpu
}  // namespace schwarz

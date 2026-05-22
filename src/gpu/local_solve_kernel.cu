#include "cuda_common.h"
#include <cuda_runtime.h>

namespace schwarz {
namespace gpu {

// Batched ILU(0) triangular solve kernel.
// Each block handles one subdomain. The ILU factors are stored
// in a single concatenated CSR with per-subdomain offsets.
//
// This kernel is suitable when subdomains are small enough
// to be solved by a single block (< ~1024 DOFs).
__global__ void batched_ilu_solve_kernel(
    const int* __restrict__ row_ptr,
    const int* __restrict__ col_idx,
    const double* __restrict__ lu_vals,
    const int* __restrict__ diag_ptr,
    const double* __restrict__ rhs,
    double* __restrict__ sol,
    const int* __restrict__ sub_offsets,  // sub_offsets[s], sub_offsets[s+1]
    int n_subdomains)
{
    int sid = blockIdx.x;
    if (sid >= n_subdomains) return;

    int start = sub_offsets[sid];
    int end   = sub_offsets[sid + 1];
    int n_local = end - start;

    // Use shared memory for the local solution vector
    extern __shared__ double shared_buf[];
    double* tmp = shared_buf;

    int tid = threadIdx.x;

    // Copy RHS into shared memory
    for (int i = tid; i < n_local; i += blockDim.x)
        tmp[i] = rhs[start + i];
    __syncthreads();

    // Forward solve (L): sequential within block
    if (tid == 0) {
        for (int i = 0; i < n_local; ++i) {
            int gi = start + i;
            double sum = tmp[i];
            for (int j = row_ptr[gi]; j < diag_ptr[gi]; ++j) {
                int col = col_idx[j];
                if (col >= start && col < end)
                    sum -= lu_vals[j] * tmp[col - start];
            }
            tmp[i] = sum;
        }
    }
    __syncthreads();

    // Backward solve (U): sequential within block
    if (tid == 0) {
        for (int i = n_local - 1; i >= 0; --i) {
            int gi = start + i;
            double sum = tmp[i];
            for (int j = diag_ptr[gi] + 1; j < row_ptr[gi + 1]; ++j) {
                int col = col_idx[j];
                if (col >= start && col < end)
                    sum -= lu_vals[j] * tmp[col - start];
            }
            tmp[i] = sum / lu_vals[diag_ptr[gi]];
        }
    }
    __syncthreads();

    // Write result back
    for (int i = tid; i < n_local; i += blockDim.x)
        sol[start + i] = tmp[i];
}

void launch_batched_ilu_solve(
    const int* d_row_ptr,
    const int* d_col_idx,
    const double* d_lu_vals,
    const int* d_diag_ptr,
    const double* d_rhs,
    double* d_sol,
    const int* d_sub_offsets,
    int n_subdomains,
    int max_local_n,
    cudaStream_t stream)
{
    int threads = std::min(256, max_local_n);
    size_t shared_size = max_local_n * sizeof(double);
    batched_ilu_solve_kernel<<<n_subdomains, threads, shared_size, stream>>>(
        d_row_ptr, d_col_idx, d_lu_vals, d_diag_ptr,
        d_rhs, d_sol, d_sub_offsets, n_subdomains);
    CUDA_CHECK(cudaGetLastError());
}

}  // namespace gpu
}  // namespace schwarz

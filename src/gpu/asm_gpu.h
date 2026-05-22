#pragma once

#include "cuda_common.h"
#include "cuda_sparse.h"
#include "cuda_blas.h"
#include "../core/sparse_matrix.h"
#include "../decomp/domain_decomp.h"
#include "../precond/preconditioner.h"
#include "../precond/ilu0.h"

#include <vector>
#include <memory>
#include <set>
#include <functional>

namespace schwarz {
namespace gpu {

// Forward declaration
void launch_batched_ilu_solve(
    const int* d_row_ptr, const int* d_col_idx,
    const double* d_lu_vals, const int* d_diag_ptr,
    const double* d_rhs, double* d_sol,
    const int* d_sub_offsets, int n_subdomains,
    int max_local_n, cudaStream_t stream);

// GPU-offloaded ASM/RAS preconditioner.
// All subdomain ILU factors are concatenated and uploaded to GPU.
// The batched kernel solves all subdomains in parallel on the GPU.
class ASMGPUPrecond {
public:
    ASMGPUPrecond(int nparts, int overlap, bool restricted = false)
        : nparts_(nparts), overlap_(overlap), restricted_(restricted) {}

    void setup(const CSRMatrix& A, CudaHandles& handles) {
        n_ = A.nrows;
        dd_ = build_decomposition(A, nparts_, overlap_);
        handles_ = &handles;

        std::vector<int> all_row_ptr, all_col_idx, all_diag_ptr;
        std::vector<double> all_lu_vals;
        sub_offsets_.resize(nparts_ + 1);
        sub_offsets_[0] = 0;
        max_local_n_ = 0;

        // Compute subdomain offsets
        for (int s = 0; s < nparts_; ++s) {
            int ns = static_cast<int>(dd_.subdomain_dofs[s].size());
            sub_offsets_[s + 1] = sub_offsets_[s] + ns;
            if (ns > max_local_n_) max_local_n_ = ns;
        }

        // Build concatenated ILU factors with global row/col offsets
        int nnz_offset = 0;
        for (int s = 0; s < nparts_; ++s) {
            CSRMatrix A_local = A.extract_submatrix(dd_.subdomain_dofs[s]);
            ILU0Precond ilu;
            ilu.setup(A_local);

            int ns = A_local.nrows;
            for (int i = 0; i < ns; ++i) {
                all_row_ptr.push_back(ilu.row_ptr()[i] + nnz_offset);
                all_diag_ptr.push_back(ilu.diag_ptr()[i] + nnz_offset);
            }
            for (int j = 0; j < ilu.row_ptr()[ns]; ++j) {
                all_col_idx.push_back(ilu.col_idx()[j] + sub_offsets_[s]);
                all_lu_vals.push_back(ilu.lu_vals()[j]);
            }
            nnz_offset += ilu.row_ptr()[ns];
        }
        // Final sentinel
        all_row_ptr.push_back(nnz_offset);

        int total_n = sub_offsets_[nparts_];

        d_row_ptr_.allocate(all_row_ptr.size());
        d_col_idx_.allocate(all_col_idx.size());
        d_lu_vals_.allocate(all_lu_vals.size());
        d_diag_ptr_.allocate(all_diag_ptr.size());
        d_sub_offsets_.allocate(nparts_ + 1);
        d_rhs_.allocate(total_n);
        d_sol_.allocate(total_n);

        d_row_ptr_.upload(all_row_ptr.data(), all_row_ptr.size(), handles.stream);
        d_col_idx_.upload(all_col_idx.data(), all_col_idx.size(), handles.stream);
        d_lu_vals_.upload(all_lu_vals.data(), all_lu_vals.size(), handles.stream);
        d_diag_ptr_.upload(all_diag_ptr.data(), all_diag_ptr.size(), handles.stream);
        d_sub_offsets_.upload(sub_offsets_.data(), nparts_ + 1, handles.stream);

        // Build owned_local_indices for RAS mode
        if (restricted_) {
            owned_local_indices_.resize(nparts_);
            for (int s = 0; s < nparts_; ++s) {
                std::set<int> owned_set(
                    dd_.owned_dofs[s].begin(), dd_.owned_dofs[s].end());
                const auto& all_dofs = dd_.subdomain_dofs[s];
                for (int i = 0; i < static_cast<int>(all_dofs.size()); ++i) {
                    if (owned_set.count(all_dofs[i]))
                        owned_local_indices_[s].push_back(i);
                }
            }
        }

        CUDA_CHECK(cudaStreamSynchronize(handles.stream));
    }

    // Apply on host pointers (download/upload around GPU kernel)
    void apply(const double* x, double* y, int n) const {
        std::fill(y, y + n, 0.0);

        int total_local = sub_offsets_[nparts_];
        std::vector<double> h_rhs(total_local, 0.0);

        // Scatter global residual to concatenated local arrays
        for (int s = 0; s < nparts_; ++s) {
            const auto& dofs = dd_.subdomain_dofs[s];
            int offset = sub_offsets_[s];
            for (int i = 0; i < static_cast<int>(dofs.size()); ++i)
                h_rhs[offset + i] = x[dofs[i]];
        }

        d_rhs_.upload(h_rhs.data(), total_local, handles_->stream);

        launch_batched_ilu_solve(
            d_row_ptr_.ptr, d_col_idx_.ptr, d_lu_vals_.ptr, d_diag_ptr_.ptr,
            d_rhs_.ptr, d_sol_.ptr, d_sub_offsets_.ptr,
            nparts_, max_local_n_, handles_->stream);

        std::vector<double> h_sol(total_local);
        d_sol_.download(h_sol.data(), total_local, handles_->stream);
        CUDA_CHECK(cudaStreamSynchronize(handles_->stream));

        // Gather: prolongate back to global
        for (int s = 0; s < nparts_; ++s) {
            const auto& dofs = dd_.subdomain_dofs[s];
            int offset = sub_offsets_[s];

            if (restricted_) {
                for (int li : owned_local_indices_[s])
                    y[dofs[li]] = h_sol[offset + li];
            } else {
                for (int i = 0; i < static_cast<int>(dofs.size()); ++i)
                    y[dofs[i]] += h_sol[offset + i];
            }
        }
    }

    const DomainDecomposition& decomposition() const { return dd_; }
    int global_n() const { return n_; }

    // Device-resident apply: input/output are device pointers.
    // Stages through host for scatter/gather, kernel runs on GPU.
    // For use as PrecondApplyFn in gmres_gpu.
    void apply_device(const double* d_x, double* d_y, int n) const {
        std::vector<double> h_x(n), h_y(n);
        CUDA_CHECK(cudaMemcpyAsync(h_x.data(), d_x, n * sizeof(double),
                                   cudaMemcpyDeviceToHost, handles_->stream));
        CUDA_CHECK(cudaStreamSynchronize(handles_->stream));

        apply(h_x.data(), h_y.data(), n);

        CUDA_CHECK(cudaMemcpyAsync(d_y, h_y.data(), n * sizeof(double),
                                   cudaMemcpyHostToDevice, handles_->stream));
        CUDA_CHECK(cudaStreamSynchronize(handles_->stream));
    }

    // Create a PrecondApplyFn lambda for gmres_gpu
    std::function<void(const double*, double*, int)> make_precond_fn() {
        return [this](const double* d_x, double* d_y, int n) {
            this->apply_device(d_x, d_y, n);
        };
    }

private:
    int nparts_, overlap_;
    bool restricted_;
    int n_ = 0;
    int max_local_n_ = 0;
    DomainDecomposition dd_;
    CudaHandles* handles_ = nullptr;
    std::vector<int> sub_offsets_;

    mutable DeviceBuffer<int> d_row_ptr_, d_col_idx_, d_diag_ptr_, d_sub_offsets_;
    mutable DeviceBuffer<double> d_lu_vals_, d_rhs_, d_sol_;
    std::vector<std::vector<int>> owned_local_indices_;
};

}  // namespace gpu
}  // namespace schwarz

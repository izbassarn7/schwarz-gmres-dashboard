#pragma once

#include "preconditioner.h"
#include "distributed_schwarz.h"
#include "coarse_solve.h"
#include "../core/sparse_matrix.h"

#include <mpi.h>
#include <vector>
#include <memory>
#include <numeric>
#include <algorithm>
#include <cmath>

namespace schwarz {

// Distributed two-level additive Schwarz preconditioner.
//
// M^{-1} = fine-level correction + coarse-level correction
//
// The fine level is distributed across ranks using DistributedSchwarzPrecond.
// The coarse level aggregates each local subdomain into one coarse DOF.
// Coarse DOFs are numbered globally: rank r owns coarse IDs [r*local_nparts, (r+1)*local_nparts).
//
// Coarse operator is assembled globally via MPI_Allreduce.
class DistributedTwoLevelSchwarzPrecond : public Preconditioner {
public:
    DistributedTwoLevelSchwarzPrecond(int local_nparts, int overlap,
                                       DistSchwarzType type = DistSchwarzType::RAS,
                                       MPI_Comm comm = MPI_COMM_WORLD)
        : local_nparts_(local_nparts), overlap_(overlap), type_(type), comm_(comm)
    {
        MPI_Comm_rank(comm_, &rank_);
        MPI_Comm_size(comm_, &nranks_);
    }

    // Setup from distributed matrix: local CSR + owned row count + global mapping
    void setup(const CSRMatrix& A_local, int n_owned,
               const std::vector<int>& local_to_global)
    {
        n_owned_ = n_owned;
        n_local_ = A_local.nrows;
        local_to_global_ = local_to_global;

        // Setup fine-level preconditioner
        fine_precond_ = std::make_unique<DistributedSchwarzPrecond>(local_nparts_, overlap_, type_);
        fine_precond_->setup(A_local, n_owned, local_to_global);

        // Extract subdomain decomposition from fine preconditioner
        // We need access to the subdomains to build aggregates
        extract_subdomains_from_fine(A_local, n_owned);

        // Build coarse operator
        build_coarse_operator(A_local);

        // Setup coarse solver
        coarse_solver_ = std::make_unique<CoarseSolver>();
        coarse_solver_->setup(A0_, n_coarse_);
    }

    // Fallback: single-rank setup (delegates to setup with identity local_to_global)
    void setup(const CSRMatrix& A) override {
        std::vector<int> ltg(A.nrows);
        std::iota(ltg.begin(), ltg.end(), 0);
        setup(A, A.nrows, ltg);
    }

    void apply(const double* x, double* y, int n) const override {
        // Fine-level correction
        std::vector<double> z_fine(n, 0.0);
        fine_precond_->apply(x, z_fine.data(), n);

        // Restrict to coarse: r0_local[s] = sum of x values in subdomain s
        std::vector<double> r0_local(local_nparts_, 0.0);
        for (int s = 0; s < local_nparts_; ++s) {
            for (int i : subdomain_owned_dofs_[s]) {
                r0_local[s] += x[i];
            }
        }

        // Gather all coarse RHS values globally
        std::vector<double> r0_global(n_coarse_, 0.0);
        MPI_Allgather(r0_local.data(), local_nparts_, MPI_DOUBLE,
                      r0_global.data(), local_nparts_, MPI_DOUBLE, comm_);

        // Solve coarse system on rank 0, broadcast result
        std::vector<double> y0_global(n_coarse_, 0.0);
        if (rank_ == 0) {
            coarse_solver_->solve(r0_global.data(), y0_global.data());
        }
        MPI_Bcast(y0_global.data(), n_coarse_, MPI_DOUBLE, 0, comm_);

        // Prolongate: contribution from coarse level
        // For each owned DOF i in local subdomain s:
        // z_coarse[i] = y0_global[rank * local_nparts + s] / |agg_s|
        std::vector<double> z_coarse(n, 0.0);
        for (int s = 0; s < local_nparts_; ++s) {
            int coarse_id = rank_ * local_nparts_ + s;
            double coarse_val = y0_global[coarse_id];
            int agg_size = static_cast<int>(subdomain_owned_dofs_[s].size());
            if (agg_size > 0) {
                for (int i : subdomain_owned_dofs_[s]) {
                    z_coarse[i] = coarse_val / static_cast<double>(agg_size);
                }
            }
        }

        // Additive combination
        for (int i = 0; i < n; ++i) {
            y[i] = z_fine[i] + z_coarse[i];
        }
    }

private:
    int local_nparts_;
    int overlap_;
    DistSchwarzType type_;
    MPI_Comm comm_;
    int rank_ = 0;
    int nranks_ = 0;

    int n_owned_ = 0;
    int n_local_ = 0;
    std::vector<int> local_to_global_;

    // Fine preconditioner
    std::unique_ptr<DistributedSchwarzPrecond> fine_precond_;

    // Subdomain decomposition (same as in fine preconditioner)
    // subdomain_owned_dofs_[s] = owned DOFs in local subdomain s
    std::vector<std::vector<int>> subdomain_owned_dofs_;

    // Coarse problem
    int n_coarse_ = 0;  // nranks * local_nparts
    std::vector<double> A0_;  // Dense coarse operator (flattened, row-major)
    std::unique_ptr<CoarseSolver> coarse_solver_;

    // Extract subdomain structure from fine preconditioner by re-partitioning
    void extract_subdomains_from_fine(const CSRMatrix& A_local, int n_owned)
    {
        subdomain_owned_dofs_.clear();
        subdomain_owned_dofs_.resize(local_nparts_);

        // Re-partition owned rows using the same logic as DistributedSchwarzPrecond
        if (local_nparts_ <= 1) {
            for (int i = 0; i < n_owned; ++i)
                subdomain_owned_dofs_[0].push_back(i);
        } else {
            // Build owned submatrix for METIS
            CSRMatrix A_owned;
            A_owned.nrows = n_owned;
            A_owned.ncols = n_owned;
            A_owned.row_ptr.resize(n_owned + 1);
            std::vector<int> ci;
            std::vector<double> cv;

            for (int i = 0; i < n_owned; ++i) {
                A_owned.row_ptr[i] = static_cast<int>(ci.size());
                for (int j = A_local.row_ptr[i]; j < A_local.row_ptr[i + 1]; ++j) {
                    int col = A_local.col_idx[j];
                    if (col < n_owned) {
                        ci.push_back(col);
                        cv.push_back(A_local.vals[j]);
                    }
                }
            }
            A_owned.row_ptr[n_owned] = static_cast<int>(ci.size());
            A_owned.col_idx = ci;
            A_owned.vals = cv;

            auto part = partition_metis(A_owned, local_nparts_);
            for (int i = 0; i < n_owned; ++i) {
                subdomain_owned_dofs_[part[i]].push_back(i);
            }
        }

        std::sort(subdomain_owned_dofs_.begin(), subdomain_owned_dofs_.end());
        for (auto& dofs : subdomain_owned_dofs_)
            std::sort(dofs.begin(), dofs.end());
    }

    // Build coarse operator via assembly and global reduction
    void build_coarse_operator(const CSRMatrix& A_local)
    {
        n_coarse_ = nranks_ * local_nparts_;

        // Local contribution to coarse operator (dense)
        std::vector<double> A0_local(local_nparts_ * local_nparts_, 0.0);

        // Accumulate A[i][j] into A0[s1][s2] for i in agg_s1, j in agg_s2
        for (int s1 = 0; s1 < local_nparts_; ++s1) {
            for (int i : subdomain_owned_dofs_[s1]) {
                for (int j = A_local.row_ptr[i]; j < A_local.row_ptr[i + 1]; ++j) {
                    int col = A_local.col_idx[j];
                    double val = A_local.vals[j];

                    // Find which subdomain owns column col
                    // If col < n_owned, it's an owned column; otherwise it's a ghost
                    // We need to determine its subdomain within this rank
                    int s2 = -1;
                    if (col < n_owned_) {
                        // Owned column: find its subdomain
                        for (int ss = 0; ss < local_nparts_; ++ss) {
                            if (std::find(subdomain_owned_dofs_[ss].begin(),
                                         subdomain_owned_dofs_[ss].end(), col)
                                != subdomain_owned_dofs_[ss].end()) {
                                s2 = ss;
                                break;
                            }
                        }
                    } else {
                        // Ghost column: skip (coarse coupling only through owned columns)
                        continue;
                    }

                    if (s2 >= 0) {
                        A0_local[s1 * local_nparts_ + s2] += val;
                    }
                }
            }
        }

        // Global reduction: MPI_Allreduce to sum across all ranks
        std::vector<double> A0_global(n_coarse_ * n_coarse_, 0.0);

        // Scatter A0_local into the global matrix at the correct position
        for (int s1 = 0; s1 < local_nparts_; ++s1) {
            for (int s2 = 0; s2 < local_nparts_; ++s2) {
                int global_s1 = rank_ * local_nparts_ + s1;
                int global_s2 = rank_ * local_nparts_ + s2;
                A0_global[global_s1 * n_coarse_ + global_s2] =
                    A0_local[s1 * local_nparts_ + s2];
            }
        }

        // Reduce: sum all local contributions
        std::vector<double> A0_reduced(n_coarse_ * n_coarse_, 0.0);
        MPI_Allreduce(A0_global.data(), A0_reduced.data(),
                      n_coarse_ * n_coarse_, MPI_DOUBLE, MPI_SUM, comm_);

        A0_ = A0_reduced;
    }
};

}  // namespace schwarz

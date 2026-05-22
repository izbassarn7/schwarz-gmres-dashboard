#pragma once

#include "preconditioner.h"
#include "ilu0.h"
#include "../core/sparse_matrix.h"
#include "../core/distributed_matrix.h"
#include "../decomp/domain_decomp.h"

#include <mpi.h>
#include <vector>
#include <memory>
#include <set>

namespace schwarz {

enum class DistSchwarzType { ASM, RAS };

// MPI-distributed Schwarz preconditioner.
// Each rank owns one or more subdomains from its local rows.
// Local solves are ILU(0). Overlap DOFs come from halo exchange.
class DistributedSchwarzPrecond : public Preconditioner {
public:
    DistributedSchwarzPrecond(int local_nparts, int overlap,
                              DistSchwarzType type = DistSchwarzType::RAS)
        : local_nparts_(local_nparts), overlap_(overlap), type_(type) {}

    // Setup from the local portion of the distributed matrix.
    // The local CSR must include ghost columns.
    void setup(const CSRMatrix& A_local, int n_owned,
               const std::vector<int>& local_to_global)
    {
        n_owned_ = n_owned;
        n_local_ = A_local.nrows;

        // Partition local rows into local_nparts_ subdomains
        if (local_nparts_ <= 1) {
            // Single subdomain = all local rows
            subdomain_dofs_.resize(1);
            owned_sub_dofs_.resize(1);
            for (int i = 0; i < n_owned_; ++i) {
                subdomain_dofs_[0].push_back(i);
                owned_sub_dofs_[0].push_back(i);
            }
        } else {
            // Use METIS on the local subgraph (owned rows only)
            CSRMatrix A_owned;
            A_owned.nrows = n_owned_;
            A_owned.ncols = n_owned_;
            A_owned.row_ptr.resize(n_owned_ + 1);
            std::vector<int> ci;
            std::vector<double> cv;

            for (int i = 0; i < n_owned_; ++i) {
                A_owned.row_ptr[i] = static_cast<int>(ci.size());
                for (int j = A_local.row_ptr[i]; j < A_local.row_ptr[i + 1]; ++j) {
                    int col = A_local.col_idx[j];
                    if (col < n_owned_) {
                        ci.push_back(col);
                        cv.push_back(A_local.vals[j]);
                    }
                }
            }
            A_owned.row_ptr[n_owned_] = static_cast<int>(ci.size());
            A_owned.col_idx = ci;
            A_owned.vals = cv;

            auto part = partition_metis(A_owned, local_nparts_);

            subdomain_dofs_.resize(local_nparts_);
            owned_sub_dofs_.resize(local_nparts_);
            for (int i = 0; i < n_owned_; ++i) {
                subdomain_dofs_[part[i]].push_back(i);
                owned_sub_dofs_[part[i]].push_back(i);
            }
        }

        // Expand subdomains with overlap layers.
        // Only expand into owned DOFs (indices < n_owned_).
        // Ghost DOFs (>= n_owned_) belong to other ranks and have no local rows.
        for (int s = 0; s < static_cast<int>(subdomain_dofs_.size()); ++s) {
            std::set<int> dofs(subdomain_dofs_[s].begin(),
                               subdomain_dofs_[s].end());
            for (int layer = 0; layer < overlap_; ++layer) {
                std::set<int> new_dofs;
                for (int idx : dofs) {
                    if (idx >= n_owned_) continue;  // skip ghost rows
                    for (int j = A_local.row_ptr[idx]; j < A_local.row_ptr[idx + 1]; ++j) {
                        int col = A_local.col_idx[j];
                        // Only add owned DOFs to overlap (col < n_owned_)
                        if (col < n_owned_ && dofs.find(col) == dofs.end())
                            new_dofs.insert(col);
                    }
                }
                dofs.insert(new_dofs.begin(), new_dofs.end());
            }
            subdomain_dofs_[s].assign(dofs.begin(), dofs.end());
            std::sort(subdomain_dofs_[s].begin(), subdomain_dofs_[s].end());
        }

        // Build local ILU(0) solvers per subdomain
        int nsub = static_cast<int>(subdomain_dofs_.size());
        local_solvers_.resize(nsub);
        for (int s = 0; s < nsub; ++s) {
            CSRMatrix A_sub = A_local.extract_submatrix(subdomain_dofs_[s]);
            local_solvers_[s] = std::make_unique<ILU0Precond>();
            local_solvers_[s]->setup(A_sub);
        }

        // For RAS: identify which local indices in each subdomain are "owned"
        if (type_ == DistSchwarzType::RAS) {
            owned_local_idx_.resize(nsub);
            for (int s = 0; s < nsub; ++s) {
                std::set<int> owned_set(owned_sub_dofs_[s].begin(),
                                         owned_sub_dofs_[s].end());
                for (int i = 0; i < static_cast<int>(subdomain_dofs_[s].size()); ++i) {
                    if (owned_set.count(subdomain_dofs_[s][i]))
                        owned_local_idx_[s].push_back(i);
                }
            }
        }
    }

    // Preconditioner interface: not used directly for distributed setup
    void setup(const CSRMatrix& A) override {
        // For non-distributed use, fall back to single-rank setup
        std::vector<int> ltg(A.nrows);
        std::iota(ltg.begin(), ltg.end(), 0);
        setup(A, A.nrows, ltg);
    }

    void apply(const double* x, double* y, int n) const override {
        std::fill(y, y + n, 0.0);

        int nsub = static_cast<int>(subdomain_dofs_.size());

        #pragma omp parallel for schedule(dynamic)
        for (int s = 0; s < nsub; ++s) {
            const auto& dofs = subdomain_dofs_[s];
            int ns = static_cast<int>(dofs.size());

            std::vector<double> r_local(ns);
            std::vector<double> z_local(ns);

            // Ghost DOFs (index >= n) are read as 0 — we don't have the
            // residual value for them in the distributed GMRES vector.
            for (int i = 0; i < ns; ++i)
                r_local[i] = (dofs[i] < n) ? x[dofs[i]] : 0.0;

            local_solvers_[s]->apply(r_local.data(), z_local.data(), ns);

            if (type_ == DistSchwarzType::RAS) {
                for (int li : owned_local_idx_[s]) {
                    int gi = dofs[li];
                    if (gi < n) y[gi] = z_local[li];
                }
            } else {
                #pragma omp critical
                {
                    for (int i = 0; i < ns; ++i)
                        if (dofs[i] < n) y[dofs[i]] += z_local[i];
                }
            }
        }
    }

private:
    int local_nparts_;
    int overlap_;
    DistSchwarzType type_;
    int n_owned_ = 0;
    int n_local_ = 0;

    std::vector<std::vector<int>> subdomain_dofs_;
    std::vector<std::vector<int>> owned_sub_dofs_;
    std::vector<std::unique_ptr<ILU0Precond>> local_solvers_;
    std::vector<std::vector<int>> owned_local_idx_;
};

}  // namespace schwarz

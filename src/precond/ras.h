#pragma once

#include "preconditioner.h"
#include "ilu0.h"
#include "../core/sparse_matrix.h"
#include "../decomp/domain_decomp.h"
#include <vector>
#include <memory>
#include <set>

namespace schwarz {

// Restricted Additive Schwarz (RAS) preconditioner.
// z = sum_i Rtilde_i^T * A_i^{-1} * R_i * r
// Rtilde_i only injects owned DOFs (no overlap contribution).
class RASPrecond : public Preconditioner {
public:
    RASPrecond(int nparts, int overlap)
        : nparts_(nparts), overlap_(overlap) {}

    void setup(const CSRMatrix& A) override {
        n_ = A.nrows;
        dd_ = build_decomposition(A, nparts_, overlap_);

        local_solvers_.resize(nparts_);
        local_matrices_.resize(nparts_);
        owned_local_indices_.resize(nparts_);

        for (int s = 0; s < nparts_; ++s) {
            local_matrices_[s] = A.extract_submatrix(dd_.subdomain_dofs[s]);
            local_solvers_[s] = std::make_unique<ILU0Precond>();
            local_solvers_[s]->setup(local_matrices_[s]);

            // Map owned DOFs to local indices within subdomain_dofs[s]
            std::set<int> owned_set(
                dd_.owned_dofs[s].begin(), dd_.owned_dofs[s].end());
            const auto& all_dofs = dd_.subdomain_dofs[s];
            for (int i = 0; i < static_cast<int>(all_dofs.size()); ++i) {
                if (owned_set.count(all_dofs[i]))
                    owned_local_indices_[s].push_back(i);
            }
        }
    }

    void apply(const double* x, double* y, int n) const override {
        std::fill(y, y + n, 0.0);

        #pragma omp parallel for schedule(dynamic)
        for (int s = 0; s < nparts_; ++s) {
            const auto& dofs = dd_.subdomain_dofs[s];
            int ns = static_cast<int>(dofs.size());

            std::vector<double> r_local(ns);
            std::vector<double> z_local(ns);

            // R_i * r: restrict with overlap
            for (int i = 0; i < ns; ++i)
                r_local[i] = x[dofs[i]];

            // A_i^{-1} * r_local
            local_solvers_[s]->apply(r_local.data(), z_local.data(), ns);

            // Rtilde_i^T: prolongate only owned DOFs (restricted)
            for (int li : owned_local_indices_[s]) {
                int gi = dofs[li];
                // Each owned DOF belongs to exactly one subdomain, no race
                y[gi] = z_local[li];
            }
        }
    }

    const DomainDecomposition& decomposition() const { return dd_; }

protected:
    int nparts_;
    int overlap_;
    int n_ = 0;
    DomainDecomposition dd_;
    std::vector<CSRMatrix> local_matrices_;
    std::vector<std::unique_ptr<ILU0Precond>> local_solvers_;
    std::vector<std::vector<int>> owned_local_indices_;
};

}  // namespace schwarz

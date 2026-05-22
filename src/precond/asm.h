#pragma once

#include "preconditioner.h"
#include "ilu0.h"
#include "../core/sparse_matrix.h"
#include "../decomp/domain_decomp.h"
#include <vector>
#include <memory>

namespace schwarz {

// Additive Schwarz Method (ASM) preconditioner.
// z = sum_i R_i^T * A_i^{-1} * R_i * r
// Local solves use ILU(0).
class ASMPrecond : public Preconditioner {
public:
    ASMPrecond(int nparts, int overlap)
        : nparts_(nparts), overlap_(overlap) {}

    void setup(const CSRMatrix& A) override {
        n_ = A.nrows;
        dd_ = build_decomposition(A, nparts_, overlap_);

        local_solvers_.resize(nparts_);
        local_matrices_.resize(nparts_);

        for (int s = 0; s < nparts_; ++s) {
            local_matrices_[s] = A.extract_submatrix(dd_.subdomain_dofs[s]);
            local_solvers_[s] = std::make_unique<ILU0Precond>();
            local_solvers_[s]->setup(local_matrices_[s]);
        }
    }

    void apply(const double* x, double* y, int n) const override {
        std::fill(y, y + n, 0.0);

        // Can be parallelized with OpenMP
        #pragma omp parallel for schedule(dynamic)
        for (int s = 0; s < nparts_; ++s) {
            const auto& dofs = dd_.subdomain_dofs[s];
            int ns = static_cast<int>(dofs.size());

            std::vector<double> r_local(ns);
            std::vector<double> z_local(ns);

            // R_i * r: restrict global residual to subdomain
            for (int i = 0; i < ns; ++i)
                r_local[i] = x[dofs[i]];

            // A_i^{-1} * r_local
            local_solvers_[s]->apply(r_local.data(), z_local.data(), ns);

            // R_i^T * z_local: prolongate to global (ASM adds overlap contributions)
            #pragma omp critical
            {
                for (int i = 0; i < ns; ++i)
                    y[dofs[i]] += z_local[i];
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
};

}  // namespace schwarz

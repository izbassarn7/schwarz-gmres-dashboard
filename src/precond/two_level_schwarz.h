#pragma once

#include "preconditioner.h"
#include "asm.h"
#include "ras.h"
#include "coarse_space.h"
#include "coarse_solve.h"
#include "../core/sparse_matrix.h"
#include "../decomp/domain_decomp.h"

#include <vector>
#include <memory>

namespace schwarz {

// Two-level additive Schwarz preconditioner:
//   M^{-1} = sum_i R_i^T A_i^{-1} R_i  +  R_0^T A_0^{-1} R_0
//
// The fine-level component can be ASM or RAS.
// The coarse-level uses piecewise constant coarse space by default.
enum class FineLevelType { ASM, RAS };

class TwoLevelSchwarzPrecond : public Preconditioner {
public:
    TwoLevelSchwarzPrecond(int nparts, int overlap,
                           FineLevelType fine_type = FineLevelType::RAS)
        : nparts_(nparts), overlap_(overlap), fine_type_(fine_type) {}

    void setup(const CSRMatrix& A) override {
        n_ = A.nrows;

        // Setup fine-level preconditioner
        if (fine_type_ == FineLevelType::ASM) {
            auto p = std::make_unique<ASMPrecond>(nparts_, overlap_);
            p->setup(A);
            dd_ = p->decomposition();
            fine_precond_ = std::move(p);
        } else {
            auto p = std::make_unique<RASPrecond>(nparts_, overlap_);
            p->setup(A);
            dd_ = p->decomposition();
            fine_precond_ = std::move(p);
        }

        // Build coarse space and coarse solver
        coarse_space_.build_piecewise_constant(A, dd_);
        coarse_solver_.setup(coarse_space_.A0, coarse_space_.n_coarse);
    }

    void apply(const double* x, double* y, int n) const override {
        // Fine-level correction
        std::vector<double> z_fine(n);
        fine_precond_->apply(x, z_fine.data(), n);

        // Coarse-level correction
        auto r0 = coarse_space_.apply_restrict(x);
        auto z0 = coarse_solver_.solve(r0);
        std::vector<double> z_coarse(n, 0.0);
        coarse_space_.prolongate(z0.data(), z_coarse.data());

        // Additive combination
        for (int i = 0; i < n; ++i)
            y[i] = z_fine[i] + z_coarse[i];
    }

private:
    int nparts_;
    int overlap_;
    FineLevelType fine_type_;
    int n_ = 0;
    DomainDecomposition dd_;
    std::unique_ptr<Preconditioner> fine_precond_;
    CoarseSpace coarse_space_;
    CoarseSolver coarse_solver_;
};

}  // namespace schwarz

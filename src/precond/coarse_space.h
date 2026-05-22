#pragma once

#include "../core/sparse_matrix.h"
#include "../decomp/domain_decomp.h"
#include <vector>
#include <cmath>

namespace schwarz {

// Coarse space builder for two-level Schwarz methods.
//
// R_0 is the restriction operator (n_coarse x n_fine), stored as a dense
// matrix in row-major form. For piecewise constant, R_0[s][i] = 1/sqrt(|Omega_s|)
// for DOFs in subdomain s.
struct CoarseSpace {
    int n_fine = 0;
    int n_coarse = 0;
    // R0[s] is the s-th coarse basis vector (length n_fine)
    std::vector<std::vector<double>> R0;

    // Coarse operator A_0 = R_0 * A * R_0^T (dense, n_coarse x n_coarse)
    std::vector<double> A0;

    // Build piecewise constant coarse space:
    // phi_s(x) = 1/sqrt(|Omega_s|) if x in Omega_s, else 0
    void build_piecewise_constant(const CSRMatrix& A,
                                  const DomainDecomposition& dd)
    {
        n_fine = A.nrows;
        n_coarse = dd.n_subdomains;
        R0.resize(n_coarse, std::vector<double>(n_fine, 0.0));

        for (int s = 0; s < n_coarse; ++s) {
            double scale = 1.0 / std::sqrt(static_cast<double>(dd.owned_dofs[s].size()));
            for (int dof : dd.owned_dofs[s])
                R0[s][dof] = scale;
        }

        build_coarse_operator(A);
    }

    // A_0 = R_0 * A * R_0^T
    void build_coarse_operator(const CSRMatrix& A) {
        A0.resize(n_coarse * n_coarse, 0.0);

        // Precompute A * R0^T columns: (A * phi_s) for each s
        std::vector<std::vector<double>> A_phi(n_coarse, std::vector<double>(n_fine));
        for (int s = 0; s < n_coarse; ++s)
            A.spmv(R0[s].data(), A_phi[s].data());

        // A_0[i][j] = R0[i]^T * A * R0[j] = dot(R0[i], A_phi[j])
        for (int i = 0; i < n_coarse; ++i) {
            for (int j = 0; j < n_coarse; ++j) {
                double val = 0.0;
                for (int k = 0; k < n_fine; ++k)
                    val += R0[i][k] * A_phi[j][k];
                A0[i * n_coarse + j] = val;
            }
        }
    }

    // Restrict: r_0 = R_0 * r (n_coarse vector)
    std::vector<double> apply_restrict(const double* r) const {
        std::vector<double> r0(n_coarse, 0.0);
        for (int s = 0; s < n_coarse; ++s) {
            for (int k = 0; k < n_fine; ++k)
                r0[s] += R0[s][k] * r[k];
        }
        return r0;
    }

    // Prolongate: z = R_0^T * z_0 (n_fine vector, added to output)
    void prolongate(const double* z0, double* z) const {
        for (int s = 0; s < n_coarse; ++s) {
            for (int k = 0; k < n_fine; ++k)
                z[k] += R0[s][k] * z0[s];
        }
    }
};

}  // namespace schwarz

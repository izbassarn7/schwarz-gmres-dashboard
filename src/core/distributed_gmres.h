#pragma once

#include "distributed_matrix.h"
#include "gmres.h"
#include "vector.h"
#include "../precond/preconditioner.h"

#include <mpi.h>
#include <vector>
#include <cmath>
#include <chrono>
#include <algorithm>

namespace schwarz {

// Distributed restarted GMRES(m) with left preconditioning.
// Matrix is distributed across MPI ranks. All BLAS-1 ops use
// MPI_Allreduce for global dot products and norms.
inline GMRESResult distributed_gmres(
    DistributedCSRMatrix& dA,
    const double* b_local,     // local_nrows entries
    double* x_local,           // n_local_cols entries (owned + ghost)
    const Preconditioner* M,
    const GMRESParams& params = {})
{
    auto t0 = std::chrono::high_resolution_clock::now();

    int n_own = dA.n_owned;
    int n_cols = dA.n_local_cols;
    int m = params.restart;
    GMRESResult result;

    std::vector<double> r(n_own), w(n_cols, 0.0), z(n_own), Av(n_own);
    std::vector<std::vector<double>> V(m + 1, std::vector<double>(n_cols, 0.0));

    std::vector<double> H((m + 1) * m, 0.0);
    std::vector<double> g(m + 1, 0.0);
    std::vector<double> y(m);

    struct GivensRotation {
        double c = 1.0, s = 0.0;
        void compute(double a, double b) {
            if (std::abs(b) < 1e-30) { c = 1.0; s = 0.0; }
            else if (std::abs(b) > std::abs(a)) {
                double t = -a / b; s = 1.0 / std::sqrt(1 + t*t); c = s*t;
            } else {
                double t = -b / a; c = 1.0 / std::sqrt(1 + t*t); s = c*t;
            }
        }
        void apply(double& h1, double& h2) const {
            double t1 = c*h1 - s*h2; double t2 = s*h1 + c*h2;
            h1 = t1; h2 = t2;
        }
    };
    std::vector<GivensRotation> rots(m);

    int total_iter = 0;

    for (int cycle = 0; cycle < params.max_iter; ++cycle) {
        // r = b - A*x (A*x includes halo exchange inside dA.spmv)
        dA.spmv(x_local, r.data());
        for (int i = 0; i < n_own; ++i)
            r[i] = b_local[i] - r[i];

        // z = M^{-1} r
        if (M)
            M->apply(r.data(), z.data(), n_own);
        else
            std::copy(r.begin(), r.end(), z.begin());

        double beta = dA.dist_norm2(z.data());
        if (beta < params.tol) {
            result.converged = true;
            result.residual_norm = beta;
            break;
        }

        double inv_beta = 1.0 / beta;
        for (int i = 0; i < n_own; ++i)
            V[0][i] = z[i] * inv_beta;
        // Zero ghost portion
        for (int i = n_own; i < n_cols; ++i)
            V[0][i] = 0.0;

        std::fill(H.begin(), H.end(), 0.0);
        std::fill(g.begin(), g.end(), 0.0);
        g[0] = beta;

        int j;
        for (j = 0; j < m; ++j) {
            // w = M^{-1} * A * V[j]
            dA.spmv(V[j].data(), Av.data());
            if (M)
                M->apply(Av.data(), w.data(), n_own);
            else
                std::copy(Av.begin(), Av.end(), w.begin());

            // Modified Gram-Schmidt with distributed dot products
            for (int i = 0; i <= j; ++i) {
                double h = dA.dist_dot(w.data(), V[i].data());
                H[i + j * (m + 1)] = h;
                for (int k = 0; k < n_own; ++k)
                    w[k] -= h * V[i][k];
            }

            double h_jp1 = dA.dist_norm2(w.data());
            H[(j + 1) + j * (m + 1)] = h_jp1;

            if (h_jp1 < 1e-30) { j++; break; }

            double inv_h = 1.0 / h_jp1;
            for (int i = 0; i < n_own; ++i)
                V[j + 1][i] = w[i] * inv_h;
            for (int i = n_own; i < n_cols; ++i)
                V[j + 1][i] = 0.0;

            // Apply previous Givens rotations
            for (int i = 0; i < j; ++i)
                rots[i].apply(H[i + j*(m+1)], H[(i+1) + j*(m+1)]);

            rots[j].compute(H[j + j*(m+1)], H[(j+1) + j*(m+1)]);
            rots[j].apply(H[j + j*(m+1)], H[(j+1) + j*(m+1)]);
            rots[j].apply(g[j], g[j+1]);

            double res = std::abs(g[j + 1]);
            result.residual_history.push_back(res);
            total_iter++;

            if (res < params.tol) {
                result.converged = true;
                j++;
                break;
            }
        }

        // Solve H*y = g
        int k = j;
        for (int i = k - 1; i >= 0; --i) {
            y[i] = g[i];
            for (int l = i + 1; l < k; ++l)
                y[i] -= H[i + l*(m+1)] * y[l];
            y[i] /= H[i + i*(m+1)];
        }

        // x = x + V*y (only owned portion)
        for (int i = 0; i < k; ++i)
            for (int kk = 0; kk < n_own; ++kk)
                x_local[kk] += y[i] * V[i][kk];

        if (result.converged) break;
        if (total_iter >= params.max_iter) break;
    }

    result.iterations = total_iter;
    if (!result.residual_history.empty())
        result.residual_norm = result.residual_history.back();

    auto t1 = std::chrono::high_resolution_clock::now();
    result.time_solve_s = std::chrono::duration<double>(t1 - t0).count();
    return result;
}

}  // namespace schwarz

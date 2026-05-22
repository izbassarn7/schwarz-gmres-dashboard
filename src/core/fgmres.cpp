#include "fgmres.h"
#include <chrono>
#include <cstring>
#include <cassert>

namespace schwarz {

namespace {

struct GivensRot {
    double c = 1.0, s = 0.0;
    void compute(double a, double b) {
        if (std::abs(b) < 1e-30) { c = 1.0; s = 0.0; }
        else if (std::abs(b) > std::abs(a)) {
            double t = -a / b; s = 1.0 / std::sqrt(1.0 + t*t); c = s*t;
        } else {
            double t = -b / a; c = 1.0 / std::sqrt(1.0 + t*t); s = c*t;
        }
    }
    void apply(double& h1, double& h2) const {
        double t1 = c*h1 - s*h2, t2 = s*h1 + c*h2;
        h1 = t1; h2 = t2;
    }
};

// Core FGMRES implementation.
// prec: pointer to preconditioner apply function (may be nullptr).
GMRESResult fgmres_impl(
    const CSRMatrix& A,
    const std::vector<double>& b,
    std::vector<double>& x,
    FlexApply prec_apply,
    const GMRESParams& params)
{
    auto t0 = std::chrono::high_resolution_clock::now();

    int n = A.nrows;
    int m = params.restart;
    GMRESResult result;

    // Arnoldi vectors V[j] and preconditioned direction vectors Z[j]
    // FGMRES stores BOTH V (size m+1) and Z (size m).
    std::vector<std::vector<double>> V(m + 1, std::vector<double>(n, 0.0));
    std::vector<std::vector<double>> Z(m,     std::vector<double>(n, 0.0));

    std::vector<double> H((m + 1) * m, 0.0);
    std::vector<double> g(m + 1, 0.0);
    std::vector<double> y(m, 0.0);
    std::vector<GivensRot> rots(m);

    std::vector<double> r(n), w(n);

    int total_iter = 0;

    for (int cycle = 0; cycle < params.max_iter; ++cycle) {
        // r = b - A*x
        A.spmv(x.data(), r.data());
        for (int i = 0; i < n; ++i) r[i] = b[i] - r[i];

        double beta = norm2(r.data(), n);
        if (beta < params.tol) {
            result.converged = true;
            result.residual_norm = beta;
            break;
        }

        double inv_beta = 1.0 / beta;
        for (int i = 0; i < n; ++i) V[0][i] = r[i] * inv_beta;

        std::fill(H.begin(), H.end(), 0.0);
        std::fill(g.begin(), g.end(), 0.0);
        g[0] = beta;

        int j;
        for (j = 0; j < m; ++j) {
            // z_j = M^{-1} v_j  (flexible: allowed to vary per j)
            if (prec_apply)
                prec_apply(V[j].data(), Z[j].data(), n);
            else
                std::copy(V[j].begin(), V[j].end(), Z[j].begin());

            // w = A * z_j
            A.spmv(Z[j].data(), w.data());

            // Modified Gram-Schmidt orthogonalisation against V[0..j]
            for (int i = 0; i <= j; ++i) {
                double h = dot(w.data(), V[i].data(), n);
                H[i + j * (m + 1)] = h;
                axpy(-h, V[i].data(), w.data(), n);
            }

            double h_new = norm2(w.data(), n);
            H[(j + 1) + j * (m + 1)] = h_new;

            if (h_new < 1e-30) { j++; break; }

            double inv_h = 1.0 / h_new;
            for (int i = 0; i < n; ++i) V[j + 1][i] = w[i] * inv_h;

            // Apply previous Givens rotations
            for (int i = 0; i < j; ++i)
                rots[i].apply(H[i + j*(m+1)], H[(i+1) + j*(m+1)]);

            // New Givens rotation
            rots[j].compute(H[j + j*(m+1)], H[(j+1) + j*(m+1)]);
            rots[j].apply(H[j + j*(m+1)], H[(j+1) + j*(m+1)]);
            rots[j].apply(g[j], g[j+1]);

            double res = std::abs(g[j + 1]);
            result.residual_history.push_back(res);
            total_iter++;

            if (res < params.tol) {
                j++;
                result.converged = true;
                break;
            }
        }

        // Solve upper triangular H*y = g
        int k = j;
        for (int i = k - 1; i >= 0; --i) {
            y[i] = g[i];
            for (int l = i + 1; l < k; ++l)
                y[i] -= H[i + l*(m+1)] * y[l];
            y[i] /= H[i + i*(m+1)];
        }

        // x = x + Z*y  (right-preconditioned update uses Z, not V)
        for (int i = 0; i < k; ++i)
            axpy(y[i], Z[i].data(), x.data(), n);

        if (result.converged) break;
        if (total_iter >= params.max_iter) break;
    }

    result.iterations = total_iter;
    result.residual_norm = result.residual_history.empty()
                           ? 0.0 : result.residual_history.back();

    auto t1 = std::chrono::high_resolution_clock::now();
    result.time_solve_s = std::chrono::duration<double>(t1 - t0).count();
    return result;
}

}  // namespace

GMRESResult fgmres(
    const CSRMatrix& A,
    const std::vector<double>& b,
    std::vector<double>& x,
    const Preconditioner* M,
    const GMRESParams& params)
{
    FlexApply prec_apply;
    if (M) {
        prec_apply = [M](const double* in, double* out, int n) {
            M->apply(in, out, n);
        };
    }
    return fgmres_impl(A, b, x, prec_apply, params);
}

GMRESResult fgmres_flex(
    const CSRMatrix& A,
    const std::vector<double>& b,
    std::vector<double>& x,
    FlexApply prec_apply,
    const GMRESParams& params)
{
    return fgmres_impl(A, b, x, prec_apply, params);
}

}  // namespace schwarz

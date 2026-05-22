#pragma once

// Condition number estimation for the preconditioned operator M^{-1}A.
//
// Uses the Lanczos process to build a k×k tridiagonal matrix T_k whose
// eigenvalues (Ritz values) approximate eigenvalues of M^{-1}A.
// kappa(M^{-1}A) ≈ lambda_max / lambda_min from T_k.
//
// NOTE: Standard Lanczos requires a symmetric operator. M^{-1}A is
// symmetric only when A and M^{-1} commute (e.g. M is a polynomial in A).
// For non-symmetric preconditioned operators the Ritz values are
// approximations and may be complex; we report real parts.
//
// For the dissertation, the important insight is:
//   kappa(A)      >> 1  → unpreconditioned system is ill-conditioned
//   kappa(M^{-1}A) < kappa(A)  → preconditioner reduces condition number
//
// Reference: Saad, "Iterative Methods for Sparse Linear Systems", Ch. 6.

#include "../core/sparse_matrix.h"
#include "../core/vector.h"
#include "../precond/preconditioner.h"

#include <vector>
#include <cmath>
#include <iostream>
#include <algorithm>
#include <limits>

namespace schwarz {

struct ConditionEstimate {
    double lambda_min    = 0.0;
    double lambda_max    = 0.0;
    double kappa         = 0.0;
    int    lanczos_steps = 0;
    bool   valid         = false;
};

// Simple symmetric tridiagonal eigenvalue via QR iteration (symmetric QR/QL).
// alpha = diagonal (length n), beta = sub-diagonal (length n-1).
// Returns sorted eigenvalues.
static std::vector<double> tridiag_eigs_ql(
    std::vector<double> d, std::vector<double> e)
{
    int n = static_cast<int>(d.size());
    e.resize(n, 0.0);      // ensure e has n elements (e[n-1] = 0)

    const int MAX_ITER = 100 * n;

    for (int l = 0; l < n; ++l) {
        int m, iter = 0;
        do {
            // Find m: smallest index >= l such that e[m] is small
            for (m = l; m < n - 1; ++m) {
                double off = std::abs(d[m]) + std::abs(d[m + 1]);
                if (std::abs(e[m]) <= 1e-14 * off) break;
            }
            if (m == l) break;

            if (++iter > MAX_ITER) break;

            // Wilkinson shift
            double g = (d[l + 1] - d[l]) / (2.0 * e[l]);
            double r = std::hypot(g, 1.0);
            g = d[m] - d[l] + e[l] / (g + std::copysign(r, g));

            double s = 1.0, c = 1.0, p = 0.0;
            for (int i = m - 1; i >= l; --i) {
                double f = s * e[i];
                double b = c * e[i];
                r = std::hypot(f, g);
                e[i + 1] = r;
                if (r == 0.0) { d[i + 1] -= p; e[m] = 0.0; break; }
                s = f / r;  c = g / r;
                g = d[i + 1] - p;
                r = (d[i] - g) * s + 2.0 * c * b;
                p = s * r;
                d[i + 1] = g + p;
                g = c * r - b;
            }
            d[l] -= p;
            e[l] = g;
            e[m] = 0.0;
        } while (m != l);
    }

    std::sort(d.begin(), d.end());
    return d;
}

// Estimate condition number of (M^{-1} A) using k Lanczos steps.
// If M == nullptr, estimates condition of A itself.
inline ConditionEstimate estimate_condition(
    const CSRMatrix& A,
    const Preconditioner* M,
    int k = 80,
    unsigned int seed = 42)
{
    int n = A.nrows;
    k = std::min(k, n);
    if (k < 2) return {};

    // Random unit starting vector
    std::vector<double> v(n), w(n), z(n);
    {
        unsigned int state = seed;
        double norm = 0.0;
        for (int i = 0; i < n; ++i) {
            state = state * 1664525u + 1013904223u;
            v[i]  = static_cast<double>(static_cast<int>(state)) / 2147483648.0;
            norm += v[i] * v[i];
        }
        norm = std::sqrt(norm);
        if (norm < 1e-14) norm = 1.0;
        for (int i = 0; i < n; ++i) v[i] /= norm;
    }

    std::vector<double> alpha_vec, beta_vec;
    alpha_vec.reserve(k);
    beta_vec.reserve(k - 1);

    std::vector<double> v_prev(n, 0.0), v_curr = v;
    double beta = 0.0;

    for (int j = 0; j < k; ++j) {
        // w = A * v_curr
        A.spmv(v_curr.data(), w.data());
        // z = M^{-1} w  (or w if no preconditioner)
        if (M)
            M->apply(w.data(), z.data(), n);
        else
            z = w;

        // alpha_j = <v_curr, z>
        double alpha = dot(v_curr.data(), z.data(), n);
        alpha_vec.push_back(alpha);

        // z -= alpha * v_curr + beta * v_prev
        for (int i = 0; i < n; ++i)
            z[i] -= alpha * v_curr[i] + beta * v_prev[i];

        double beta_new = norm2(z.data(), n);

        if (beta_new < 1e-10 * (1.0 + std::abs(alpha))) {
            // Invariant subspace — Lanczos converged exactly
            break;
        }

        if (j < k - 1) {
            beta_vec.push_back(beta_new);
            v_prev = v_curr;
            beta   = beta_new;
            double inv_b = 1.0 / beta_new;
            for (int i = 0; i < n; ++i)
                v_curr[i] = z[i] * inv_b;
        }
    }

    int m = static_cast<int>(alpha_vec.size());
    if (m < 2) return {};

    // Validate diagonal: check for NaN/inf
    for (double a : alpha_vec) {
        if (!std::isfinite(a)) return {};
    }
    for (double b : beta_vec) {
        if (!std::isfinite(b)) return {};
    }

    auto eigs = tridiag_eigs_ql(alpha_vec, beta_vec);

    // Filter out any NaN or non-finite eigenvalues
    std::vector<double> valid_eigs;
    valid_eigs.reserve(eigs.size());
    for (double e : eigs)
        if (std::isfinite(e)) valid_eigs.push_back(e);

    if (valid_eigs.empty()) return {};

    double lmin = valid_eigs.front();
    double lmax = valid_eigs.back();

    ConditionEstimate est;
    est.lambda_min    = lmin;
    est.lambda_max    = lmax;
    est.kappa         = (std::abs(lmin) > 1e-14) ? lmax / lmin : 1e30;
    est.lanczos_steps = m;
    est.valid         = (lmin > 0.0);  // meaningful only if SPD
    return est;
}

inline void print_condition_estimate(const ConditionEstimate& est,
                                     const std::string& label = "")
{
    std::cout << "  Condition estimate" << (label.empty() ? "" : " (" + label + ")")
              << " [Lanczos " << est.lanczos_steps << " steps]:\n";
    if (!est.valid || est.lambda_min <= 0.0) {
        // Non-symmetric operator: report Ritz bounds as-is
        std::cout << "    Ritz(min) = " << est.lambda_min << "\n"
                  << "    Ritz(max) = " << est.lambda_max << "\n"
                  << "    kappa_est = " << est.kappa << "\n"
                  << "    (Note: M^{-1}A is non-symmetric; bounds are approximate)\n";
    } else {
        std::cout << "    lambda_min = " << est.lambda_min << "\n"
                  << "    lambda_max = " << est.lambda_max << "\n"
                  << "    kappa      = " << est.kappa << "\n";
    }
}

}  // namespace schwarz

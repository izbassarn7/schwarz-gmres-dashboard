#pragma once

// Preconditioned BiCGSTAB — Van der Vorst (1992).
//
// BiCGSTAB (Bi-Conjugate Gradient STABilised) is a Krylov subspace method
// for non-symmetric linear systems.  It is cheaper per iteration than GMRES
// (no Gram-Schmidt orthogonalisation, no growing Hessenberg matrix) but may
// exhibit irregular convergence or breakdown on highly non-symmetric problems.
//
// Algorithm (right-preconditioned variant):
//   Solve  A (M^{-1} u) = b,  then  x = M^{-1} u.
//
//   r  = b - A x_0
//   r̂  = r_0    (shadow residual — FIXED throughout)
//   ρ₀ = α = ω = 1,  v = p = 0
//
//   for i = 1, 2, ...:
//     ρᵢ   = (r̂, r)
//     β    = (ρᵢ/ρᵢ₋₁) · (α/ω)
//     p    = r + β(p − ω v)
//     y    = M⁻¹ p
//     v    = A y
//     α    = ρᵢ / (r̂, v)
//     s    = r − α v
//     if ‖s‖ < ε: x += α y;  CONVERGED
//     z    = M⁻¹ s
//     t    = A z
//     ω    = (t, s) / (t, t)
//     x   += α y + ω z
//     r    = s − ω t
//
// Difference from GMRES:
//   GMRES builds an orthonormal Krylov basis (restart every m steps),
//   memory = O(m·n).  BiCGSTAB uses only O(1) extra vectors (y,z,p,v,s,t,r,r̂)
//   and never restarts, but uses TWO matrix-vector products per iteration and
//   can stall if (r̂, v) ≈ 0 (breakdown).
//
// Reference: Van der Vorst, "Bi-CGSTAB: A fast and smoothly converging
//   variant of Bi-CG for the solution of nonsymmetric linear systems",
//   SIAM J. Sci. Stat. Comput., 13(2), 1992.

#include "sparse_matrix.h"
#include "vector.h"
#include "gmres.h"          // reuse GMRESParams / GMRESResult
#include "../precond/preconditioner.h"

#include <vector>
#include <cmath>
#include <chrono>
#include <iostream>

namespace schwarz {

// Solve A x = b with (optional) right preconditioner M.
// x is modified in place (use x=0 for zero start).
// Returns same GMRESResult struct as GMRES for uniform comparison.
inline GMRESResult bicgstab(
    const CSRMatrix&          A,
    const std::vector<double>& b,
    std::vector<double>&       x,
    const Preconditioner*      M,
    const GMRESParams&         params = {})
{
    const int    n       = A.nrows;
    const double tol     = params.tol;
    const int    maxiter = params.max_iter;

    auto t_start = std::chrono::high_resolution_clock::now();

    // ── Allocate working vectors ────────────────────────────────────────────
    std::vector<double> r(n), r_hat(n),
                        p(n,0.0), v(n,0.0),
                        y(n),     z(n),
                        s(n),     t_vec(n),
                        tmp(n);

    // r = b - A x
    A.spmv(x.data(), tmp.data());
    for (int i = 0; i < n; ++i) r[i] = b[i] - tmp[i];

    // r̂ = r  (shadow residual — never updated)
    r_hat = r;

    double r0_norm = norm2(r.data(), n);
    if (r0_norm < 1e-14) {
        // Already converged at x_0
        auto t_end = std::chrono::high_resolution_clock::now();
        return {0, r0_norm, true,
                std::chrono::duration<double>(t_end - t_start).count(), {}};
    }

    double rho   = 1.0, alpha = 1.0, omega = 1.0;
    double rho_prev;

    GMRESResult result;
    result.residual_history.reserve(maxiter);

    for (int iter = 1; iter <= maxiter; ++iter) {
        rho_prev = rho;
        rho = dot(r_hat.data(), r.data(), n);

        if (std::abs(rho) < 1e-300) {
            // Breakdown: shadow residual became orthogonal to r
            result.iterations    = iter;
            result.residual_norm = norm2(r.data(), n);
            result.converged     = false;
            break;
        }

        // First iteration: p = r  (β·p term vanishes because p=0)
        if (iter == 1) {
            p = r;
        } else {
            double beta = (rho / rho_prev) * (alpha / omega);
            for (int i = 0; i < n; ++i)
                p[i] = r[i] + beta * (p[i] - omega * v[i]);
        }

        // y = M⁻¹ p   (right preconditioner)
        if (M) M->apply(p.data(), y.data(), n);
        else   y = p;

        // v = A y
        A.spmv(y.data(), v.data());

        double rhat_v = dot(r_hat.data(), v.data(), n);
        if (std::abs(rhat_v) < 1e-300) {
            result.iterations    = iter;
            result.residual_norm = norm2(r.data(), n);
            result.converged     = false;
            break;
        }

        alpha = rho / rhat_v;

        // s = r − α v
        for (int i = 0; i < n; ++i) s[i] = r[i] - alpha * v[i];

        double s_norm = norm2(s.data(), n);
        if (s_norm < tol * r0_norm) {
            // Lucky breakdown — converged after half-step
            for (int i = 0; i < n; ++i) x[i] += alpha * y[i];
            result.iterations    = iter;
            result.residual_norm = s_norm;
            result.converged     = true;
            result.residual_history.push_back(s_norm);
            break;
        }

        // z = M⁻¹ s
        if (M) M->apply(s.data(), z.data(), n);
        else   z = s;

        // t = A z
        A.spmv(z.data(), t_vec.data());

        double tt = dot(t_vec.data(), t_vec.data(), n);
        if (tt < 1e-300) {
            // t ≈ 0 — also an exact solution along s direction
            for (int i = 0; i < n; ++i) x[i] += alpha * y[i];
            result.iterations    = iter;
            result.residual_norm = s_norm;
            result.converged     = (s_norm < tol * r0_norm);
            break;
        }

        omega = dot(t_vec.data(), s.data(), n) / tt;

        // x += α y + ω z
        for (int i = 0; i < n; ++i)
            x[i] += alpha * y[i] + omega * z[i];

        // r = s − ω t
        for (int i = 0; i < n; ++i)
            r[i] = s[i] - omega * t_vec[i];

        double r_norm = norm2(r.data(), n);
        result.residual_history.push_back(r_norm);

        if (r_norm < tol * r0_norm) {
            result.iterations    = iter;
            result.residual_norm = r_norm;
            result.converged     = true;
            break;
        }

        if (std::abs(omega) < 1e-300) {
            // Stagnation breakdown
            result.iterations    = iter;
            result.residual_norm = r_norm;
            result.converged     = false;
            break;
        }

        // Not yet converged — keep last values for reporting
        result.iterations    = iter;
        result.residual_norm = r_norm;
        result.converged     = false;
    }

    auto t_end = std::chrono::high_resolution_clock::now();
    result.time_solve_s = std::chrono::duration<double>(t_end - t_start).count();
    return result;
}

}  // namespace schwarz

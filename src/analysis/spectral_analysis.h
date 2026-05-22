#pragma once

// Spectral radius estimation of the error propagation operator E = I - M^{-1}A.
//
// Convergence of preconditioned GMRES requires rho(I - M^{-1}A) < 1.
// We estimate this via the power method on E.
//
// Reference: Saad, "Iterative Methods for Sparse Linear Systems", Ch. 12.

#include "../core/sparse_matrix.h"
#include "../core/vector.h"
#include "../precond/preconditioner.h"

#include <vector>
#include <cmath>
#include <iostream>
#include <algorithm>

namespace schwarz {

struct SpectralResult {
    double spectral_radius  = 0.0;   // rho(I - M^{-1}A)
    bool   convergence_ok   = false; // true if rho < 1
    int    power_iters      = 0;
    // Dominant eigenvector direction (normalised)
    std::vector<double> eigenvector;
};

// Power iteration for the spectral radius of E = I - M^{-1}A.
// Each step: v_{k+1} = E v_k / ||E v_k||
//            mu_k    = <v_k, E v_k>  (Rayleigh quotient)
inline SpectralResult estimate_spectral_radius(
    const CSRMatrix& A,
    const Preconditioner* M,
    int max_iter = 200,
    double tol   = 1e-6,
    unsigned int seed = 123)
{
    int n = A.nrows;

    // Random starting vector
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
        for (int i = 0; i < n; ++i) v[i] /= norm;
    }

    double mu_prev = 0.0;
    int actual_iters = 0;

    for (int it = 0; it < max_iter; ++it) {
        // w = A * v
        A.spmv(v.data(), w.data());
        // z = M^{-1} w
        if (M)
            M->apply(w.data(), z.data(), n);
        else
            z = w;

        // Ev = v - z  (= v - M^{-1}Av)
        for (int i = 0; i < n; ++i)
            w[i] = v[i] - z[i];

        // Rayleigh quotient: mu = <v, Ev> / <v, v> = <v, Ev> (v already unit)
        double mu = dot(v.data(), w.data(), n);

        // Normalise
        double ev_norm = norm2(w.data(), n);
        if (ev_norm < 1e-14) {
            actual_iters = it + 1;
            mu = 0.0;
            break;
        }

        for (int i = 0; i < n; ++i)
            v[i] = w[i] / ev_norm;

        actual_iters = it + 1;

        if (std::abs(mu - mu_prev) < tol * (1.0 + std::abs(mu))) break;
        mu_prev = mu;
    }

    SpectralResult res;
    res.spectral_radius = std::abs(mu_prev);
    res.convergence_ok  = (res.spectral_radius < 1.0);
    res.power_iters     = actual_iters;
    res.eigenvector     = v;
    return res;
}

inline void print_spectral_result(const SpectralResult& sr,
                                  const std::string& label = "")
{
    std::cout << "  Spectral radius" << (label.empty() ? "" : " (" + label + ")")
              << " [power iter " << sr.power_iters << " steps]:\n"
              << "    rho(I - M^{-1}A) = " << sr.spectral_radius << "\n"
              << "    Convergence cond : " << (sr.convergence_ok ? "SATISFIED (rho < 1)" : "VIOLATED (rho >= 1)")
              << "\n";
}

}  // namespace schwarz

#pragma once

#include "sparse_matrix.h"
#include "vector.h"
#include "../precond/preconditioner.h"
#include <vector>
#include <cmath>
#include <functional>

namespace schwarz {

struct GMRESResult {
    int iterations = 0;
    double residual_norm = 0.0;
    bool converged = false;
    double time_solve_s = 0.0;
    std::vector<double> residual_history;
};

struct GMRESParams {
    int restart = 30;
    int max_iter = 1000;
    double tol = 1e-10;
};

// Restarted GMRES(m) with left preconditioning.
// Solves Ax = b, starting from x (modified in place).
GMRESResult gmres(
    const CSRMatrix& A,
    const std::vector<double>& b,
    std::vector<double>& x,
    const Preconditioner* M,
    const GMRESParams& params = {});

}  // namespace schwarz

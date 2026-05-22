#pragma once

// Flexible GMRES (FGMRES) — Saad (1993).
//
// FGMRES generalises GMRES to allow variable (non-stationary) right
// preconditioners. It is required when the preconditioner itself applies
// an iterative method (e.g. nested Krylov) or varies between applications
// (e.g. two-level Schwarz with inexact coarse solve).
//
// Algorithm (right-preconditioned variant):
//   Find x_m = x_0 + Z_m y_m, where Z_m = M^{-1} V_m, such that
//   || b - A Z_m y_m || is minimised in the Krylov space.
//
// Key difference from GMRES:
//   GMRES stores V_m (Arnoldi vectors), FGMRES stores both
//   V_m (search directions) and Z_m = M^{-1} V_m (preconditioned directions).
//   The preconditioner is called ONCE PER STEP (not per iteration of a cycle),
//   so even an inexact preconditioner application is allowed.
//
// Reference: Saad, "A Flexible Inner-Outer Preconditioned GMRES Algorithm",
//            SIAM J. Sci. Comput., 1993.

#include "sparse_matrix.h"
#include "gmres.h"   // reuse GMRESParams / GMRESResult
#include "vector.h"
#include "../precond/preconditioner.h"

#include <vector>
#include <cmath>
#include <chrono>
#include <functional>

namespace schwarz {

// Signature for a flexible preconditioner: may differ per call (e.g. inner GMRES).
// Defaults to a fixed Preconditioner* if the flexible interface is not needed.
using FlexApply = std::function<void(const double*, double*, int)>;

// FGMRES(m) with right preconditioning.
// Solves A x = b, starting from x (modified in place).
// prec_apply: callable y = M^{-1}(x).  Pass nullptr to disable.
GMRESResult fgmres(
    const CSRMatrix& A,
    const std::vector<double>& b,
    std::vector<double>& x,
    const Preconditioner* M,
    const GMRESParams& params = {});

// Overload accepting a flexible (non-stationary) apply functor.
GMRESResult fgmres_flex(
    const CSRMatrix& A,
    const std::vector<double>& b,
    std::vector<double>& x,
    FlexApply prec_apply,   // may be nullptr
    const GMRESParams& params = {});

}  // namespace schwarz

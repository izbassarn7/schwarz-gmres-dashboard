#include "../src/core/sparse_matrix.h"
#include "../src/core/gmres.h"
#include "../src/core/vector.h"
#include "../src/precond/jacobi.h"
#include "../src/precond/ilu0.h"

#include <iostream>
#include <cmath>
#include <cassert>
#include <vector>

using namespace schwarz;

static int test_count = 0;
static int pass_count = 0;

#define CHECK(cond, msg) do { \
    test_count++; \
    if (!(cond)) { \
        std::cerr << "FAIL: " << msg << " (line " << __LINE__ << ")\n"; \
    } else { \
        pass_count++; \
        std::cout << "PASS: " << msg << "\n"; \
    } \
} while(0)

static void test_identity() {
    CSRMatrix I = make_identity(10);
    std::vector<double> b = {1,2,3,4,5,6,7,8,9,10};
    std::vector<double> x(10, 0.0);

    GMRESParams p;
    p.restart = 10;
    p.tol = 1e-12;
    auto res = gmres(I, b, x, nullptr, p);

    CHECK(res.converged, "Identity converges");
    CHECK(res.iterations <= 2, "Identity converges in <= 2 iters");

    double err = 0;
    for (int i = 0; i < 10; ++i)
        err += std::abs(x[i] - b[i]);
    CHECK(err < 1e-10, "Identity solution is correct");
}

static void test_small_spd() {
    // 3x3 SPD: [[4,1,0],[1,3,1],[0,1,2]]
    CSRMatrix A;
    A.nrows = 3; A.ncols = 3;
    A.row_ptr = {0, 2, 5, 7};
    A.col_idx = {0,1, 0,1,2, 1,2};
    A.vals    = {4,1, 1,3,1, 1,2};

    // Exact solution: [1, 2, 3]
    std::vector<double> x_exact = {1.0, 2.0, 3.0};
    std::vector<double> b(3);
    A.spmv(x_exact, b);

    // No preconditioner
    {
        std::vector<double> x(3, 0.0);
        auto res = gmres(A, b, x, nullptr);
        CHECK(res.converged, "Small SPD converges (no precond)");
        double err = 0;
        for (int i = 0; i < 3; ++i) err += std::abs(x[i] - x_exact[i]);
        CHECK(err < 1e-8, "Small SPD correct solution (no precond)");
    }

    // Jacobi
    {
        JacobiPrecond jac;
        jac.setup(A);
        std::vector<double> x(3, 0.0);
        auto res = gmres(A, b, x, &jac);
        CHECK(res.converged, "Small SPD converges (Jacobi)");
        double err = 0;
        for (int i = 0; i < 3; ++i) err += std::abs(x[i] - x_exact[i]);
        CHECK(err < 1e-8, "Small SPD correct solution (Jacobi)");
    }

    // ILU(0)
    {
        ILU0Precond ilu;
        ilu.setup(A);
        std::vector<double> x(3, 0.0);
        auto res = gmres(A, b, x, &ilu);
        CHECK(res.converged, "Small SPD converges (ILU0)");
        double err = 0;
        for (int i = 0; i < 3; ++i) err += std::abs(x[i] - x_exact[i]);
        CHECK(err < 1e-8, "Small SPD correct solution (ILU0)");
    }
}

static void test_2d_poisson_small() {
    // 5-point stencil on 5x5 grid = 25 unknowns
    int nx = 5;
    int n = nx * nx;

    CSRMatrix A;
    A.nrows = n; A.ncols = n;
    A.row_ptr.resize(n + 1, 0);

    std::vector<int> ci;
    std::vector<double> cv;

    for (int i = 0; i < nx; ++i) {
        for (int j = 0; j < nx; ++j) {
            int idx = i * nx + j;
            int start = static_cast<int>(ci.size());
            A.row_ptr[idx] = start;

            if (i > 0)      { ci.push_back(idx - nx); cv.push_back(-1.0); }
            if (j > 0)      { ci.push_back(idx - 1);  cv.push_back(-1.0); }
            ci.push_back(idx); cv.push_back(4.0);
            if (j < nx - 1) { ci.push_back(idx + 1);  cv.push_back(-1.0); }
            if (i < nx - 1) { ci.push_back(idx + nx); cv.push_back(-1.0); }
        }
    }
    A.row_ptr[n] = static_cast<int>(ci.size());
    A.col_idx = ci;
    A.vals = cv;

    std::vector<double> x_exact(n, 1.0);
    std::vector<double> b(n);
    A.spmv(x_exact, b);

    ILU0Precond ilu;
    ilu.setup(A);

    std::vector<double> x(n, 0.0);
    GMRESParams p;
    p.restart = 25;
    p.tol = 1e-10;
    auto res = gmres(A, b, x, &ilu, p);

    CHECK(res.converged, "2D Poisson 5x5 converges");
    CHECK(res.residual_norm < 1e-10, "2D Poisson residual < 1e-10");

    double err = 0;
    for (int i = 0; i < n; ++i) err += std::abs(x[i] - x_exact[i]);
    CHECK(err < 1e-6, "2D Poisson solution correct");
}

int main() {
    std::cout << "=== GMRES Tests ===\n";
    test_identity();
    test_small_spd();
    test_2d_poisson_small();

    std::cout << "\n" << pass_count << "/" << test_count << " tests passed.\n";
    return (pass_count == test_count) ? 0 : 1;
}

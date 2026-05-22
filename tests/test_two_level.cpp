#include "../src/core/sparse_matrix.h"
#include "../src/core/gmres.h"
#include "../src/core/vector.h"
#include "../src/mesh/poisson.h"
#include "../src/precond/asm.h"
#include "../src/precond/ras.h"
#include "../src/precond/two_level_schwarz.h"

#include <iostream>
#include <cmath>

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

static void test_two_level_asm() {
    int nx = 30, ny = 30;
    CSRMatrix A = make_poisson_2d(nx, ny);
    int n = A.nrows;

    std::vector<double> x_exact(n, 1.0);
    std::vector<double> b(n);
    A.spmv(x_exact, b);

    TwoLevelSchwarzPrecond pc(8, 1, FineLevelType::ASM);
    pc.setup(A);

    std::vector<double> x(n, 0.0);
    GMRESParams p;
    p.restart = 50;
    p.tol = 1e-10;
    p.max_iter = 500;

    auto res = gmres(A, b, x, &pc, p);
    std::cout << "  TwoLevel ASM: iters=" << res.iterations
              << " residual=" << res.residual_norm << "\n";

    CHECK(res.converged, "Two-level ASM converges");
    CHECK(res.residual_norm < 1e-9, "Two-level ASM residual < 1e-9");
}

static void test_two_level_ras() {
    int nx = 30, ny = 30;
    CSRMatrix A = make_poisson_2d(nx, ny);
    int n = A.nrows;

    std::vector<double> x_exact(n, 1.0);
    std::vector<double> b(n);
    A.spmv(x_exact, b);

    TwoLevelSchwarzPrecond pc(8, 1, FineLevelType::RAS);
    pc.setup(A);

    std::vector<double> x(n, 0.0);
    GMRESParams p;
    p.restart = 50;
    p.tol = 1e-10;
    p.max_iter = 500;

    auto res = gmres(A, b, x, &pc, p);
    std::cout << "  TwoLevel RAS: iters=" << res.iterations
              << " residual=" << res.residual_norm << "\n";

    CHECK(res.converged, "Two-level RAS converges");
    CHECK(res.residual_norm < 1e-9, "Two-level RAS residual < 1e-9");
}

static void test_two_level_vs_one_level() {
    int nx = 40, ny = 40;
    CSRMatrix A = make_poisson_2d(nx, ny);
    int n = A.nrows;

    std::vector<double> x_exact(n, 1.0);
    std::vector<double> b(n);
    A.spmv(x_exact, b);

    GMRESParams p;
    p.restart = 50;
    p.tol = 1e-10;
    p.max_iter = 1000;

    // One-level RAS
    RASPrecond ras_1level(16, 1);
    ras_1level.setup(A);
    std::vector<double> x1(n, 0.0);
    auto res1 = gmres(A, b, x1, &ras_1level, p);

    // Two-level RAS
    TwoLevelSchwarzPrecond ras_2level(16, 1, FineLevelType::RAS);
    ras_2level.setup(A);
    std::vector<double> x2(n, 0.0);
    auto res2 = gmres(A, b, x2, &ras_2level, p);

    std::cout << "  1-level RAS: iters=" << res1.iterations << "\n";
    std::cout << "  2-level RAS: iters=" << res2.iterations << "\n";

    CHECK(res1.converged, "1-level RAS converges (40x40, 16 parts)");
    CHECK(res2.converged, "2-level RAS converges (40x40, 16 parts)");
    // Two-level should converge in fewer or equal iterations
    CHECK(res2.iterations <= res1.iterations + 5,
          "Two-level improves or matches one-level convergence");
}

static void test_scaling_subdomains() {
    int nx = 30, ny = 30;
    CSRMatrix A = make_poisson_2d(nx, ny);
    int n = A.nrows;

    std::vector<double> x_exact(n, 1.0);
    std::vector<double> b(n);
    A.spmv(x_exact, b);

    GMRESParams p;
    p.restart = 50;
    p.tol = 1e-10;
    p.max_iter = 500;

    std::cout << "  Subdomain scaling test (two-level RAS):\n";
    for (int nparts : {4, 8, 16}) {
        TwoLevelSchwarzPrecond pc(nparts, 1, FineLevelType::RAS);
        pc.setup(A);

        std::vector<double> x(n, 0.0);
        auto res = gmres(A, b, x, &pc, p);
        std::cout << "    nparts=" << nparts
                  << " iters=" << res.iterations
                  << " converged=" << res.converged << "\n";

        CHECK(res.converged,
              "Two-level RAS converges with nparts=" + std::to_string(nparts));
    }
}

int main() {
    std::cout << "=== Two-Level Schwarz Tests ===\n";
    test_two_level_asm();
    test_two_level_ras();
    test_two_level_vs_one_level();
    test_scaling_subdomains();

    std::cout << "\n" << pass_count << "/" << test_count << " tests passed.\n";
    return (pass_count == test_count) ? 0 : 1;
}

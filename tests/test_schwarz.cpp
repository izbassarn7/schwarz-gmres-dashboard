#include "../src/core/sparse_matrix.h"
#include "../src/core/gmres.h"
#include "../src/core/vector.h"
#include "../src/mesh/poisson.h"
#include "../src/precond/asm.h"
#include "../src/precond/ras.h"

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

static void test_asm_poisson2d() {
    int nx = 20, ny = 20;
    CSRMatrix A = make_poisson_2d(nx, ny);
    int n = A.nrows;

    std::vector<double> x_exact(n, 1.0);
    std::vector<double> b(n);
    A.spmv(x_exact, b);

    ASMPrecond asm_pc(4, 1);
    asm_pc.setup(A);

    std::vector<double> x(n, 0.0);
    GMRESParams p;
    p.restart = 30;
    p.tol = 1e-10;
    p.max_iter = 500;

    auto res = gmres(A, b, x, &asm_pc, p);

    std::cout << "  ASM: iters=" << res.iterations
              << " residual=" << res.residual_norm << "\n";

    CHECK(res.converged, "ASM on 20x20 Poisson converges");
    CHECK(res.residual_norm < 1e-9, "ASM residual < 1e-9");
}

static void test_ras_poisson2d() {
    int nx = 20, ny = 20;
    CSRMatrix A = make_poisson_2d(nx, ny);
    int n = A.nrows;

    std::vector<double> x_exact(n, 1.0);
    std::vector<double> b(n);
    A.spmv(x_exact, b);

    RASPrecond ras_pc(4, 1);
    ras_pc.setup(A);

    std::vector<double> x(n, 0.0);
    GMRESParams p;
    p.restart = 30;
    p.tol = 1e-10;
    p.max_iter = 500;

    auto res = gmres(A, b, x, &ras_pc, p);

    std::cout << "  RAS: iters=" << res.iterations
              << " residual=" << res.residual_norm << "\n";

    CHECK(res.converged, "RAS on 20x20 Poisson converges");
    CHECK(res.residual_norm < 1e-9, "RAS residual < 1e-9");
}

static void test_asm_vs_ras_iteration_count() {
    int nx = 30, ny = 30;
    CSRMatrix A = make_poisson_2d(nx, ny);
    int n = A.nrows;

    std::vector<double> x_exact(n, 1.0);
    std::vector<double> b(n);
    A.spmv(x_exact, b);

    GMRESParams p;
    p.restart = 50;
    p.tol = 1e-10;
    p.max_iter = 1000;

    ASMPrecond asm_pc(8, 2);
    asm_pc.setup(A);
    std::vector<double> x1(n, 0.0);
    auto res_asm = gmres(A, b, x1, &asm_pc, p);

    RASPrecond ras_pc(8, 2);
    ras_pc.setup(A);
    std::vector<double> x2(n, 0.0);
    auto res_ras = gmres(A, b, x2, &ras_pc, p);

    std::cout << "  ASM iters=" << res_asm.iterations
              << "  RAS iters=" << res_ras.iterations << "\n";

    CHECK(res_asm.converged, "ASM 30x30 converges");
    CHECK(res_ras.converged, "RAS 30x30 converges");
    // RAS typically converges in fewer iterations than ASM
    CHECK(res_ras.iterations <= res_asm.iterations + 5,
          "RAS iters <= ASM iters (within margin)");
}

static void test_overlap_effect() {
    int nx = 20, ny = 20;
    CSRMatrix A = make_poisson_2d(nx, ny);
    int n = A.nrows;

    std::vector<double> x_exact(n, 1.0);
    std::vector<double> b(n);
    A.spmv(x_exact, b);

    GMRESParams p;
    p.restart = 30;
    p.tol = 1e-10;
    p.max_iter = 500;

    int iters_ov0, iters_ov2;

    {
        RASPrecond ras(4, 0);
        ras.setup(A);
        std::vector<double> x(n, 0.0);
        auto res = gmres(A, b, x, &ras, p);
        iters_ov0 = res.iterations;
        std::cout << "  Overlap=0: " << iters_ov0 << " iters\n";
    }
    {
        RASPrecond ras(4, 2);
        ras.setup(A);
        std::vector<double> x(n, 0.0);
        auto res = gmres(A, b, x, &ras, p);
        iters_ov2 = res.iterations;
        std::cout << "  Overlap=2: " << iters_ov2 << " iters\n";
    }

    CHECK(iters_ov2 <= iters_ov0,
          "More overlap => fewer or equal iterations");
}

int main() {
    std::cout << "=== Schwarz Tests ===\n";
    test_asm_poisson2d();
    test_ras_poisson2d();
    test_asm_vs_ras_iteration_count();
    test_overlap_effect();

    std::cout << "\n" << pass_count << "/" << test_count << " tests passed.\n";
    return (pass_count == test_count) ? 0 : 1;
}

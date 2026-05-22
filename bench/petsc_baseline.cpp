#ifdef USE_PETSC

#include <petsc.h>
#include "../src/io/matrix_market.h"
#include "../src/io/metrics_logger.h"
#include "../src/mesh/poisson.h"

#include <iostream>
#include <string>
#include <chrono>
#include <cstring>

using namespace schwarz;

static PetscErrorCode create_petsc_matrix(const CSRMatrix& A, Mat* pmat) {
    PetscErrorCode ierr;
    ierr = MatCreateSeqAIJ(PETSC_COMM_WORLD, A.nrows, A.ncols, 0, nullptr, pmat);
    CHKERRQ(ierr);

    for (int i = 0; i < A.nrows; ++i) {
        for (int j = A.row_ptr[i]; j < A.row_ptr[i + 1]; ++j) {
            PetscInt row = i, col = A.col_idx[j];
            PetscScalar val = A.vals[j];
            ierr = MatSetValues(*pmat, 1, &row, 1, &col, &val, INSERT_VALUES);
            CHKERRQ(ierr);
        }
    }
    ierr = MatAssemblyBegin(*pmat, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
    ierr = MatAssemblyEnd(*pmat, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
    return 0;
}

int main(int argc, char* argv[]) {
    PetscInitialize(&argc, &argv, nullptr, nullptr);

    int rank;
    MPI_Comm_rank(PETSC_COMM_WORLD, &rank);

    std::string matrix_file;
    std::string pc_type_str = "asm";
    int overlap = 1;
    int restart = 30;
    double tol = 1e-10;
    int max_iter = 1000;
    int nx = 100, ny = 100;
    std::string output = "results/petsc_baseline.jsonl";

    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--matrix") && i + 1 < argc)
            matrix_file = argv[++i];
        else if (!std::strcmp(argv[i], "--pc") && i + 1 < argc)
            pc_type_str = argv[++i];
        else if (!std::strcmp(argv[i], "--overlap") && i + 1 < argc)
            overlap = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--restart") && i + 1 < argc)
            restart = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--nx") && i + 1 < argc)
            nx = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--ny") && i + 1 < argc)
            ny = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--output") && i + 1 < argc)
            output = argv[++i];
    }

    CSRMatrix A_host;
    std::string matrix_name;
    if (!matrix_file.empty()) {
        A_host = read_matrix_market(matrix_file);
        matrix_name = matrix_file;
    } else {
        A_host = make_poisson_2d(nx, ny);
        matrix_name = "poisson2d_" + std::to_string(nx) + "x" + std::to_string(ny);
    }

    Mat A;
    create_petsc_matrix(A_host, &A);

    Vec b, x;
    int n = A_host.nrows;
    VecCreateSeq(PETSC_COMM_WORLD, n, &b);
    VecDuplicate(b, &x);

    // b = A * ones
    Vec ones;
    VecDuplicate(b, &ones);
    VecSet(ones, 1.0);
    MatMult(A, ones, b);
    VecDestroy(&ones);
    VecSet(x, 0.0);

    KSP ksp;
    PC pc;

    auto t_setup_start = std::chrono::high_resolution_clock::now();

    KSPCreate(PETSC_COMM_WORLD, &ksp);
    KSPSetOperators(ksp, A, A);
    KSPSetType(ksp, KSPGMRES);
    KSPGMRESSetRestart(ksp, restart);

    KSPGetPC(ksp, &pc);

    if (pc_type_str == "asm") {
        PCSetType(pc, PCASM);
        PCASMSetOverlap(pc, overlap);
        PCASMSetType(pc, PC_ASM_BASIC);
    } else if (pc_type_str == "ras") {
        PCSetType(pc, PCASM);
        PCASMSetOverlap(pc, overlap);
        PCASMSetType(pc, PC_ASM_RESTRICT);
    } else if (pc_type_str == "ilu") {
        PCSetType(pc, PCILU);
    } else if (pc_type_str == "jacobi") {
        PCSetType(pc, PCJACOBI);
    }

    KSPSetTolerances(ksp, tol, PETSC_DEFAULT, PETSC_DEFAULT, max_iter);
    KSPSetUp(ksp);

    auto t_setup_end = std::chrono::high_resolution_clock::now();
    double t_setup = std::chrono::duration<double>(t_setup_end - t_setup_start).count();

    auto t_solve_start = std::chrono::high_resolution_clock::now();
    KSPSolve(ksp, b, x);
    auto t_solve_end = std::chrono::high_resolution_clock::now();
    double t_solve = std::chrono::duration<double>(t_solve_end - t_solve_start).count();

    PetscInt its;
    PetscReal rnorm;
    KSPGetIterationNumber(ksp, &its);
    KSPGetResidualNorm(ksp, &rnorm);

    KSPConvergedReason reason;
    KSPGetConvergedReason(ksp, &reason);

    if (rank == 0) {
        std::cout << "PETSc baseline: " << matrix_name << "\n";
        std::cout << "  PC: " << pc_type_str << "\n";
        std::cout << "  Converged: " << (reason > 0 ? "yes" : "no")
                  << " (reason=" << reason << ")\n";
        std::cout << "  Iterations: " << its << "\n";
        std::cout << "  Residual: " << rnorm << "\n";
        std::cout << "  Setup time: " << t_setup << " s\n";
        std::cout << "  Solve time: " << t_solve << " s\n";

        Metrics m;
        m.matrix = matrix_name;
        m.method = "PETSc_GMRES+" + pc_type_str;
        m.n = n;
        m.nnz = A_host.nnz();
        m.overlap = overlap;
        m.restart = restart;
        m.tol = tol;
        m.iterations = its;
        m.residual_norm = rnorm;
        m.time_setup_s = t_setup;
        m.time_solve_s = t_solve;
        m.write_jsonl(output);
    }

    KSPDestroy(&ksp);
    VecDestroy(&b);
    VecDestroy(&x);
    MatDestroy(&A);

    PetscFinalize();
    return (reason > 0) ? 0 : 1;
}

#else
#include <iostream>
int main() {
    std::cerr << "PETSc support not enabled. Rebuild with -DUSE_PETSC=ON\n";
    return 1;
}
#endif

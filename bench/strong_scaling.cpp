#include "../src/core/sparse_matrix.h"
#include "../src/core/gmres.h"
#include "../src/core/vector.h"
#include "../src/mesh/poisson.h"
#include "../src/precond/jacobi.h"
#include "../src/precond/ilu0.h"
#include "../src/precond/asm.h"
#include "../src/precond/ras.h"
#include "../src/precond/two_level_schwarz.h"
#include "../src/io/matrix_market.h"
#include "../src/io/metrics_logger.h"

#include <mpi.h>
#include <omp.h>
#include <iostream>
#include <iomanip>
#include <string>
#include <memory>
#include <chrono>
#include <cstring>
#include <vector>
#include <numeric>

using namespace schwarz;

static std::unique_ptr<Preconditioner> make_precond(
    const std::string& method, int nparts, int overlap, const CSRMatrix& A)
{
    if (method == "asm") {
        auto p = std::make_unique<ASMPrecond>(nparts, overlap);
        p->setup(A); return p;
    } else if (method == "ras") {
        auto p = std::make_unique<RASPrecond>(nparts, overlap);
        p->setup(A); return p;
    } else if (method == "twolevel_asm") {
        auto p = std::make_unique<TwoLevelSchwarzPrecond>(nparts, overlap, FineLevelType::ASM);
        p->setup(A); return p;
    } else if (method == "twolevel_ras") {
        auto p = std::make_unique<TwoLevelSchwarzPrecond>(nparts, overlap, FineLevelType::RAS);
        p->setup(A); return p;
    } else if (method == "ilu0") {
        auto p = std::make_unique<ILU0Precond>();
        p->setup(A); return p;
    } else {
        auto p = std::make_unique<JacobiPrecond>();
        p->setup(A); return p;
    }
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int rank, nranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);

    int nx = 200, ny = 200;
    std::string matrix_file;
    std::string method = "twolevel_ras";
    int overlap   = 1;
    int restart   = 50;
    double tol    = 1e-10;
    int max_iter  = 2000;
    std::string output = "results/strong_scaling.jsonl";
    // nparts sweep: 1,2,4,8,16,32
    std::vector<int> nparts_list = {1, 2, 4, 8, 16, 32};

    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--nx") && i + 1 < argc)
            nx = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--ny") && i + 1 < argc)
            ny = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--matrix") && i + 1 < argc)
            matrix_file = argv[++i];
        else if (!std::strcmp(argv[i], "--method") && i + 1 < argc)
            method = argv[++i];
        else if (!std::strcmp(argv[i], "--overlap") && i + 1 < argc)
            overlap = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--restart") && i + 1 < argc)
            restart = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--tol") && i + 1 < argc)
            tol = std::atof(argv[++i]);
        else if (!std::strcmp(argv[i], "--maxiter") && i + 1 < argc)
            max_iter = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--output") && i + 1 < argc)
            output = argv[++i];
    }

    CSRMatrix A;
    std::string matrix_name;
    if (!matrix_file.empty()) {
        if (rank == 0) A = read_matrix_market(matrix_file);
        matrix_name = matrix_file;
    } else {
        A = make_poisson_2d(nx, ny);
        matrix_name = "poisson2d_" + std::to_string(nx) + "x" + std::to_string(ny);
    }

    int n     = A.nrows;
    int nnz_a = A.nnz();

    std::vector<double> x_exact(n, 1.0);
    std::vector<double> b(n);
    A.spmv(x_exact, b);

    int threads = omp_get_max_threads();

    GMRESParams params;
    params.restart  = restart;
    params.tol      = tol;
    params.max_iter = max_iter;

    if (rank == 0) {
        std::cout << "=== Strong Scaling Study ===\n";
        std::cout << "Matrix: " << matrix_name
                  << "  n=" << n << "  nnz=" << nnz_a << "\n";
        std::cout << "Method: GMRES(" << restart << ") + " << method
                  << "  overlap=" << overlap << "\n";
        std::cout << std::setw(8)  << "nparts"
                  << std::setw(12) << "iters"
                  << std::setw(12) << "setup(s)"
                  << std::setw(12) << "solve(s)"
                  << std::setw(10) << "speedup"
                  << std::setw(12) << "efficiency"
                  << std::setw(10) << "residual"
                  << "\n";
        std::cout << std::string(76, '-') << "\n";
    }

    double t1_solve = -1.0;  // baseline solve time with nparts=1

    for (int nparts : nparts_list) {
        // Skip if nparts > n (degenerate)
        if (nparts > n) continue;

        auto t_setup_s = std::chrono::high_resolution_clock::now();
        auto M = make_precond(method, nparts, overlap, A);
        auto t_setup_e = std::chrono::high_resolution_clock::now();
        double t_setup = std::chrono::duration<double>(t_setup_e - t_setup_s).count();

        std::vector<double> x(n, 0.0);
        auto res = gmres(A, b, x, M.get(), params);

        double t_solve = res.time_solve_s;

        // Speedup S_p = T_1 / T_p ; Efficiency E_p = S_p / p
        if (nparts == 1) t1_solve = t_solve;
        double speedup    = (t1_solve > 0.0) ? (t1_solve / t_solve) : 1.0;
        double efficiency = speedup / static_cast<double>(nparts);

        if (rank == 0) {
            std::cout << std::fixed << std::setprecision(4)
                      << std::setw(8)  << nparts
                      << std::setw(12) << res.iterations
                      << std::setw(12) << t_setup
                      << std::setw(12) << t_solve
                      << std::setw(10) << speedup
                      << std::setw(12) << efficiency
                      << std::setw(10) << std::scientific << res.residual_norm
                      << "\n";

            Metrics m;
            m.matrix        = matrix_name;
            m.method        = "strong_scaling_GMRES+" + method;
            m.n             = n;
            m.nnz           = nnz_a;
            m.ranks         = nparts;
            m.threads       = threads;
            m.overlap       = overlap;
            m.restart       = restart;
            m.tol           = tol;
            m.iterations    = res.iterations;
            m.residual_norm = res.residual_norm;
            m.time_setup_s  = t_setup;
            m.time_solve_s  = t_solve;
            m.speedup       = speedup;
            m.efficiency    = efficiency;
            m.write_jsonl(output);
        }
    }

    if (rank == 0)
        std::cout << "\nResults written to " << output << "\n";

    MPI_Finalize();
    return 0;
}

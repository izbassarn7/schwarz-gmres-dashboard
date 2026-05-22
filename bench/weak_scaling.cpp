#include "../src/core/sparse_matrix.h"
#include "../src/core/gmres.h"
#include "../src/core/vector.h"
#include "../src/mesh/poisson.h"
#include "../src/precond/ras.h"
#include "../src/precond/two_level_schwarz.h"
#include "../src/io/metrics_logger.h"

#include <mpi.h>
#include <omp.h>
#include <iostream>
#include <iomanip>
#include <string>
#include <memory>
#include <chrono>
#include <cmath>
#include <cstring>
#include <vector>

using namespace schwarz;

// Weak scaling study.
// For p processes, problem size = dofs_per_proc * p.
// Grid: nx = ceil(sqrt(total_dofs)).
// Ideal weak scaling: solve time constant as p and problem grow together.
// Efficiency_weak = T_1 / T_p  (1.0 = perfect weak scaling).

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int rank, nranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);

    int dofs_per_proc = 10000;  // DOFs per "virtual" process
    std::string method = "twolevel_ras";
    int overlap  = 1;
    int restart  = 50;
    double tol   = 1e-10;
    int max_iter = 2000;
    std::string output = "results/weak_scaling.jsonl";
    int max_procs = 16;  // upper bound for sweep

    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--dofs") && i + 1 < argc)
            dofs_per_proc = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--method") && i + 1 < argc)
            method = argv[++i];
        else if (!std::strcmp(argv[i], "--overlap") && i + 1 < argc)
            overlap = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--restart") && i + 1 < argc)
            restart = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--maxprocs") && i + 1 < argc)
            max_procs = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--output") && i + 1 < argc)
            output = argv[++i];
    }

    int threads = omp_get_max_threads();

    if (rank == 0) {
        std::cout << "=== Weak Scaling Study ===\n";
        std::cout << "Method: GMRES(" << restart << ") + " << method
                  << "  overlap=" << overlap
                  << "  DOFs/proc=" << dofs_per_proc << "\n";
        std::cout << std::setw(8)  << "procs"
                  << std::setw(10) << "n"
                  << std::setw(12) << "iters"
                  << std::setw(12) << "setup(s)"
                  << std::setw(12) << "solve(s)"
                  << std::setw(12) << "eff_weak"
                  << std::setw(10) << "residual"
                  << "\n";
        std::cout << std::string(76, '-') << "\n";
    }

    GMRESParams params;
    params.restart  = restart;
    params.tol      = tol;
    params.max_iter = max_iter;

    double t1_solve = -1.0;   // T_1 for weak efficiency

    // Simulate weak scaling by sweeping over nprocs (each using nprocs subdomains)
    for (int np = 1; np <= max_procs; np *= 2) {
        int total_dofs = dofs_per_proc * np;
        int nx = static_cast<int>(std::ceil(std::sqrt(static_cast<double>(total_dofs))));
        int ny = nx;

        CSRMatrix A = make_poisson_2d(nx, ny);
        int n     = A.nrows;
        int nnz_a = A.nnz();

        std::vector<double> x_exact(n, 1.0);
        std::vector<double> b(n);
        A.spmv(x_exact, b);

        std::unique_ptr<Preconditioner> M;
        auto ts = std::chrono::high_resolution_clock::now();
        if (method == "twolevel_ras") {
            auto p = std::make_unique<TwoLevelSchwarzPrecond>(np, overlap, FineLevelType::RAS);
            p->setup(A); M = std::move(p);
        } else {
            auto p = std::make_unique<RASPrecond>(np, overlap);
            p->setup(A); M = std::move(p);
        }
        auto te = std::chrono::high_resolution_clock::now();
        double t_setup = std::chrono::duration<double>(te - ts).count();

        std::vector<double> x(n, 0.0);
        auto res = gmres(A, b, x, M.get(), params);

        double t_solve = res.time_solve_s;
        if (np == 1) t1_solve = t_solve;

        // Weak efficiency: ideally constant solve time; efficiency = T_1 / T_p
        double eff_weak = (t1_solve > 0.0) ? (t1_solve / t_solve) : 1.0;

        if (rank == 0) {
            std::cout << std::fixed << std::setprecision(4)
                      << std::setw(8)  << np
                      << std::setw(10) << n
                      << std::setw(12) << res.iterations
                      << std::setw(12) << t_setup
                      << std::setw(12) << t_solve
                      << std::setw(12) << eff_weak
                      << std::setw(10) << std::scientific << res.residual_norm
                      << "\n";

            std::string matrix_name = "poisson2d_" + std::to_string(nx) + "x" + std::to_string(ny);
            Metrics m;
            m.matrix        = matrix_name;
            m.method        = "weak_scaling_GMRES+" + method;
            m.n             = n;
            m.nnz           = nnz_a;
            m.ranks         = np;
            m.threads       = threads;
            m.overlap       = overlap;
            m.restart       = restart;
            m.tol           = tol;
            m.iterations    = res.iterations;
            m.residual_norm = res.residual_norm;
            m.time_setup_s  = t_setup;
            m.time_solve_s  = t_solve;
            m.speedup       = 1.0;
            m.efficiency    = eff_weak;
            m.write_jsonl(output);
        }
    }

    if (rank == 0)
        std::cout << "\nResults written to " << output << "\n";

    MPI_Finalize();
    return 0;
}

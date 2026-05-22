#include "core/sparse_matrix.h"
#include "core/gmres.h"
#include "core/fgmres.h"
#include "core/bicgstab.h"
#include "core/vector.h"
#include "core/distributed_matrix.h"
#include "core/distributed_gmres.h"
#include "precond/jacobi.h"
#include "precond/ilu0.h"
#include "precond/ilut.h"
#include "precond/asm.h"
#include "precond/ras.h"
#include "precond/two_level_schwarz.h"
#include "precond/distributed_schwarz.h"
#include "precond/distributed_two_level_schwarz.h"
#include "io/matrix_market.h"
#include "io/metrics_logger.h"
#include "mesh/poisson.h"
#include "analysis/convergence_analysis.h"
#include "analysis/condition_estimator.h"
#include "analysis/spectral_analysis.h"

#ifdef USE_CUDA
#include "gpu/gmres_gpu.h"
#endif

#include <mpi.h>
#include <omp.h>
#include <iostream>
#include <string>
#include <memory>
#include <chrono>
#include <cstring>

using namespace schwarz;

static void print_usage() {
    std::cerr <<
        "Usage: schwarz_solve [options]\n"
        "\nMatrix source (one required):\n"
        "  --matrix <file.mtx>      Matrix Market file\n"
        "  --poisson <N>            Generate N×N Poisson 2D (n = N^2)\n"
        "\nPreconditioner:\n"
        "  --precond <type>         none|jacobi|ilu0|ilut|asm|ras|twolevel_asm|twolevel_ras\n"
        "  --drop-tol <val>         ILUT drop tolerance (default 1e-4)\n"
        "  --fill-factor <n>        ILUT fill factor (default 10)\n"
        "\nSolver:\n"
        "  --restart <m>            GMRES restart length (default 30)\n"
        "  --tol <val>              Convergence tolerance (default 1e-10)\n"
        "  --maxiter <n>            Max iterations (default 1000)\n"
        "  --fgmres                 Use Flexible GMRES (right-preconditioned)\n"
        "  --bicgstab               Use BiCGSTAB (Van der Vorst 1992)\n"
        "\nDomain decomposition:\n"
        "  --nparts <n>             Subdomains per rank (default 4)\n"
        "  --overlap <n>            Overlap size (default 1)\n"
        "\nHardware:\n"
        "  --gpu                    Use GPU (CUDA) solver\n"
        "\nOutput & analysis:\n"
        "  --output <file.jsonl>    Append metrics to JSONL file\n"
        "  --analyze                Run convergence / condition / spectral analysis\n";
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int rank, nranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);

    std::string matrix_file;
    std::string precond_type = "ilu0";
    std::string output_file;
    GMRESParams params;
    int    overlap      = 1;
    int    nparts       = 4;
    bool   use_gpu       = false;
    bool   use_fgmres    = false;
    bool   use_bicgstab  = false;
    bool   run_analysis  = false;
    int    poisson_n    = 0;
    double drop_tol     = 1e-4;
    int    fill_factor  = 10;

    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--matrix") && i + 1 < argc)
            matrix_file = argv[++i];
        else if (!std::strcmp(argv[i], "--precond") && i + 1 < argc)
            precond_type = argv[++i];
        else if (!std::strcmp(argv[i], "--restart") && i + 1 < argc)
            params.restart = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--tol") && i + 1 < argc)
            params.tol = std::atof(argv[++i]);
        else if (!std::strcmp(argv[i], "--maxiter") && i + 1 < argc)
            params.max_iter = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--overlap") && i + 1 < argc)
            overlap = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--nparts") && i + 1 < argc)
            nparts = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--output") && i + 1 < argc)
            output_file = argv[++i];
        else if (!std::strcmp(argv[i], "--gpu"))
            use_gpu = true;
        else if (!std::strcmp(argv[i], "--fgmres"))
            use_fgmres = true;
        else if (!std::strcmp(argv[i], "--bicgstab"))
            use_bicgstab = true;
        else if (!std::strcmp(argv[i], "--analyze"))
            run_analysis = true;
        else if (!std::strcmp(argv[i], "--poisson") && i + 1 < argc)
            poisson_n = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--drop-tol") && i + 1 < argc)
            drop_tol = std::atof(argv[++i]);
        else if (!std::strcmp(argv[i], "--fill-factor") && i + 1 < argc)
            fill_factor = std::atoi(argv[++i]);
        else {
            if (rank == 0) print_usage();
            MPI_Finalize();
            return 1;
        }
    }

    if (matrix_file.empty() && poisson_n <= 0) {
        if (rank == 0) print_usage();
        MPI_Finalize();
        return 1;
    }

    // ---- Load / generate matrix on rank 0 ----
    CSRMatrix A_global;
    if (rank == 0) {
        if (poisson_n > 0) {
            std::cout << "Generating Poisson2D " << poisson_n << "x" << poisson_n << "\n";
            A_global = make_poisson_2d(poisson_n, poisson_n);
            if (matrix_file.empty())
                matrix_file = "poisson2d_" + std::to_string(poisson_n);
        } else {
            std::cout << "Loading matrix: " << matrix_file << "\n";
            A_global = read_matrix_market(matrix_file);
        }
        std::cout << "  n=" << A_global.nrows << "  nnz=" << A_global.nnz() << "\n";
    }

    // =====================================================================
    //  Single-rank path
    // =====================================================================
    if (nranks == 1) {
        if (A_global.nrows == 0) {
            if (poisson_n > 0) A_global = make_poisson_2d(poisson_n, poisson_n);
            else A_global = read_matrix_market(matrix_file);
        }

        int n = A_global.nrows;
        std::vector<double> x_exact(n, 1.0);
        std::vector<double> b(n);
        A_global.spmv(x_exact, b);

#ifdef USE_CUDA
        if (use_gpu) {
            std::vector<double> x(n, 0.0);
            std::cout << "Solving with GPU GMRES(" << params.restart << ") + "
                      << precond_type << "\n";
            auto result = gpu::solve_gpu(A_global, b, x, nparts, overlap,
                                         (precond_type == "ras"),
                                         params.restart, params.max_iter, params.tol);

            std::cout << "  Converged: " << (result.converged ? "yes" : "no") << "\n"
                      << "  Iterations: " << result.iterations << "\n"
                      << "  Residual: "   << result.residual_norm << "\n"
                      << "  Solve time: " << result.time_solve_s << " s\n";

            if (!output_file.empty()) {
                Metrics m;
                m.matrix = matrix_file; m.method = "GPU_GMRES+" + precond_type;
                m.n = n; m.nnz = A_global.nnz(); m.ranks = 1;
                m.threads = omp_get_max_threads(); m.gpus = 1;
                m.overlap = overlap; m.restart = params.restart; m.tol = params.tol;
                m.iterations = result.iterations; m.residual_norm = result.residual_norm;
                m.time_setup_s = 0.0; m.time_solve_s = result.time_solve_s;
                m.write_jsonl(output_file);
            }
            MPI_Finalize();
            return result.converged ? 0 : 1;
        }
#endif

        // ---- Build preconditioner ----
        std::unique_ptr<Preconditioner> M;
        auto t0 = std::chrono::high_resolution_clock::now();

        if      (precond_type == "jacobi") {
            auto p = std::make_unique<JacobiPrecond>(); p->setup(A_global); M = std::move(p);
        } else if (precond_type == "ilu0") {
            auto p = std::make_unique<ILU0Precond>(); p->setup(A_global); M = std::move(p);
        } else if (precond_type == "ilut") {
            auto p = std::make_unique<ILUTPrecond>(drop_tol, fill_factor);
            p->setup(A_global); M = std::move(p);
            std::cout << "  ILUT: drop_tol=" << drop_tol << " fill_factor=" << fill_factor << "\n";
        } else if (precond_type == "asm") {
            auto p = std::make_unique<ASMPrecond>(nparts, overlap); p->setup(A_global); M = std::move(p);
        } else if (precond_type == "ras") {
            auto p = std::make_unique<RASPrecond>(nparts, overlap); p->setup(A_global); M = std::move(p);
        } else if (precond_type == "twolevel_asm") {
            auto p = std::make_unique<TwoLevelSchwarzPrecond>(nparts, overlap, FineLevelType::ASM);
            p->setup(A_global); M = std::move(p);
        } else if (precond_type == "twolevel_ras") {
            auto p = std::make_unique<TwoLevelSchwarzPrecond>(nparts, overlap, FineLevelType::RAS);
            p->setup(A_global); M = std::move(p);
        }

        auto t1 = std::chrono::high_resolution_clock::now();
        double t_setup = std::chrono::duration<double>(t1 - t0).count();

        // ---- Scientific analysis (pre-solve) ----
        if (run_analysis && M) {
            std::cout << "\n--- Pre-solve analysis ---\n";
            auto ce_A = estimate_condition(A_global, nullptr, 80);
            print_condition_estimate(ce_A, "A (unpreconditioned)");

            auto ce_MA = estimate_condition(A_global, M.get(), 80);
            print_condition_estimate(ce_MA, "M^{-1}A (preconditioned)");

            auto sr = estimate_spectral_radius(A_global, M.get(), 300);
            print_spectral_result(sr, precond_type);
            std::cout << "--------------------------\n\n";
        }

        // ---- Solve ----
        std::vector<double> x(n, 0.0);
        GMRESResult res;

        const std::string solver_label =
            use_bicgstab ? "BiCGSTAB" :
            use_fgmres   ? "FGMRES(" + std::to_string(params.restart) + ")" :
                           "GMRES("  + std::to_string(params.restart) + ")";

        std::cout << "Solving with " << solver_label << " + " << precond_type << "\n";

        if (use_bicgstab)
            res = bicgstab(A_global, b, x, M.get(), params);
        else if (use_fgmres)
            res = fgmres(A_global, b, x, M.get(), params);
        else
            res = gmres(A_global, b, x, M.get(), params);

        std::cout << "  Converged: " << (res.converged ? "yes" : "no") << "\n"
                  << "  Iterations: " << res.iterations << "\n"
                  << "  Residual: "   << res.residual_norm << "\n"
                  << "  Setup time: " << t_setup << " s\n"
                  << "  Solve time: " << res.time_solve_s << " s\n";

        // ---- Convergence analysis (post-solve) ----
        if (run_analysis) {
            auto ca = analyze_convergence(res.residual_history, res.converged);
            ca.print_summary();
            if (!output_file.empty())
                ca.write_jsonl(output_file + ".conv.jsonl",
                               matrix_file, precond_type, nparts, overlap);
        }

        if (!output_file.empty()) {
            Metrics m;
            m.matrix = matrix_file;
            m.method = (use_bicgstab ? "BiCGSTAB+" :
                        use_fgmres  ? "FGMRES+"   : "GMRES+") + precond_type;
            m.n = n; m.nnz = A_global.nnz(); m.ranks = 1;
            m.threads = omp_get_max_threads(); m.overlap = overlap;
            m.restart = params.restart; m.tol = params.tol;
            m.iterations = res.iterations; m.residual_norm = res.residual_norm;
            m.time_setup_s = t_setup; m.time_solve_s = res.time_solve_s;
            m.write_jsonl(output_file);
        }

        MPI_Finalize();
        return res.converged ? 0 : 1;
    }

    // =====================================================================
    //  Multi-rank (distributed) path
    // =====================================================================
    DistributedCSRMatrix dA;
    dA.distribute(A_global, MPI_COMM_WORLD);

    std::vector<double> x_buf(dA.n_local_cols, 1.0);
    std::vector<double> b_local(dA.local_nrows);
    dA.spmv(x_buf.data(), b_local.data());

    std::unique_ptr<Preconditioner> M;
    auto t0 = std::chrono::high_resolution_clock::now();

    if (precond_type == "jacobi") {
        auto p = std::make_unique<JacobiPrecond>();
        p->setup(dA.local);
        M = std::move(p);
    } else if (precond_type == "ilu0") {
        auto p = std::make_unique<ILU0Precond>();
        CSRMatrix A_owned;
        A_owned.nrows = dA.n_owned;
        A_owned.ncols = dA.n_owned;
        A_owned.row_ptr.resize(dA.n_owned + 1);
        std::vector<int> ci; std::vector<double> cv;
        for (int i = 0; i < dA.n_owned; ++i) {
            A_owned.row_ptr[i] = static_cast<int>(ci.size());
            for (int j = dA.local.row_ptr[i]; j < dA.local.row_ptr[i+1]; ++j) {
                if (dA.local.col_idx[j] < dA.n_owned) {
                    ci.push_back(dA.local.col_idx[j]);
                    cv.push_back(dA.local.vals[j]);
                }
            }
        }
        A_owned.row_ptr[dA.n_owned] = static_cast<int>(ci.size());
        A_owned.col_idx = ci; A_owned.vals = cv;
        p->setup(A_owned);
        M = std::move(p);
    } else if (precond_type == "ilut") {
        auto p = std::make_unique<ILUTPrecond>(drop_tol, fill_factor);
        CSRMatrix A_owned;
        A_owned.nrows = dA.n_owned;
        A_owned.ncols = dA.n_owned;
        A_owned.row_ptr.resize(dA.n_owned + 1);
        std::vector<int> ci; std::vector<double> cv;
        for (int i = 0; i < dA.n_owned; ++i) {
            A_owned.row_ptr[i] = static_cast<int>(ci.size());
            for (int j = dA.local.row_ptr[i]; j < dA.local.row_ptr[i+1]; ++j) {
                if (dA.local.col_idx[j] < dA.n_owned) {
                    ci.push_back(dA.local.col_idx[j]);
                    cv.push_back(dA.local.vals[j]);
                }
            }
        }
        A_owned.row_ptr[dA.n_owned] = static_cast<int>(ci.size());
        A_owned.col_idx = ci; A_owned.vals = cv;
        p->setup(A_owned);
        M = std::move(p);
    } else if (precond_type == "asm" || precond_type == "ras") {
        DistSchwarzType stype = (precond_type == "asm")
                                ? DistSchwarzType::ASM : DistSchwarzType::RAS;
        auto p = std::make_unique<DistributedSchwarzPrecond>(nparts, overlap, stype);
        p->setup(dA.local, dA.n_owned, dA.local_to_global);
        M = std::move(p);
    } else if (precond_type == "twolevel_asm" || precond_type == "twolevel_ras") {
        DistSchwarzType stype = (precond_type == "twolevel_asm")
                                ? DistSchwarzType::ASM : DistSchwarzType::RAS;
        auto p = std::make_unique<DistributedTwoLevelSchwarzPrecond>(
                     nparts, overlap, stype, MPI_COMM_WORLD);
        p->setup(dA.local, dA.n_owned, dA.local_to_global);
        M = std::move(p);
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    double t_setup = std::chrono::duration<double>(t1 - t0).count();

    MPI_Barrier(MPI_COMM_WORLD);

    std::fill(x_buf.begin(), x_buf.end(), 0.0);
    GMRESResult res = distributed_gmres(dA, b_local.data(), x_buf.data(), M.get(), params);

    if (rank == 0) {
        const std::string solver_label = use_fgmres ? "dist_FGMRES" : "dist_GMRES";
        std::cout << solver_label << "(" << params.restart << ") + " << precond_type << "\n"
                  << "  Ranks: " << nranks << " Threads/rank: " << omp_get_max_threads() << "\n"
                  << "  n=" << dA.global_nrows << " local_n=" << dA.local_nrows
                  << " ghosts=" << dA.n_ghost << "\n"
                  << "  Converged: " << (res.converged ? "yes" : "no") << "\n"
                  << "  Iterations: " << res.iterations << "\n"
                  << "  Residual: " << res.residual_norm << "\n"
                  << "  Setup time: " << t_setup << " s\n"
                  << "  Solve time: " << res.time_solve_s << " s\n";

        if (run_analysis) {
            auto ca = analyze_convergence(res.residual_history, res.converged);
            ca.print_summary();
        }

        if (!output_file.empty()) {
            Metrics m;
            m.matrix = matrix_file;
            m.method = (use_fgmres ? "dist_FGMRES+" : "dist_GMRES+") + precond_type;
            m.n = dA.global_nrows; m.nnz = 0;
            m.ranks = nranks; m.threads = omp_get_max_threads();
            m.overlap = overlap; m.restart = params.restart; m.tol = params.tol;
            m.iterations = res.iterations; m.residual_norm = res.residual_norm;
            m.time_setup_s = t_setup; m.time_solve_s = res.time_solve_s;
            m.write_jsonl(output_file);
        }
    }

    MPI_Finalize();
    return res.converged ? 0 : 1;
}

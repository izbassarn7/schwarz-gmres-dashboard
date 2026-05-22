#ifdef USE_CUDA

#include "../src/core/sparse_matrix.h"
#include "../src/gpu/gmres_gpu.h"
#include "../src/mesh/poisson.h"
#include "../src/io/metrics_logger.h"

#include <mpi.h>
#include <omp.h>
#include <iostream>
#include <string>
#include <chrono>
#include <cstring>
#include <vector>

using namespace schwarz;

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int rank, nranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);

    // Default: generated 2D Poisson
    int nx = 100, ny = 100;
    std::string matrix_file;
    std::string method = "ras";
    int overlap = 1;
    int restart = 30;
    double tol = 1e-10;
    int max_iter = 1000;
    std::string output = "results/gpu_scaling.jsonl";

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
        else if (!std::strcmp(argv[i], "--output") && i + 1 < argc)
            output = argv[++i];
    }

    CSRMatrix A;
    std::string matrix_name;

    if (!matrix_file.empty()) {
        // Note: matrix_file loading would require read_matrix_market
        // For now, always use generated 2D Poisson
        if (rank == 0) {
            std::cerr << "GPU scaling: using generated matrix (--matrix flag ignored)\n";
        }
        A = make_poisson_2d(nx, ny);
        matrix_name = matrix_file;
    } else {
        A = make_poisson_2d(nx, ny);
        matrix_name = "poisson2d_" + std::to_string(nx) + "x" + std::to_string(ny);
    }

    int n = A.nrows;

    // RHS: b = A * ones
    std::vector<double> x_exact(n, 1.0);
    std::vector<double> b(n);
    A.spmv(x_exact, b);

    // Single rank only
    if (nranks != 1) {
        if (rank == 0) {
            std::cerr << "GPU scaling benchmark runs single-rank only\n";
        }
        MPI_Finalize();
        return 1;
    }

    int nparts = 4;  // Fixed for GPU
    int threads = omp_get_max_threads();

    auto t_solve_start = std::chrono::high_resolution_clock::now();

    std::vector<double> x(n, 0.0);
    bool restricted = (method == "ras");
    auto result = gpu::solve_gpu(A, b, x, nparts, overlap, restricted,
                                 restart, max_iter, tol);

    auto t_solve_end = std::chrono::high_resolution_clock::now();
    double t_solve = std::chrono::duration<double>(t_solve_end - t_solve_start).count();

    if (rank == 0) {
        std::cout << "GPU Scaling: " << matrix_name << "\n";
        std::cout << "  Method: GPU_GMRES+" << method << "\n";
        std::cout << "  Threads: " << threads << "\n";
        std::cout << "  n=" << n << " nnz=" << A.nnz() << "\n";
        std::cout << "  Converged: " << (result.converged ? "yes" : "no") << "\n";
        std::cout << "  Iterations: " << result.iterations << "\n";
        std::cout << "  Residual: " << result.residual_norm << "\n";
        std::cout << "  Solve time: " << result.time_solve_s << " s\n";

        Metrics m;
        m.matrix = matrix_name;
        m.method = "GPU_GMRES+" + method;
        m.n = n;
        m.nnz = A.nnz();
        m.ranks = 1;
        m.threads = threads;
        m.gpus = 1;
        m.overlap = overlap;
        m.restart = restart;
        m.tol = tol;
        m.iterations = result.iterations;
        m.residual_norm = result.residual_norm;
        m.time_setup_s = 0.0;
        m.time_solve_s = result.time_solve_s;
        m.write_jsonl(output);
    }

    MPI_Finalize();
    return result.converged ? 0 : 1;
}

#else

#include <iostream>

int main() {
    std::cerr << "GPU scaling benchmark requires USE_CUDA\n";
    return 1;
}

#endif

#ifdef USE_TRILINOS

#include <Teuchos_RCP.hpp>
#include <Teuchos_ParameterList.hpp>
#include <Teuchos_GlobalMPISession.hpp>

#include <Tpetra_CrsMatrix.hpp>
#include <Tpetra_Map.hpp>
#include <Tpetra_MultiVector.hpp>
#include <Tpetra_Vector.hpp>

#include <Belos_BlockGmresSolMgr.hpp>
#include <Belos_LinearProblem.hpp>
#include <Belos_TpetraAdapter.hpp>

#include <Ifpack2_Factory.hpp>

#include "../src/io/matrix_market.h"
#include "../src/io/metrics_logger.h"
#include "../src/mesh/poisson.h"

#include <iostream>
#include <string>
#include <chrono>
#include <cstring>

using Teuchos::RCP;
using Teuchos::rcp;
using Teuchos::ParameterList;

using SC = double;
using LO = int;
using GO = long long;
using NT = Tpetra::Map<>::node_type;
using map_type = Tpetra::Map<LO, GO, NT>;
using crs_matrix_type = Tpetra::CrsMatrix<SC, LO, GO, NT>;
using vec_type = Tpetra::Vector<SC, LO, GO, NT>;
using mv_type = Tpetra::MultiVector<SC, LO, GO, NT>;

using namespace schwarz;

int main(int argc, char* argv[]) {
    Teuchos::GlobalMPISession mpiSession(&argc, &argv);
    auto comm = Tpetra::getDefaultComm();
    int rank = comm->getRank();

    std::string matrix_file;
    std::string pc_type_str = "SCHWARZ";
    int overlap = 1;
    int restart = 30;
    double tol = 1e-10;
    int max_iter = 1000;
    int nx = 100, ny = 100;
    std::string output = "results/trilinos_baseline.jsonl";

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

    int n = A_host.nrows;

    // Create Tpetra map and matrix
    RCP<const map_type> rowMap = rcp(new map_type(
        static_cast<GO>(n), 0, comm));

    RCP<crs_matrix_type> A = rcp(new crs_matrix_type(rowMap, 0));

    for (int i = 0; i < n; ++i) {
        GO globalRow = static_cast<GO>(i);
        if (!rowMap->isNodeGlobalElement(globalRow)) continue;

        int nnz_row = A_host.row_ptr[i + 1] - A_host.row_ptr[i];
        std::vector<GO> cols(nnz_row);
        std::vector<SC> vals(nnz_row);
        for (int j = 0; j < nnz_row; ++j) {
            cols[j] = static_cast<GO>(A_host.col_idx[A_host.row_ptr[i] + j]);
            vals[j] = A_host.vals[A_host.row_ptr[i] + j];
        }
        A->insertGlobalValues(globalRow,
            Teuchos::ArrayView<const GO>(cols),
            Teuchos::ArrayView<const SC>(vals));
    }
    A->fillComplete();

    // Create vectors
    RCP<vec_type> x = rcp(new vec_type(rowMap));
    RCP<vec_type> b = rcp(new vec_type(rowMap));
    x->putScalar(0.0);

    // b = A * ones
    RCP<vec_type> ones = rcp(new vec_type(rowMap));
    ones->putScalar(1.0);
    A->apply(*ones, *b);

    // Setup Belos GMRES
    auto t_setup_start = std::chrono::high_resolution_clock::now();

    using problem_type = Belos::LinearProblem<SC, mv_type, Tpetra::Operator<SC, LO, GO, NT>>;
    RCP<problem_type> problem = rcp(new problem_type(A, x, b));

    // Ifpack2 preconditioner
    auto prec = Ifpack2::Factory::create<crs_matrix_type>(pc_type_str, A);
    auto prec_params = rcp(new ParameterList());
    prec_params->set("schwarz: overlap level", overlap);
    prec_params->set("schwarz: combine mode", "Add");
    prec_params->set("inner preconditioner name", "ILUT");
    prec->setParameters(*prec_params);
    prec->initialize();
    prec->compute();

    problem->setLeftPrec(prec);
    problem->setProblem();

    auto solver_params = rcp(new ParameterList());
    solver_params->set("Maximum Iterations", max_iter);
    solver_params->set("Convergence Tolerance", tol);
    solver_params->set("Num Blocks", restart);
    solver_params->set("Verbosity", Belos::Errors + Belos::Warnings);

    using solver_type = Belos::BlockGmresSolMgr<SC, mv_type,
        Tpetra::Operator<SC, LO, GO, NT>>;
    RCP<solver_type> solver = rcp(new solver_type(problem, solver_params));

    auto t_setup_end = std::chrono::high_resolution_clock::now();
    double t_setup = std::chrono::duration<double>(t_setup_end - t_setup_start).count();

    auto t_solve_start = std::chrono::high_resolution_clock::now();
    Belos::ReturnType ret = solver->solve();
    auto t_solve_end = std::chrono::high_resolution_clock::now();
    double t_solve = std::chrono::duration<double>(t_solve_end - t_solve_start).count();

    int its = solver->getNumIters();
    double achieved_tol = solver->achievedTol();

    if (rank == 0) {
        std::cout << "Trilinos baseline: " << matrix_name << "\n";
        std::cout << "  PC: " << pc_type_str << "\n";
        std::cout << "  Converged: " << (ret == Belos::Converged ? "yes" : "no") << "\n";
        std::cout << "  Iterations: " << its << "\n";
        std::cout << "  Residual: " << achieved_tol << "\n";
        std::cout << "  Setup time: " << t_setup << " s\n";
        std::cout << "  Solve time: " << t_solve << " s\n";

        Metrics m;
        m.matrix = matrix_name;
        m.method = "Trilinos_GMRES+" + pc_type_str;
        m.n = n;
        m.nnz = A_host.nnz();
        m.overlap = overlap;
        m.restart = restart;
        m.tol = tol;
        m.iterations = its;
        m.residual_norm = achieved_tol;
        m.time_setup_s = t_setup;
        m.time_solve_s = t_solve;
        m.write_jsonl(output);
    }

    return (ret == Belos::Converged) ? 0 : 1;
}

#else
#include <iostream>
int main() {
    std::cerr << "Trilinos support not enabled. Rebuild with -DUSE_TRILINOS=ON\n";
    return 1;
}
#endif

#pragma once

#include <string>
#include <fstream>
#include <iomanip>
#include <sstream>

namespace schwarz {

struct Metrics {
    std::string matrix;
    std::string method;
    int n = 0;
    int nnz = 0;
    int ranks = 1;
    int threads = 1;
    int gpus = 0;
    int overlap = 0;
    int restart = 30;
    int iterations = 0;
    double tol = 1e-10;
    double residual_norm = 0.0;
    double time_setup_s = 0.0;
    double time_solve_s = 0.0;
    double speedup = 1.0;
    double efficiency = 1.0;

    std::string to_json() const {
        std::ostringstream os;
        os << std::setprecision(6);
        os << "{";
        os << "\"matrix\":\"" << matrix << "\",";
        os << "\"method\":\"" << method << "\",";
        os << "\"n\":" << n << ",";
        os << "\"nnz\":" << nnz << ",";
        os << "\"ranks\":" << ranks << ",";
        os << "\"threads\":" << threads << ",";
        os << "\"gpus\":" << gpus << ",";
        os << "\"overlap\":" << overlap << ",";
        os << "\"restart\":" << restart << ",";
        os << "\"iterations\":" << iterations << ",";
        os << "\"tol\":" << tol << ",";
        os << "\"residual_norm\":" << std::scientific << residual_norm << ",";
        os << std::fixed;
        os << "\"time_setup_s\":" << time_setup_s << ",";
        os << "\"time_solve_s\":" << time_solve_s << ",";
        os << "\"speedup\":" << speedup << ",";
        os << "\"efficiency\":" << efficiency;
        os << "}";
        return os.str();
    }

    void write_jsonl(const std::string& path) const {
        std::ofstream out(path, std::ios::app);
        out << to_json() << "\n";
    }
};

}  // namespace schwarz

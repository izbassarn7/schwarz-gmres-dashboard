#pragma once

#include <vector>
#include <cmath>
#include <string>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <algorithm>
#include <numeric>
#include <stdexcept>

namespace schwarz {

// Per-iteration convergence metrics for a single GMRES run.
struct ConvergenceMetrics {
    std::vector<double> residuals;      // ||r_k|| at each iteration
    std::vector<double> rates;          // rho_k = (r_k / r_0)^(1/k)
    double initial_residual = 0.0;
    double final_residual   = 0.0;
    double asymptotic_rate  = 0.0;     // average of last 20 rates
    int    stagnation_start = -1;      // iteration where stagnation begins (-1 = none)
    bool   converged        = false;
    int    iterations       = 0;

    static constexpr int STAGNATION_WINDOW = 20;
    static constexpr double STAGNATION_TOL = 1e-3;  // < 0.1% reduction over window

    void compute(const std::vector<double>& res_history, bool conv) {
        residuals  = res_history;
        converged  = conv;
        iterations = static_cast<int>(residuals.size());

        if (residuals.empty()) return;

        initial_residual = residuals.front();
        final_residual   = residuals.back();

        // Per-iteration convergence rate rho_k = (r_k / r_0)^(1/k)
        rates.resize(iterations);
        for (int k = 1; k <= iterations; ++k) {
            double ratio = residuals[k - 1] / (initial_residual + 1e-300);
            rates[k - 1] = (ratio > 0.0) ? std::pow(ratio, 1.0 / k) : 0.0;
        }

        // Asymptotic convergence rate from last 20 iterations
        int tail = std::min(STAGNATION_WINDOW, iterations);
        if (tail >= 2) {
            asymptotic_rate = std::accumulate(
                rates.end() - tail, rates.end(), 0.0) / tail;
        } else {
            asymptotic_rate = rates.back();
        }

        // Stagnation detection: flag if relative reduction < STAGNATION_TOL
        // over a window of STAGNATION_WINDOW consecutive iterations.
        for (int i = STAGNATION_WINDOW; i < iterations; ++i) {
            double r_start = residuals[i - STAGNATION_WINDOW];
            double r_end   = residuals[i];
            double rel_red = (r_start - r_end) / (r_start + 1e-300);
            if (rel_red < STAGNATION_TOL) {
                stagnation_start = i - STAGNATION_WINDOW + 1;
                break;
            }
        }
    }

    // Append a JSONL line with convergence metrics.
    void write_jsonl(const std::string& path,
                     const std::string& matrix,
                     const std::string& precond,
                     int nparts, int overlap) const
    {
        std::ofstream out(path, std::ios::app);
        out << std::setprecision(6);
        out << "{\"type\":\"convergence\""
            << ",\"matrix\":\"" << matrix << "\""
            << ",\"precond\":\"" << precond << "\""
            << ",\"nparts\":" << nparts
            << ",\"overlap\":" << overlap
            << ",\"iterations\":" << iterations
            << ",\"converged\":" << (converged ? "true" : "false")
            << ",\"initial_residual\":" << initial_residual
            << ",\"final_residual\":" << final_residual
            << ",\"asymptotic_rate\":" << asymptotic_rate
            << ",\"stagnation_start\":" << stagnation_start
            << ",\"residual_history\":[";
        for (int i = 0; i < static_cast<int>(residuals.size()); ++i) {
            if (i > 0) out << ",";
            out << residuals[i];
        }
        out << "]}\n";
    }

    void print_summary() const {
        std::cout << "  Convergence analysis:\n"
                  << "    Initial residual : " << initial_residual << "\n"
                  << "    Final residual   : " << final_residual   << "\n"
                  << "    Asymptotic rate  : " << asymptotic_rate  << "\n";
        if (stagnation_start >= 0)
            std::cout << "    *** Stagnation detected at iter " << stagnation_start << " ***\n";
        else
            std::cout << "    No stagnation detected\n";
    }
};

// Wrap a GMRES result (residual_history already stored in GMRESResult)
// and compute convergence metrics.
inline ConvergenceMetrics analyze_convergence(
    const std::vector<double>& residual_history, bool converged)
{
    ConvergenceMetrics m;
    m.compute(residual_history, converged);
    return m;
}

}  // namespace schwarz

#include "gmres.h"
#include <chrono>
#include <cstring>

namespace schwarz {

namespace {

struct GivensRotation {
    double c, s;
    void compute(double a, double b) {
        if (std::abs(b) < 1e-30) {
            c = 1.0; s = 0.0;
        } else if (std::abs(b) > std::abs(a)) {
            double t = -a / b;
            s = 1.0 / std::sqrt(1.0 + t * t);
            c = s * t;
        } else {
            double t = -b / a;
            c = 1.0 / std::sqrt(1.0 + t * t);
            s = c * t;
        }
    }
    void apply(double& h1, double& h2) const {
        double t1 = c * h1 - s * h2;
        double t2 = s * h1 + c * h2;
        h1 = t1;
        h2 = t2;
    }
};

}  // namespace

GMRESResult gmres(
    const CSRMatrix& A,
    const std::vector<double>& b,
    std::vector<double>& x,
    const Preconditioner* M,
    const GMRESParams& params)
{
    auto t0 = std::chrono::high_resolution_clock::now();

    int n = A.nrows;
    int m = params.restart;
    GMRESResult result;

    std::vector<double> r(n), w(n), z(n), Av(n);
    std::vector<std::vector<double>> V(m + 1, std::vector<double>(n));
    // Upper Hessenberg stored column-major: H[i + j*(m+1)]
    std::vector<double> H((m + 1) * m, 0.0);
    std::vector<double> g(m + 1, 0.0);
    std::vector<GivensRotation> rots(m);
    std::vector<double> y(m);

    int total_iter = 0;

    for (int cycle = 0; cycle < params.max_iter; ++cycle) {
        // r = b - A*x
        A.spmv(x.data(), r.data());
        for (int i = 0; i < n; ++i)
            r[i] = b[i] - r[i];

        // z = M^{-1} r
        if (M)
            M->apply(r.data(), z.data(), n);
        else
            std::copy(r.begin(), r.end(), z.begin());

        double beta = norm2(z.data(), n);
        if (beta < params.tol) {
            result.converged = true;
            result.residual_norm = beta;
            break;
        }

        for (int i = 0; i < n; ++i)
            V[0][i] = z[i] / beta;

        std::fill(H.begin(), H.end(), 0.0);
        std::fill(g.begin(), g.end(), 0.0);
        g[0] = beta;

        int j;
        for (j = 0; j < m; ++j) {
            // w = M^{-1} * A * V[j]
            A.spmv(V[j].data(), Av.data());
            if (M)
                M->apply(Av.data(), w.data(), n);
            else
                std::copy(Av.begin(), Av.end(), w.begin());

            // Modified Gram-Schmidt
            for (int i = 0; i <= j; ++i) {
                double h = dot(w.data(), V[i].data(), n);
                H[i + j * (m + 1)] = h;
                axpy(-h, V[i].data(), w.data(), n);
            }

            double h_jp1 = norm2(w.data(), n);
            H[(j + 1) + j * (m + 1)] = h_jp1;

            if (h_jp1 < 1e-30) {
                j++;
                break;
            }

            for (int i = 0; i < n; ++i)
                V[j + 1][i] = w[i] / h_jp1;

            // Apply previous Givens rotations to new column
            for (int i = 0; i < j; ++i)
                rots[i].apply(H[i + j * (m + 1)], H[(i + 1) + j * (m + 1)]);

            // Compute new rotation
            rots[j].compute(H[j + j * (m + 1)], H[(j + 1) + j * (m + 1)]);
            rots[j].apply(H[j + j * (m + 1)], H[(j + 1) + j * (m + 1)]);
            rots[j].apply(g[j], g[j + 1]);

            double res = std::abs(g[j + 1]);
            result.residual_history.push_back(res);
            total_iter++;

            if (res < params.tol) {
                j++;
                result.converged = true;
                break;
            }
        }

        // Solve upper triangular system H*y = g
        int k = j;
        for (int i = k - 1; i >= 0; --i) {
            y[i] = g[i];
            for (int l = i + 1; l < k; ++l)
                y[i] -= H[i + l * (m + 1)] * y[l];
            y[i] /= H[i + i * (m + 1)];
        }

        // x = x + V*y
        for (int i = 0; i < k; ++i)
            axpy(y[i], V[i].data(), x.data(), n);

        if (result.converged)
            break;

        if (total_iter >= params.max_iter)
            break;
    }

    result.iterations = total_iter;
    if (result.residual_history.empty()) {
        result.residual_norm = 0.0;
    } else {
        result.residual_norm = result.residual_history.back();
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    result.time_solve_s = std::chrono::duration<double>(t1 - t0).count();
    return result;
}

}  // namespace schwarz

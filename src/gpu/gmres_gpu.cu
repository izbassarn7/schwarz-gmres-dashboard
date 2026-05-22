#include "gmres_gpu.h"

#include <vector>
#include <cmath>
#include <chrono>

namespace schwarz {
namespace gpu {

GPUGMRESResult gmres_gpu(
    CudaHandles& handles,
    DeviceCSRMatrix& d_A,
    const double* d_b,
    double* d_x,
    int n,
    PrecondApplyFn precond_apply,
    int restart,
    int max_iter,
    double tol)
{
    auto t0 = std::chrono::high_resolution_clock::now();

    int m = restart;
    GPUGMRESResult result;

    DeviceBuffer<double> d_r(n), d_w(n), d_Av(n);
    std::vector<DeviceBuffer<double>> d_V(m + 1);
    for (int i = 0; i <= m; ++i)
        d_V[i].allocate(n);

    std::vector<double> H((m + 1) * m, 0.0);
    std::vector<double> g(m + 1, 0.0);
    std::vector<double> y(m);

    struct GivensRot { double c, s; };
    std::vector<GivensRot> rots(m);

    int total_iter = 0;

    for (int cycle = 0; cycle < max_iter; ++cycle) {
        // r = b - A*x
        cublas_copy(handles.cublas, n, d_b, d_r.ptr);
        d_A.spmv(handles, -1.0, d_x, 1.0, d_r.ptr);

        if (precond_apply)
            precond_apply(d_r.ptr, d_V[0].ptr, n);
        else
            cublas_copy(handles.cublas, n, d_r.ptr, d_V[0].ptr);

        double beta = cublas_nrm2(handles.cublas, n, d_V[0].ptr);
        if (beta < tol) {
            result.converged = true;
            result.residual_norm = beta;
            break;
        }

        cublas_scal(handles.cublas, n, 1.0 / beta, d_V[0].ptr);

        std::fill(H.begin(), H.end(), 0.0);
        std::fill(g.begin(), g.end(), 0.0);
        g[0] = beta;

        int j;
        for (j = 0; j < m; ++j) {
            d_A.spmv(handles, 1.0, d_V[j].ptr, 0.0, d_Av.ptr);

            if (precond_apply)
                precond_apply(d_Av.ptr, d_w.ptr, n);
            else
                cublas_copy(handles.cublas, n, d_Av.ptr, d_w.ptr);

            for (int i = 0; i <= j; ++i) {
                double h = cublas_dot(handles.cublas, n, d_w.ptr, d_V[i].ptr);
                H[i + j * (m + 1)] = h;
                cublas_axpy(handles.cublas, n, -h, d_V[i].ptr, d_w.ptr);
            }

            double h_jp1 = cublas_nrm2(handles.cublas, n, d_w.ptr);
            H[(j + 1) + j * (m + 1)] = h_jp1;

            if (h_jp1 < 1e-30) { j++; break; }

            cublas_scal(handles.cublas, n, 1.0 / h_jp1, d_w.ptr);
            cublas_copy(handles.cublas, n, d_w.ptr, d_V[j + 1].ptr);

            for (int i = 0; i < j; ++i) {
                double t1 = rots[i].c * H[i + j*(m+1)] - rots[i].s * H[(i+1) + j*(m+1)];
                double t2 = rots[i].s * H[i + j*(m+1)] + rots[i].c * H[(i+1) + j*(m+1)];
                H[i + j*(m+1)] = t1;
                H[(i+1) + j*(m+1)] = t2;
            }

            double a = H[j + j*(m+1)];
            double b_val = H[(j+1) + j*(m+1)];
            double r = std::sqrt(a*a + b_val*b_val);
            rots[j].c = a / r;
            rots[j].s = -b_val / r;

            H[j + j*(m+1)] = r;
            H[(j+1) + j*(m+1)] = 0.0;

            double g1 = rots[j].c * g[j] - rots[j].s * g[j+1];
            double g2 = rots[j].s * g[j] + rots[j].c * g[j+1];
            g[j] = g1;
            g[j+1] = g2;

            double res = std::abs(g[j+1]);
            total_iter++;

            if (res < tol) {
                result.converged = true;
                result.residual_norm = res;
                j++;
                break;
            }
        }

        int k = j;
        for (int i = k - 1; i >= 0; --i) {
            y[i] = g[i];
            for (int l = i + 1; l < k; ++l)
                y[i] -= H[i + l*(m+1)] * y[l];
            y[i] /= H[i + i*(m+1)];
        }

        for (int i = 0; i < k; ++i)
            cublas_axpy(handles.cublas, n, y[i], d_V[i].ptr, d_x);

        if (result.converged) break;
        if (total_iter >= max_iter) break;
    }

    result.iterations = total_iter;

    auto t1 = std::chrono::high_resolution_clock::now();
    result.time_solve_s = std::chrono::duration<double>(t1 - t0).count();
    return result;
}

}  // namespace gpu
}  // namespace schwarz

#pragma once

#include "../core/sparse_matrix.h"
#include <vector>

namespace schwarz {

// 5-point stencil for -Laplacian on [0,1]^2 with Dirichlet BC.
// nx, ny: grid points in each dimension. Total DOFs = nx*ny.
inline CSRMatrix make_poisson_2d(int nx, int ny) {
    int n = nx * ny;
    CSRMatrix A;
    A.nrows = n;
    A.ncols = n;
    A.row_ptr.resize(n + 1);

    std::vector<int> ci;
    std::vector<double> cv;
    ci.reserve(5 * n);
    cv.reserve(5 * n);

    for (int iy = 0; iy < ny; ++iy) {
        for (int ix = 0; ix < nx; ++ix) {
            int idx = iy * nx + ix;
            A.row_ptr[idx] = static_cast<int>(ci.size());

            if (iy > 0)      { ci.push_back(idx - nx); cv.push_back(-1.0); }
            if (ix > 0)      { ci.push_back(idx - 1);  cv.push_back(-1.0); }
            ci.push_back(idx); cv.push_back(4.0);
            if (ix < nx - 1) { ci.push_back(idx + 1);  cv.push_back(-1.0); }
            if (iy < ny - 1) { ci.push_back(idx + nx); cv.push_back(-1.0); }
        }
    }
    A.row_ptr[n] = static_cast<int>(ci.size());
    A.col_idx = std::move(ci);
    A.vals = std::move(cv);
    return A;
}

// 7-point stencil for -Laplacian on [0,1]^3 with Dirichlet BC.
inline CSRMatrix make_poisson_3d(int nx, int ny, int nz) {
    int n = nx * ny * nz;
    int nxy = nx * ny;
    CSRMatrix A;
    A.nrows = n;
    A.ncols = n;
    A.row_ptr.resize(n + 1);

    std::vector<int> ci;
    std::vector<double> cv;
    ci.reserve(7 * n);
    cv.reserve(7 * n);

    for (int iz = 0; iz < nz; ++iz) {
        for (int iy = 0; iy < ny; ++iy) {
            for (int ix = 0; ix < nx; ++ix) {
                int idx = iz * nxy + iy * nx + ix;
                A.row_ptr[idx] = static_cast<int>(ci.size());

                if (iz > 0)      { ci.push_back(idx - nxy); cv.push_back(-1.0); }
                if (iy > 0)      { ci.push_back(idx - nx);  cv.push_back(-1.0); }
                if (ix > 0)      { ci.push_back(idx - 1);   cv.push_back(-1.0); }
                ci.push_back(idx); cv.push_back(6.0);
                if (ix < nx - 1) { ci.push_back(idx + 1);   cv.push_back(-1.0); }
                if (iy < ny - 1) { ci.push_back(idx + nx);  cv.push_back(-1.0); }
                if (iz < nz - 1) { ci.push_back(idx + nxy); cv.push_back(-1.0); }
            }
        }
    }
    A.row_ptr[n] = static_cast<int>(ci.size());
    A.col_idx = std::move(ci);
    A.vals = std::move(cv);
    return A;
}

}  // namespace schwarz

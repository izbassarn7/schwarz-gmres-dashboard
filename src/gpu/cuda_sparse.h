#pragma once

#include "cuda_common.h"
#include "../core/sparse_matrix.h"
#include <vector>

namespace schwarz {
namespace gpu {

// GPU-resident CSR matrix with cuSPARSE SpMV
class DeviceCSRMatrix {
public:
    DeviceCSRMatrix() = default;

    void upload(const CSRMatrix& host, CudaHandles& handles) {
        nrows_ = host.nrows;
        ncols_ = host.ncols;
        nnz_ = host.nnz();

        d_row_ptr_.allocate(nrows_ + 1);
        d_col_idx_.allocate(nnz_);
        d_vals_.allocate(nnz_);

        d_row_ptr_.upload(host.row_ptr.data(), nrows_ + 1, handles.stream);
        d_col_idx_.upload(host.col_idx.data(), nnz_, handles.stream);
        d_vals_.upload(host.vals.data(), nnz_, handles.stream);

        CUSPARSE_CHECK(cusparseCreateCsr(
            &descr_, nrows_, ncols_, nnz_,
            d_row_ptr_.ptr, d_col_idx_.ptr, d_vals_.ptr,
            CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
            CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F));

        // Allocate SpMV buffer
        double alpha = 1.0, beta = 0.0;
        cusparseDnVecDescr_t dummy_x, dummy_y;
        DeviceBuffer<double> tmp_x(ncols_), tmp_y(nrows_);
        CUSPARSE_CHECK(cusparseCreateDnVec(&dummy_x, ncols_, tmp_x.ptr, CUDA_R_64F));
        CUSPARSE_CHECK(cusparseCreateDnVec(&dummy_y, nrows_, tmp_y.ptr, CUDA_R_64F));

        CUSPARSE_CHECK(cusparseSpMV_bufferSize(
            handles.cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE,
            &alpha, descr_, dummy_x, &beta, dummy_y,
            CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, &buffer_size_));

        if (buffer_size_ > 0)
            d_buffer_.allocate((buffer_size_ + sizeof(double) - 1) / sizeof(double));

        cusparseDestroyDnVec(dummy_x);
        cusparseDestroyDnVec(dummy_y);
    }

    // y = alpha * A * x + beta * y
    void spmv(CudaHandles& handles,
              double alpha, const double* d_x,
              double beta, double* d_y) const
    {
        cusparseDnVecDescr_t vec_x, vec_y;
        CUSPARSE_CHECK(cusparseCreateDnVec(
            &vec_x, ncols_, const_cast<double*>(d_x), CUDA_R_64F));
        CUSPARSE_CHECK(cusparseCreateDnVec(
            &vec_y, nrows_, d_y, CUDA_R_64F));

        CUSPARSE_CHECK(cusparseSpMV(
            handles.cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE,
            &alpha, descr_, vec_x, &beta, vec_y,
            CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT,
            d_buffer_.ptr));

        cusparseDestroyDnVec(vec_x);
        cusparseDestroyDnVec(vec_y);
    }

    ~DeviceCSRMatrix() {
        if (descr_) cusparseDestroySpMat(descr_);
    }

    int nrows() const { return nrows_; }
    int ncols() const { return ncols_; }

private:
    int nrows_ = 0, ncols_ = 0, nnz_ = 0;
    DeviceBuffer<int> d_row_ptr_, d_col_idx_;
    DeviceBuffer<double> d_vals_;
    cusparseSpMatDescr_t descr_ = nullptr;
    size_t buffer_size_ = 0;
    DeviceBuffer<double> d_buffer_;
};

}  // namespace gpu
}  // namespace schwarz

#pragma once

#include <mpi.h>
#include <cuda_runtime.h>
#include <vector>

namespace schwarz {

// Check if MPI is CUDA-aware at runtime
inline bool is_cuda_aware_mpi() {
#ifdef MPIX_CUDA_AWARE_SUPPORT
    return MPIX_Query_cuda_support() != 0;
#else
    return false;
#endif
}

// Wrapper for MPI send/recv that handles GPU<->CPU staging
// when CUDA-aware MPI is not available
inline void mpi_send_gpu(const double* d_buf, int count, int dest,
                         int tag, MPI_Comm comm, cudaStream_t stream)
{
    if (is_cuda_aware_mpi()) {
        MPI_Send(d_buf, count, MPI_DOUBLE, dest, tag, comm);
    } else {
        std::vector<double> h_buf(count);
        cudaMemcpyAsync(h_buf.data(), d_buf, count * sizeof(double),
                        cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
        MPI_Send(h_buf.data(), count, MPI_DOUBLE, dest, tag, comm);
    }
}

inline void mpi_recv_gpu(double* d_buf, int count, int src,
                         int tag, MPI_Comm comm, cudaStream_t stream)
{
    if (is_cuda_aware_mpi()) {
        MPI_Recv(d_buf, count, MPI_DOUBLE, src, tag, comm, MPI_STATUS_IGNORE);
    } else {
        std::vector<double> h_buf(count);
        MPI_Recv(h_buf.data(), count, MPI_DOUBLE, src, tag, comm, MPI_STATUS_IGNORE);
        cudaMemcpyAsync(d_buf, h_buf.data(), count * sizeof(double),
                        cudaMemcpyHostToDevice, stream);
    }
}

}  // namespace schwarz

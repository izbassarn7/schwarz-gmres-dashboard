#pragma once

#include "sparse_matrix.h"
#include "vector.h"
#include "../comm/halo_exchange.h"

#include <mpi.h>
#include <vector>
#include <map>
#include <set>
#include <algorithm>
#include <numeric>
#include <cassert>

namespace schwarz {

// Row-distributed CSR matrix over MPI ranks.
// Each rank owns a contiguous block of rows [row_start, row_end).
// Ghost columns are appended after local columns.
struct DistributedCSRMatrix {
    MPI_Comm comm = MPI_COMM_NULL;
    int rank = 0;
    int nranks = 0;

    int global_nrows = 0;
    int global_ncols = 0;

    int row_start = 0;     // first global row owned by this rank
    int row_end = 0;       // one past last global row
    int local_nrows = 0;   // row_end - row_start

    // Local CSR (column indices are in local numbering)
    CSRMatrix local;

    // Mapping
    std::vector<int> local_to_global;  // local col -> global col
    std::map<int, int> global_to_local; // global col -> local col

    int n_owned = 0;   // == local_nrows
    int n_ghost = 0;
    int n_local_cols = 0; // n_owned + n_ghost

    // Ghost info
    std::vector<int> ghost_global_ids;
    std::vector<int> ghost_owners;

    HaloExchange halo;

    // Distribute a global CSR matrix across ranks (rank 0 broadcasts).
    // Uses block-row distribution.
    void distribute(const CSRMatrix& A_global, MPI_Comm c) {
        comm = c;
        MPI_Comm_rank(comm, &rank);
        MPI_Comm_size(comm, &nranks);

        // Broadcast dimensions
        int dims[2] = {A_global.nrows, A_global.ncols};
        MPI_Bcast(dims, 2, MPI_INT, 0, comm);
        global_nrows = dims[0];
        global_ncols = dims[1];

        // Block-row partition
        int base = global_nrows / nranks;
        int remainder = global_nrows % nranks;
        row_start = rank * base + std::min(rank, remainder);
        row_end = row_start + base + (rank < remainder ? 1 : 0);
        local_nrows = row_end - row_start;
        n_owned = local_nrows;

        // Extract local rows from global matrix (all ranks need the data,
        // so broadcast the full matrix for simplicity; for huge matrices,
        // use scatter)
        std::vector<int> g_row_ptr, g_col_idx;
        std::vector<double> g_vals;
        int g_nnz = 0;

        if (rank == 0) {
            g_row_ptr = A_global.row_ptr;
            g_col_idx = A_global.col_idx;
            g_vals = A_global.vals;
            g_nnz = A_global.nnz();
        }

        // Broadcast row_ptr
        int rp_size = global_nrows + 1;
        g_row_ptr.resize(rp_size);
        MPI_Bcast(g_row_ptr.data(), rp_size, MPI_INT, 0, comm);
        MPI_Bcast(&g_nnz, 1, MPI_INT, 0, comm);
        g_col_idx.resize(g_nnz);
        g_vals.resize(g_nnz);
        MPI_Bcast(g_col_idx.data(), g_nnz, MPI_INT, 0, comm);
        MPI_Bcast(g_vals.data(), g_nnz, MPI_DOUBLE, 0, comm);

        // Identify ghost columns: columns referenced by local rows
        // that are NOT in [row_start, row_end)
        std::set<int> ghost_set;
        int local_nnz = g_row_ptr[row_end] - g_row_ptr[row_start];
        for (int i = row_start; i < row_end; ++i) {
            for (int j = g_row_ptr[i]; j < g_row_ptr[i + 1]; ++j) {
                int col = g_col_idx[j];
                if (col < row_start || col >= row_end)
                    ghost_set.insert(col);
            }
        }

        // Build local-to-global and global-to-local maps
        local_to_global.clear();
        global_to_local.clear();

        // Owned columns first (identity mapping offset by row_start)
        for (int i = row_start; i < row_end; ++i) {
            int lid = i - row_start;
            local_to_global.push_back(i);
            global_to_local[i] = lid;
        }

        // Ghost columns appended
        ghost_global_ids.assign(ghost_set.begin(), ghost_set.end());
        std::sort(ghost_global_ids.begin(), ghost_global_ids.end());
        n_ghost = static_cast<int>(ghost_global_ids.size());
        n_local_cols = n_owned + n_ghost;

        ghost_owners.resize(n_ghost);
        for (int i = 0; i < n_ghost; ++i) {
            int gid = ghost_global_ids[i];
            int lid = n_owned + i;
            local_to_global.push_back(gid);
            global_to_local[gid] = lid;

            // Determine owner of this global DOF
            for (int r = 0; r < nranks; ++r) {
                int rs = r * base + std::min(r, remainder);
                int re = rs + base + (r < remainder ? 1 : 0);
                if (gid >= rs && gid < re) {
                    ghost_owners[i] = r;
                    break;
                }
            }
        }

        // Build local CSR with local column numbering
        local.nrows = local_nrows;
        local.ncols = n_local_cols;
        local.row_ptr.resize(local_nrows + 1);
        local.col_idx.resize(local_nnz);
        local.vals.resize(local_nnz);

        int nnz_ptr = 0;
        for (int i = 0; i < local_nrows; ++i) {
            local.row_ptr[i] = nnz_ptr;
            int gi = row_start + i;
            for (int j = g_row_ptr[gi]; j < g_row_ptr[gi + 1]; ++j) {
                local.col_idx[nnz_ptr] = global_to_local[g_col_idx[j]];
                local.vals[nnz_ptr] = g_vals[j];
                nnz_ptr++;
            }
        }
        local.row_ptr[local_nrows] = nnz_ptr;

        // Setup halo exchange
        std::vector<int> owned_gids(local_to_global.begin(),
                                     local_to_global.begin() + n_owned);
        halo.setup(comm, owned_gids, ghost_global_ids,
                   ghost_owners, global_to_local);
    }

    // Distributed SpMV: y = A * x
    // x must have n_local_cols entries (owned + ghost), but ghost values
    // are filled by halo exchange. y has local_nrows entries.
    void spmv(double* x, double* y) const {
        // Fill ghost values via halo exchange
        const_cast<HaloExchange&>(halo).exchange(x);

        // Local SpMV
        local.spmv(x, y);
    }

    // Distributed dot product: result = x^T * y (only owned entries)
    double dist_dot(const double* x, const double* y) const {
        double local_sum = 0.0;
        for (int i = 0; i < n_owned; ++i)
            local_sum += x[i] * y[i];
        double global_sum = 0.0;
        MPI_Allreduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, comm);
        return global_sum;
    }

    double dist_norm2(const double* x) const {
        return std::sqrt(dist_dot(x, x));
    }
};

}  // namespace schwarz

#pragma once

#include "../core/sparse_matrix.h"
#include <vector>
#include <set>
#include <algorithm>
#include <numeric>
#include <metis.h>

namespace schwarz {

struct DomainDecomposition {
    int n_subdomains;
    // part[i] = subdomain id for row i
    std::vector<int> partition;
    // owned_dofs[s] = list of DOF indices owned by subdomain s (no overlap)
    std::vector<std::vector<int>> owned_dofs;
    // subdomain_dofs[s] = list of DOF indices including overlap
    std::vector<std::vector<int>> subdomain_dofs;
};

// Graph partition via METIS
inline std::vector<int> partition_metis(const CSRMatrix& A, int nparts) {
    if (nparts <= 1) {
        return std::vector<int>(A.nrows, 0);
    }

    idx_t n = A.nrows;
    idx_t ncon = 1;
    idx_t np = nparts;
    idx_t objval;

    std::vector<idx_t> xadj(A.row_ptr.begin(), A.row_ptr.end());
    std::vector<idx_t> adjncy;
    adjncy.reserve(A.nnz());

    // Build adjacency (exclude self-loops for METIS)
    std::vector<idx_t> xadj2(n + 1, 0);
    for (idx_t i = 0; i < n; ++i) {
        for (idx_t j = A.row_ptr[i]; j < A.row_ptr[i + 1]; ++j) {
            if (A.col_idx[j] != i)
                adjncy.push_back(A.col_idx[j]);
        }
        xadj2[i + 1] = static_cast<idx_t>(adjncy.size());
    }

    std::vector<idx_t> part(n);
    int ret = METIS_PartGraphKway(
        &n, &ncon, xadj2.data(), adjncy.data(),
        nullptr, nullptr, nullptr,
        &np, nullptr, nullptr, nullptr,
        &objval, part.data());

    if (ret != METIS_OK)
        throw std::runtime_error("METIS partitioning failed");

    return std::vector<int>(part.begin(), part.end());
}

// Build overlap layers around each subdomain
inline DomainDecomposition build_decomposition(
    const CSRMatrix& A, int nparts, int overlap)
{
    DomainDecomposition dd;
    dd.n_subdomains = nparts;
    dd.partition = partition_metis(A, nparts);

    dd.owned_dofs.resize(nparts);
    for (int i = 0; i < A.nrows; ++i)
        dd.owned_dofs[dd.partition[i]].push_back(i);

    dd.subdomain_dofs.resize(nparts);
    for (int s = 0; s < nparts; ++s) {
        std::set<int> dofs(dd.owned_dofs[s].begin(), dd.owned_dofs[s].end());

        for (int layer = 0; layer < overlap; ++layer) {
            std::set<int> new_dofs;
            for (int idx : dofs) {
                for (int j = A.row_ptr[idx]; j < A.row_ptr[idx + 1]; ++j) {
                    int col = A.col_idx[j];
                    if (dofs.find(col) == dofs.end())
                        new_dofs.insert(col);
                }
            }
            dofs.insert(new_dofs.begin(), new_dofs.end());
        }

        dd.subdomain_dofs[s].assign(dofs.begin(), dofs.end());
        std::sort(dd.subdomain_dofs[s].begin(), dd.subdomain_dofs[s].end());
    }

    return dd;
}

}  // namespace schwarz

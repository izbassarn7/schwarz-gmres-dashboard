#pragma once

#include "../core/sparse_matrix.h"
#include <vector>
#include <set>
#include <algorithm>

namespace schwarz {

// Expand a set of DOF indices by `layers` overlap using adjacency from A.
inline std::vector<int> expand_overlap(
    const CSRMatrix& A,
    const std::vector<int>& initial_dofs,
    int layers)
{
    std::set<int> dofs(initial_dofs.begin(), initial_dofs.end());

    for (int l = 0; l < layers; ++l) {
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

    std::vector<int> result(dofs.begin(), dofs.end());
    std::sort(result.begin(), result.end());
    return result;
}

}  // namespace schwarz

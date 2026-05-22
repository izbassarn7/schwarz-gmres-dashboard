#pragma once

#include "../core/sparse_matrix.h"
#include <string>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <stdexcept>
#include <vector>
#include <tuple>

namespace schwarz {

inline CSRMatrix read_matrix_market(const std::string& filename) {
    std::ifstream in(filename);
    if (!in.is_open())
        throw std::runtime_error("Cannot open file: " + filename);

    std::string line;
    bool symmetric = false;
    bool pattern_only = false;

    // Read header
    while (std::getline(in, line)) {
        if (line[0] == '%') {
            if (line.find("symmetric") != std::string::npos)
                symmetric = true;
            if (line.find("pattern") != std::string::npos)
                pattern_only = true;
            continue;
        }
        break;
    }

    int nrows, ncols, nnz_file;
    {
        std::istringstream iss(line);
        iss >> nrows >> ncols >> nnz_file;
    }

    struct Triplet {
        int row, col;
        double val;
    };
    std::vector<Triplet> entries;
    entries.reserve(symmetric ? 2 * nnz_file : nnz_file);

    for (int k = 0; k < nnz_file; ++k) {
        if (!std::getline(in, line))
            throw std::runtime_error("Premature end of file");
        std::istringstream iss(line);
        int r, c;
        double v = 1.0;
        iss >> r >> c;
        if (!pattern_only)
            iss >> v;
        --r; --c;  // 1-indexed to 0-indexed

        entries.push_back({r, c, v});
        if (symmetric && r != c)
            entries.push_back({c, r, v});
    }

    std::sort(entries.begin(), entries.end(),
        [](const Triplet& a, const Triplet& b) {
            return a.row < b.row || (a.row == b.row && a.col < b.col);
        });

    CSRMatrix A;
    A.nrows = nrows;
    A.ncols = ncols;
    A.row_ptr.resize(nrows + 1, 0);
    A.col_idx.resize(entries.size());
    A.vals.resize(entries.size());

    for (auto& e : entries)
        A.row_ptr[e.row + 1]++;
    for (int i = 0; i < nrows; ++i)
        A.row_ptr[i + 1] += A.row_ptr[i];

    // Deduplicate (sum duplicates)
    std::vector<int> pos(nrows + 1);
    std::copy(A.row_ptr.begin(), A.row_ptr.end(), pos.begin());

    for (auto& e : entries) {
        int p = pos[e.row]++;
        A.col_idx[p] = e.col;
        A.vals[p] = e.val;
    }

    return A;
}

inline void write_matrix_market(const std::string& filename, const CSRMatrix& A) {
    std::ofstream out(filename);
    out << "%%MatrixMarket matrix coordinate real general\n";
    out << A.nrows << " " << A.ncols << " " << A.nnz() << "\n";
    for (int i = 0; i < A.nrows; ++i) {
        for (int j = A.row_ptr[i]; j < A.row_ptr[i + 1]; ++j) {
            out << (i + 1) << " " << (A.col_idx[j] + 1) << " " << A.vals[j] << "\n";
        }
    }
}

}  // namespace schwarz

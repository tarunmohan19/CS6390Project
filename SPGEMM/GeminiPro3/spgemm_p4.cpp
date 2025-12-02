#include <vector>
#include <map>
#include <algorithm>
#include <utility>
#include <iostream>
#include <mpi.h>
#include <cassert>
#include <functional>
#include "functions.h"

// Typedef for the matrix entry to make code cleaner
using MatrixEntry = std::pair<std::pair<int, int>, int>;

/**
 * Helper to broadcast a list of matrix entries from a root rank to a communicator.
 * * Since MPI cannot natively send std::vector<pair...>, we serialize the data
 * into three integer arrays (rows, cols, values), broadcast sizes, and then data.
 */
void broadcast_matrix_chunk(std::vector<MatrixEntry>& buffer, int root, MPI_Comm comm) {
    int rank;
    MPI_Comm_rank(comm, &rank);

    int count = 0;
    if (rank == root) {
        count = static_cast<int>(buffer.size());
    }

    // 1. Broadcast the number of elements
    MPI_Bcast(&count, 1, MPI_INT, root, comm);

    // 2. Prepare buffers
    std::vector<int> rows;
    std::vector<int> cols;
    std::vector<int> vals;

    if (rank == root) {
        rows.reserve(count);
        cols.reserve(count);
        vals.reserve(count);
        for (const auto& entry : buffer) {
            rows.push_back(entry.first.first);
            cols.push_back(entry.first.second);
            vals.push_back(entry.second);
        }
    } else {
        rows.resize(count);
        cols.resize(count);
        vals.resize(count);
    }

    // 3. Broadcast the actual data
    if (count > 0) {
        MPI_Bcast(rows.data(), count, MPI_INT, root, comm);
        MPI_Bcast(cols.data(), count, MPI_INT, root, comm);
        MPI_Bcast(vals.data(), count, MPI_INT, root, comm);
    }

    // 4. Reconstruct on receivers
    if (rank != root) {
        buffer.clear();
        buffer.reserve(count);
        for (int i = 0; i < count; ++i) {
            buffer.push_back({{rows[i], cols[i]}, vals[i]});
        }
    }
}

void spgemm_2d(int m, int p, int n,
               std::vector<std::pair<std::pair<int,int>, int>> &A,
               std::vector<std::pair<std::pair<int,int>, int>> &B,
               std::vector<std::pair<std::pair<int,int>, int>> &C,
               std::function<int(int, int)> plus, std::function<int(int, int)> times,
               MPI_Comm row_comm, MPI_Comm col_comm)
{
    int row_rank, row_size;
    MPI_Comm_rank(row_comm, &row_rank);
    MPI_Comm_size(row_comm, &row_size);

    int col_rank, col_size;
    MPI_Comm_rank(col_comm, &col_rank);
    MPI_Comm_size(col_comm, &col_size);

    // Grid dimension (assuming q x q grid)
    int q = row_size; 

    // Buffers to hold incoming broadcast data
    std::vector<MatrixEntry> A_recv;
    std::vector<MatrixEntry> B_recv;
    
    // Use a vector for C to maximize insertion speed, sort/reduce later.
    // std::map is simpler but significantly slower for large matrices.
    std::vector<MatrixEntry> C_raw;

    // SUMMA Loop
    for (int k = 0; k < q; ++k) {
        // --- Phase 1: Communication ---
        
        // 1. Broadcast A chunk horizontally
        // The process in grid column 'k' (rank k in row_comm) has the A slice needed.
        if (row_rank == k) {
            A_recv = A; 
        }
        broadcast_matrix_chunk(A_recv, k, row_comm);

        // 2. Broadcast B chunk vertically
        // The process in grid row 'k' (rank k in col_comm) has the B slice needed.
        if (col_rank == k) {
            B_recv = B;
        }
        broadcast_matrix_chunk(B_recv, k, col_comm);

        // --- Phase 2: Local Sparse Multiplication (A_recv * B_recv) ---
        
        if (A_recv.empty() || B_recv.empty()) {
            continue;
        }

        // Optimization: Convert B_recv to a CSR-like lookup for speed.
        // Sort B by row to allow binary search or efficient range finding.
        std::sort(B_recv.begin(), B_recv.end(), [](const MatrixEntry& a, const MatrixEntry& b) {
            if (a.first.first != b.first.first) return a.first.first < b.first.first;
            return a.first.second < b.first.second;
        });

        // Create an index to quickly find the start of each row in B
        // Since coordinate values can be arbitrary large, we cannot use a simple array.
        // We iterate through sorted B to identify ranges.
        
        // Iterate A elements
        for (const auto& a_elem : A_recv) {
            int a_row = a_elem.first.first;
            int a_col = a_elem.first.second; // This must match B's row
            int a_val = a_elem.second;

            // Find range in B where B.row == a_col
            // Since B is sorted by row, we use equal_range logic or custom iteration
            // Lower bound search for the specific row in B
            auto it = std::lower_bound(B_recv.begin(), B_recv.end(), a_col, 
                [](const MatrixEntry& entry, int val) {
                    return entry.first.first < val;
                });

            // Iterate until the row index changes
            while (it != B_recv.end() && it->first.first == a_col) {
                int b_col = it->first.second;
                int b_val = it->second;

                // C[i, j] += A[i, k] * B[k, j]
                int product = times(a_val, b_val);
                
                // We assume global coordinates are passed, so we push them directly.
                // Note: If distribute_matrix_2d localized coordinates, we would need offsets,
                // but standard spgemm signatures usually retain global coords in COO lists 
                // or the helper functions handle it. Assuming standard COO format here.
                C_raw.push_back({{a_row, b_col}, product});
                
                it++;
            }
        }
    }

    // --- Phase 3: Consolidate C ---
    
    if (!C_raw.empty()) {
        // Sort to bring identical coordinates together
        std::sort(C_raw.begin(), C_raw.end());

        // Compress: sum values with same coordinates
        C.reserve(C_raw.size());
        
        // Push first element
        C.push_back(C_raw[0]);

        for (size_t i = 1; i < C_raw.size(); ++i) {
            if (C_raw[i].first == C.back().first) {
                // Same coordinate, accumulate value
                C.back().second = plus(C.back().second, C_raw[i].second);
            } else {
                // New coordinate
                C.push_back(C_raw[i]);
            }
        }
        
        // Filter out zeros if necessary (though not explicitly required by signature, 
        // sparse matrices usually don't store explicit zeros).
        // For this generic implementation, we keep what `plus` returns.
    }
}

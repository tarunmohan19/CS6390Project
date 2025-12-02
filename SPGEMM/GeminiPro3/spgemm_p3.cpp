#include <vector>
#include <map>
#include <algorithm>
#include <utility>
#include <iostream>
#include <mpi.h>
#include <cassert>
#include "functions.h"

// Helper to broadcast vectors of triplets (sparse matrix data)
void bcast_sparse_data(std::vector<int>& rows, std::vector<int>& cols, 
                       std::vector<int>& vals, int root, MPI_Comm comm) {
    int count = 0;
    int rank;
    MPI_Comm_rank(comm, &rank);

    // 1. Broadcast the number of non-zero elements
    if (rank == root) {
        count = rows.size();
    }
    MPI_Bcast(&count, 1, MPI_INT, root, comm);

    // 2. Resize receivers
    if (rank != root) {
        rows.resize(count);
        cols.resize(count);
        vals.resize(count);
    }

    // 3. Broadcast the actual data arrays
    // Note: If count is 0, we can skip or Bcast 0 bytes, but standard Bcast is safe.
    if (count > 0) {
        MPI_Bcast(rows.data(), count, MPI_INT, root, comm);
        MPI_Bcast(cols.data(), count, MPI_INT, root, comm);
        MPI_Bcast(vals.data(), count, MPI_INT, root, comm);
    }
}

void spgemm_2d(int m, int p, int n,
               std::vector<std::pair<std::pair<int,int>, int>> &A,
               std::vector<std::pair<std::pair<int,int>, int>> &B,
               std::vector<std::pair<std::pair<int,int>, int>> &C,
               std::function<int(int, int)> plus, std::function<int(int, int)> times,
               MPI_Comm row_comm, MPI_Comm col_comm)
{
    int rank_row, size_row;
    int rank_col, size_col;

    // Get grid information
    MPI_Comm_rank(row_comm, &rank_row); // My column index
    MPI_Comm_size(row_comm, &size_row); // Total columns (q)
    
    MPI_Comm_rank(col_comm, &rank_col); // My row index
    MPI_Comm_size(col_comm, &size_col); // Total rows (q)

    // Ensure the grid is square as expected by SUMMA logic here
    int q = size_row; 

    // Use a map for C to accumulate values easily during the K-steps.
    // Key: {row, col}, Value: accumulated result
    std::map<std::pair<int, int>, int> C_accum;

    // Buffers for broadcasting
    std::vector<int> buf_A_r, buf_A_c, buf_A_v;
    std::vector<int> buf_B_r, buf_B_c, buf_B_v;

    // --- The SUMMA Loop ---
    for (int k = 0; k < q; ++k) {
        
        // --- 1. Broadcast A chunk across the Row ---
        // The process with column index 'k' sends its A to its row.
        buf_A_r.clear(); buf_A_c.clear(); buf_A_v.clear();
        
        if (rank_row == k) {
            // I am the root for this row broadcast (I hold the slice of A)
            for (const auto& entry : A) {
                buf_A_r.push_back(entry.first.first);
                buf_A_c.push_back(entry.first.second);
                buf_A_v.push_back(entry.second);
            }
        }
        bcast_sparse_data(buf_A_r, buf_A_c, buf_A_v, k, row_comm);

        // --- 2. Broadcast B chunk across the Column ---
        // The process with row index 'k' sends its B to its column.
        buf_B_r.clear(); buf_B_c.clear(); buf_B_v.clear();

        if (rank_col == k) {
            // I am the root for this column broadcast (I hold the slice of B)
            for (const auto& entry : B) {
                buf_B_r.push_back(entry.first.first);
                buf_B_c.push_back(entry.first.second);
                buf_B_v.push_back(entry.second);
            }
        }
        bcast_sparse_data(buf_B_r, buf_B_c, buf_B_v, k, col_comm);

        // --- 3. Local Sparse Multiplication ---
        // We need to compute: C_accum += buf_A * buf_B
        // To do this efficiently, we convert the received B-chunk into a lookup table.
        // Map: Row Index of B -> List of {Column Index of B, Value}
        std::map<int, std::vector<std::pair<int, int>>> B_adj;
        for (size_t i = 0; i < buf_B_r.size(); ++i) {
            B_adj[buf_B_r[i]].push_back({buf_B_c[i], buf_B_v[i]});
        }

        // Iterate through the received A-chunk
        for (size_t i = 0; i < buf_A_r.size(); ++i) {
            int rA = buf_A_r[i];
            int cA = buf_A_c[i]; // This corresponds to the row index in B
            int vA = buf_A_v[i];

            // If there are corresponding rows in B (i.e., A.col == B.row)
            if (B_adj.find(cA) != B_adj.end()) {
                for (const auto& b_entry : B_adj[cA]) {
                    int cB = b_entry.first;
                    int vB = b_entry.second;

                    int product = times(vA, vB);

                    // Accumulate into C
                    std::pair<int, int> coords = {rA, cB};
                    if (C_accum.find(coords) == C_accum.end()) {
                        C_accum[coords] = product;
                    } else {
                        C_accum[coords] = plus(C_accum[coords], product);
                    }
                }
            }
        }
    }

    // --- 4. Finalize Output ---
    // Flatten the map back into the output vector C
    C.clear();
    for (const auto& entry : C_accum) {
        C.push_back({entry.first, entry.second});
    }
}

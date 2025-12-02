#include <vector>
#include <map>
#include <algorithm>
#include <utility>
#include <iostream>
#include <mpi.h>
#include <cassert>
#include "functions.h"

// Helper to serialize sparse matrix to a flat int vector for MPI transmission
// Layout: [row, col, val, row, col, val, ...]
void serialize_matrix(const std::vector<std::pair<std::pair<int, int>, int>>& matrix, 
                      std::vector<int>& buffer) {
    buffer.reserve(matrix.size() * 3);
    for (const auto& entry : matrix) {
        buffer.push_back(entry.first.first);  // row
        buffer.push_back(entry.first.second); // col
        buffer.push_back(entry.second);       // value
    }
}

void spgemm_2d(int m, int p, int n,
               std::vector<std::pair<std::pair<int,int>, int>> &A,
               std::vector<std::pair<std::pair<int,int>, int>> &B,
               std::vector<std::pair<std::pair<int,int>, int>> &C,
               std::function<int(int, int)> plus, std::function<int(int, int)> times,
               MPI_Comm row_comm, MPI_Comm col_comm)
{
    // 1. Determine Grid Topology
    int my_row_idx, my_col_idx;
    int grid_dim; // Assuming square grid based on problem description (q*q)

    // row_comm contains all processes in the same row. 
    // The rank within row_comm is effectively the column coordinate.
    MPI_Comm_rank(row_comm, &my_col_idx);
    MPI_Comm_size(row_comm, &grid_dim);

    // col_comm contains all processes in the same column.
    // The rank within col_comm is effectively the row coordinate.
    MPI_Comm_rank(col_comm, &my_row_idx);

    // 2. Accumulator for C
    // We use a map to handle sparse accumulation (merging duplicates) efficiently during the steps.
    // Key: <row, col>, Value: accumulated value
    std::map<std::pair<int, int>, int> C_map;

    // 3. SUMMA Loop
    // Iterate k from 0 to grid_dim - 1
    for (int k = 0; k < grid_dim; ++k) {
        
        // --- Broadcast A Horizontal (along row_comm) ---
        std::vector<int> buf_A;
        
        // If I am in column k, I own the slice of A required for this step
        if (my_col_idx == k) {
            serialize_matrix(A, buf_A);
        }

        // Broadcast size of A data
        int size_A = (my_col_idx == k) ? static_cast<int>(buf_A.size()) : 0;
        MPI_Bcast(&size_A, 1, MPI_INT, k, row_comm);

        // Resize buffer and broadcast data
        buf_A.resize(size_A);
        MPI_Bcast(buf_A.data(), size_A, MPI_INT, k, row_comm);

        // --- Broadcast B Vertical (along col_comm) ---
        std::vector<int> buf_B;

        // If I am in row k, I own the slice of B required for this step
        if (my_row_idx == k) {
            serialize_matrix(B, buf_B);
        }

        // Broadcast size of B data
        int size_B = (my_row_idx == k) ? static_cast<int>(buf_B.size()) : 0;
        MPI_Bcast(&size_B, 1, MPI_INT, k, col_comm);

        // Resize buffer and broadcast data
        buf_B.resize(size_B);
        MPI_Bcast(buf_B.data(), size_B, MPI_INT, k, col_comm);

        // --- Local Computation (A_recv * B_recv) ---
        // Optimization: Create a lookup structure for B to avoid O(N^2) search.
        // Map: Row Index -> Vector of Pairs (Column Index, Value)
        std::map<int, std::vector<std::pair<int, int>>> B_lookup;
        for (size_t i = 0; i < buf_B.size(); i += 3) {
            int r = buf_B[i];
            int c = buf_B[i+1];
            int v = buf_B[i+2];
            B_lookup[r].push_back({c, v});
        }

        // Iterate through A elements
        for (size_t i = 0; i < buf_A.size(); i += 3) {
            int r_A = buf_A[i];
            int c_A = buf_A[i+1]; // This column index matches B's row index
            int v_A = buf_A[i+2];

            // If there are entries in B corresponding to this column of A
            if (B_lookup.count(c_A)) {
                const auto& b_row_entries = B_lookup[c_A];
                for (const auto& b_entry : b_row_entries) {
                    int c_B = b_entry.first;
                    int v_B = b_entry.second;

                    // Calculate partial product
                    int product = times(v_A, v_B);
                    
                    // Accumulate into C
                    // Resulting coordinate: (Row of A, Column of B)
                    std::pair<int, int> coord = {r_A, c_B};
                    
                    if (C_map.count(coord)) {
                        C_map[coord] = plus(C_map[coord], product);
                    } else {
                        C_map[coord] = product;
                    }
                }
            }
        }
    }

    // 4. Finalize C
    // Convert the map back to the vector format expected by the caller
    C.clear();
    C.reserve(C_map.size());
    for (const auto& kv : C_map) {
        C.push_back({kv.first, kv.second});
    }
}

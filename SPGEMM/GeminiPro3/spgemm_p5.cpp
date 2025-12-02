#include <vector>
#include <map>
#include <unordered_map>
#include <algorithm>
#include <utility>
#include <iostream>
#include <mpi.h>
#include <cassert>
#include <functional>

// Helper to flatten sparse matrix data for MPI communication
// Layout: [i1, j1, v1, i2, j2, v2, ...]
void pack_matrix(const std::vector<std::pair<std::pair<int,int>, int>> &source, 
                 std::vector<int> &dest) {
    dest.clear();
    dest.reserve(source.size() * 3);
    for (const auto &entry : source) {
        dest.push_back(entry.first.first);  // row
        dest.push_back(entry.first.second); // col
        dest.push_back(entry.second);       // value
    }
}

// Helper to unpack MPI buffer back to sparse format
void unpack_matrix(const std::vector<int> &source, 
                   std::vector<std::pair<std::pair<int,int>, int>> &dest) {
    dest.clear();
    size_t num_entries = source.size() / 3;
    dest.reserve(num_entries);
    for (size_t k = 0; k < num_entries; ++k) {
        int r = source[3*k];
        int c = source[3*k+1];
        int v = source[3*k+2];
        dest.push_back({{r, c}, v});
    }
}

void spgemm_2d(int m, int p, int n,
               std::vector<std::pair<std::pair<int,int>, int>> &A,
               std::vector<std::pair<std::pair<int,int>, int>> &B,
               std::vector<std::pair<std::pair<int,int>, int>> &C,
               std::function<int(int, int)> plus, std::function<int(int, int)> times,
               MPI_Comm row_comm, MPI_Comm col_comm)
{
    int my_row, my_col, grid_dim_q;
    
    // Determine grid coordinates and dimensions
    // row_comm contains ranks in the same row, so rank in row_comm is the column index
    MPI_Comm_rank(row_comm, &my_col); 
    MPI_Comm_size(row_comm, &grid_dim_q);
    // col_comm contains ranks in the same column, so rank in col_comm is the row index
    MPI_Comm_rank(col_comm, &my_row);

    // Accumulator for C: using a map to handle sparse updates and additions
    // Key: <row, col>, Value: accumulated result
    std::map<std::pair<int, int>, int> C_accum;

    // Buffers for communication
    std::vector<int> send_buffer;
    std::vector<int> recv_buffer;
    
    // Temporary containers for the matrix blocks current being processed
    std::vector<std::pair<std::pair<int,int>, int>> current_A;
    std::vector<std::pair<std::pair<int,int>, int>> current_B;

    // SUMMA Loop: Iterate through the common dimension blocks (k from 0 to q-1)
    for (int k = 0; k < grid_dim_q; ++k) {
        
        // ---------------------------------------------------------
        // 1. Broadcast A-block along the Row
        // ---------------------------------------------------------
        int root_col = k; // The process at column 'k' owns the A-block for this step
        int msg_size = 0;

        if (my_col == root_col) {
            pack_matrix(A, send_buffer);
            msg_size = static_cast<int>(send_buffer.size());
        }

        // Broadcast size of data first
        MPI_Bcast(&msg_size, 1, MPI_INT, root_col, row_comm);

        // Prepare receive buffer
        recv_buffer.resize(msg_size);
        if (my_col == root_col) {
            std::copy(send_buffer.begin(), send_buffer.end(), recv_buffer.begin());
        }

        // Broadcast actual data
        MPI_Bcast(recv_buffer.data(), msg_size, MPI_INT, root_col, row_comm);
        unpack_matrix(recv_buffer, current_A);

        // ---------------------------------------------------------
        // 2. Broadcast B-block along the Column
        // ---------------------------------------------------------
        int root_row = k; // The process at row 'k' owns the B-block for this step
        msg_size = 0;
        
        if (my_row == root_row) {
            pack_matrix(B, send_buffer);
            msg_size = static_cast<int>(send_buffer.size());
        }

        MPI_Bcast(&msg_size, 1, MPI_INT, root_row, col_comm);

        recv_buffer.resize(msg_size);
        if (my_row == root_row) {
            std::copy(send_buffer.begin(), send_buffer.end(), recv_buffer.begin());
        }

        MPI_Bcast(recv_buffer.data(), msg_size, MPI_INT, root_row, col_comm);
        unpack_matrix(recv_buffer, current_B);

        // ---------------------------------------------------------
        // 3. Local Sparse Multiplication (current_A x current_B)
        // ---------------------------------------------------------
        if (current_A.empty() || current_B.empty()) {
            continue; // Optimization: skip if either block is empty
        }

        // Optimization: Create an Adjacency List (Index) for B to allow fast row lookups.
        // We map: B_row_index -> list of {B_col_index, B_value}
        // Using unordered_map for O(1) average access, assuming typical sparsity.
        std::unordered_map<int, std::vector<std::pair<int, int>>> B_adj;
        for (const auto &entry : current_B) {
            int r = entry.first.first;
            int c = entry.first.second;
            int val = entry.second;
            B_adj[r].push_back({c, val});
        }

        // Iterate through A. For A(i, k), find all B(k, j)
        for (const auto &entry_A : current_A) {
            int row_A = entry_A.first.first;
            int col_A = entry_A.first.second; // This acts as the 'k' index
            int val_A = entry_A.second;

            // Look up the corresponding row in B (which is col_A)
            auto it = B_adj.find(col_A);
            if (it != B_adj.end()) {
                // Iterate through the row of B
                for (const auto &entry_B : it->second) {
                    int col_B = entry_B.first;
                    int val_B = entry_B.second;

                    // Compute C(i, j) += A(i, k) * B(k, j)
                    int product = times(val_A, val_B);
                    std::pair<int, int> c_coord = {row_A, col_B};

                    // Accumulate using the 'plus' functor
                    if (C_accum.find(c_coord) == C_accum.end()) {
                        C_accum[c_coord] = product;
                    } else {
                        C_accum[c_coord] = plus(C_accum[c_coord], product);
                    }
                }
            }
        }
    }

    // ---------------------------------------------------------
    // 4. Final Output Generation
    // ---------------------------------------------------------
    C.clear();
    C.reserve(C_accum.size());
    for (const auto &kv : C_accum) {
        C.push_back({kv.first, kv.second});
    }
    // The main function typically expects sorted output for correctness checks
    std::sort(C.begin(), C.end());
}

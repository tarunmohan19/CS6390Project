#include <vector>
#include <map>
#include <algorithm>
#include <utility>
#include <iostream>
#include <mpi.h>
#include <cassert>
#include "functions.h"

// Helper function to broadcast a sparse matrix buffer within a communicator.
// It serializes the vector of pairs into contiguous arrays for MPI transport.
void bcast_sparse_matrix(std::vector<std::pair<std::pair<int, int>, int>> &buffer,
                         int root, MPI_Comm comm)
{
    int rank;
    MPI_Comm_rank(comm, &rank);

    int count = 0;
    if (rank == root) {
        count = static_cast<int>(buffer.size());
    }

    // 1. Broadcast number of elements
    MPI_Bcast(&count, 1, MPI_INT, root, comm);

    // Optimize: if no data, clear and return
    if (count == 0) {
        if (rank != root) buffer.clear();
        return;
    }

    std::vector<int> idx1;
    std::vector<int> idx2;
    std::vector<int> vals;

    // 2. Root serializes data
    if (rank == root) {
        idx1.resize(count);
        idx2.resize(count);
        vals.resize(count);
        for (int i = 0; i < count; ++i) {
            idx1[i] = buffer[i].first.first;
            idx2[i] = buffer[i].first.second;
            vals[i] = buffer[i].second;
        }
    } else {
        idx1.resize(count);
        idx2.resize(count);
        vals.resize(count);
    }

    // 3. Broadcast components
    MPI_Bcast(idx1.data(), count, MPI_INT, root, comm);
    MPI_Bcast(idx2.data(), count, MPI_INT, root, comm);
    MPI_Bcast(vals.data(), count, MPI_INT, root, comm);

    // 4. Non-roots reconstruct the buffer
    if (rank != root) {
        buffer.clear();
        buffer.reserve(count);
        for (int i = 0; i < count; ++i) {
            buffer.push_back({{idx1[i], idx2[i]}, vals[i]});
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
    int col_rank, col_size;

    MPI_Comm_rank(row_comm, &row_rank); // My rank within my row (equivalent to grid column index)
    MPI_Comm_size(row_comm, &row_size); // q (grid width)
    
    MPI_Comm_rank(col_comm, &col_rank); // My rank within my col (equivalent to grid row index)
    MPI_Comm_size(col_comm, &col_size); // q (grid height)

    // Ensure we are on a square grid as expected by standard SUMMA
    assert(row_size == col_size);
    int q = row_size;

    // Accumulator for C. 
    // Uses a map to handle sparse accumulation: C(i,j) += A(i,k) * B(k,j)
    // Key: <row, col>, Value: accumulated value
    std::map<std::pair<int, int>, int> C_map;

    // SUMMA Loop
    for (int k = 0; k < q; ++k) {
        // --- Step 1: Broadcast A block ---
        // The process at column 'k' in the grid has the required block of A for this iteration.
        // It broadcasts to everyone else in its row.
        std::vector<std::pair<std::pair<int,int>, int>> A_recv;
        
        if (row_rank == k) {
            A_recv = A; // Copy local data to send buffer
        }
        bcast_sparse_matrix(A_recv, k, row_comm);

        // --- Step 2: Broadcast B block ---
        // The process at row 'k' in the grid has the required block of B for this iteration.
        // It broadcasts to everyone else in its column.
        std::vector<std::pair<std::pair<int,int>, int>> B_recv;
        
        if (col_rank == k) {
            B_recv = B; // Copy local data to send buffer
        }
        bcast_sparse_matrix(B_recv, k, col_comm);

        // --- Step 3: Local Sparse Matrix Multiply ---
        // A_recv * B_recv -> Accumulate into C_map
        if (A_recv.empty() || B_recv.empty()) continue;

        // Optimization: Create an adjacency list (row lookup) for B_recv 
        // to speed up matching inner dimensions.
        // B_adj: Map<Row Index, List of {Col Index, Value}>
        std::map<int, std::vector<std::pair<int, int>>> B_adj;
        for (const auto &entry : B_recv) {
            int r = entry.first.first;
            int c = entry.first.second;
            int v = entry.second;
            B_adj[r].push_back({c, v});
        }

        // Iterate through A_recv
        for (const auto &entryA : A_recv) {
            int rA = entryA.first.first;
            int cA = entryA.first.second; // Inner dimension (k)
            int vA = entryA.second;

            // Find rows in B that match cA
            auto it = B_adj.find(cA);
            if (it != B_adj.end()) {
                // Multiply A entry with all matching B entries
                for (const auto &entryB : it->second) {
                    int cB = entryB.first;
                    int vB = entryB.second;

                    // Calculate product
                    int product = times(vA, vB);

                    // Accumulate
                    std::pair<int, int> coords = {rA, cB};
                    if (C_map.find(coords) != C_map.end()) {
                        C_map[coords] = plus(C_map[coords], product);
                    } else {
                        C_map[coords] = product;
                    }
                }
            }
        }
    }

    // --- Step 4: Convert accumulator to output vector ---
    C.clear();
    C.reserve(C_map.size());
    for (const auto &kv : C_map) {
        // Only store non-zero values (assuming 0 is identity/empty for sparse)
        // Note: Depending on the 'plus' op, result might be 0, but usually we keep explicitly computed values
        // or check if (kv.second != 0). We will push all entries found.
        C.push_back({{kv.first.first, kv.first.second}, kv.second});
    }

    // Sort to maintain canonical sparse format order
    std::sort(C.begin(), C.end());
}

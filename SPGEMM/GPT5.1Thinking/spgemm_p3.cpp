#include <vector>
#include <map>
#include <algorithm>
#include <utility>
#include <iostream>
#include <mpi.h>
#include <cassert>
#include <cmath>
#include <functional>

void spgemm_2d(int m, int p, int n,
               std::vector<std::pair<std::pair<int,int>, int>> &A,
               std::vector<std::pair<std::pair<int,int>, int>> &B,
               std::vector<std::pair<std::pair<int,int>, int>> &C,
               std::function<int(int, int)> plus, std::function<int(int, int)> times,
               MPI_Comm row_comm, MPI_Comm col_comm)
{
    int world_rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    // Process grid is q x q, with q^2 = world_size
    int q = static_cast<int>(std::sqrt(static_cast<double>(world_size)) + 1e-9);
    assert(q * q == world_size);

    // Helper: redistribute a matrix into a canonical 2D block distribution
    // rows: #rows of matrix, cols: #cols of matrix
    auto redistribute_matrix = [&](int rows, int cols,
                                   std::vector<std::pair<std::pair<int,int>, int>> &M)
    {
        const int P = world_size;
        std::vector<int> send_counts(P, 0);

        // First pass: count how many entries go to each destination rank
        for (auto &entry : M) {
            int i = entry.first.first;
            int j = entry.first.second;

            int dest_pr = (i * q) / rows;  // row block
            if (dest_pr >= q) dest_pr = q - 1;
            int dest_pc = (j * q) / cols;  // col block
            if (dest_pc >= q) dest_pc = q - 1;

            int dest_rank = dest_pr * q + dest_pc;
            ++send_counts[dest_rank];
        }

        std::vector<int> send_displs(P, 0);
        for (int r = 1; r < P; ++r) {
            send_displs[r] = send_displs[r-1] + send_counts[r-1];
        }
        int total_send = send_displs[P-1] + send_counts[P-1];

        // Pack indices and values into three separate arrays
        std::vector<int> send_i(total_send);
        std::vector<int> send_j(total_send);
        std::vector<int> send_val(total_send);

        std::vector<int> offset = send_displs; // working copy
        for (auto &entry : M) {
            int i = entry.first.first;
            int j = entry.first.second;
            int v = entry.second;

            int dest_pr = (i * q) / rows;
            if (dest_pr >= q) dest_pr = q - 1;
            int dest_pc = (j * q) / cols;
            if (dest_pc >= q) dest_pc = q - 1;
            int dest_rank = dest_pr * q + dest_pc;

            int pos = offset[dest_rank]++;
            send_i[pos]   = i;
            send_j[pos]   = j;
            send_val[pos] = v;
        }

        // Figure out how many entries we receive from each rank
        std::vector<int> recv_counts(P, 0);
        MPI_Alltoall(send_counts.data(), 1, MPI_INT,
                     recv_counts.data(), 1, MPI_INT, MPI_COMM_WORLD);

        std::vector<int> recv_displs(P, 0);
        for (int r = 1; r < P; ++r) {
            recv_displs[r] = recv_displs[r-1] + recv_counts[r-1];
        }
        int total_recv = recv_displs[P-1] + recv_counts[P-1];

        std::vector<int> recv_i(total_recv);
        std::vector<int> recv_j(total_recv);
        std::vector<int> recv_val(total_recv);

        // Perform redistribution: three Alltoallv calls
        MPI_Alltoallv(send_i.data(),   send_counts.data(), send_displs.data(), MPI_INT,
                      recv_i.data(),   recv_counts.data(), recv_displs.data(), MPI_INT,
                      MPI_COMM_WORLD);
        MPI_Alltoallv(send_j.data(),   send_counts.data(), send_displs.data(), MPI_INT,
                      recv_j.data(),   recv_counts.data(), recv_displs.data(), MPI_INT,
                      MPI_COMM_WORLD);
        MPI_Alltoallv(send_val.data(), send_counts.data(), send_displs.data(), MPI_INT,
                      recv_val.data(), recv_counts.data(), recv_displs.data(), MPI_INT,
                      MPI_COMM_WORLD);

        // Rebuild local matrix from received data
        std::vector<std::pair<std::pair<int,int>, int>> M_new;
        M_new.reserve(total_recv);
        for (int idx = 0; idx < total_recv; ++idx) {
            M_new.push_back({{recv_i[idx], recv_j[idx]}, recv_val[idx]});
        }
        M.swap(M_new);
    };

    // Redistribute A (m x p) and B (p x n) into canonical 2D block distribution
    redistribute_matrix(m, p, A);
    redistribute_matrix(p, n, B);

    // Get row/col communicator info (these correspond to 2D process rows/cols)
    int row_rank, row_size;
    MPI_Comm_rank(row_comm, &row_rank);
    MPI_Comm_size(row_comm, &row_size);
    int col_rank, col_size;
    MPI_Comm_rank(col_comm, &col_rank);
    MPI_Comm_size(col_comm, &col_size);
    assert(row_size == q && col_size == q);

    // Local C block accumulator (sparse: only store nonzeros)
    std::map<std::pair<int,int>, int> C_map;

    // SUMMA-style algorithm over q blocks in the k dimension
    for (int b = 0; b < q; ++b) {
        // --- A broadcast along process rows (row_comm) ---
        // Root for this k-block in each row is the process with row_rank == b
        int nnzA_block = (row_rank == b) ? static_cast<int>(A.size()) : 0;
        MPI_Bcast(&nnzA_block, 1, MPI_INT, b, row_comm);

        std::vector<int> A_i, A_k, A_val;
        if (nnzA_block > 0) {
            A_i.resize(nnzA_block);
            A_k.resize(nnzA_block);
            A_val.resize(nnzA_block);
        }

        if (row_rank == b && nnzA_block > 0) {
            // Pack local A block (this process owns the (row_block, k_block=b) submatrix)
            for (int idx = 0; idx < nnzA_block; ++idx) {
                A_i[idx]   = A[idx].first.first;   // row index i
                A_k[idx]   = A[idx].first.second;  // k index
                A_val[idx] = A[idx].second;        // value
            }
        }

        if (nnzA_block > 0) {
            MPI_Bcast(A_i.data(),   nnzA_block, MPI_INT, b, row_comm);
            MPI_Bcast(A_k.data(),   nnzA_block, MPI_INT, b, row_comm);
            MPI_Bcast(A_val.data(), nnzA_block, MPI_INT, b, row_comm);
        }

        // --- B broadcast along process columns (col_comm) ---
        // Root for this k-block in each column is the process with col_rank == b
        int nnzB_block = (col_rank == b) ? static_cast<int>(B.size()) : 0;
        MPI_Bcast(&nnzB_block, 1, MPI_INT, b, col_comm);

        std::vector<int> B_k, B_j, B_val;
        if (nnzB_block > 0) {
            B_k.resize(nnzB_block);
            B_j.resize(nnzB_block);
            B_val.resize(nnzB_block);
        }

        if (col_rank == b && nnzB_block > 0) {
            // Pack local B block (this process owns the (k_block=b, col_block) submatrix)
            for (int idx = 0; idx < nnzB_block; ++idx) {
                B_k[idx]   = B[idx].first.first;   // k index
                B_j[idx]   = B[idx].first.second;  // column index j
                B_val[idx] = B[idx].second;        // value
            }
        }

        if (nnzB_block > 0) {
            MPI_Bcast(B_k.data(),   nnzB_block, MPI_INT, b, col_comm);
            MPI_Bcast(B_j.data(),   nnzB_block, MPI_INT, b, col_comm);
            MPI_Bcast(B_val.data(), nnzB_block, MPI_INT, b, col_comm);
        }

        // If either side is empty for this k-block, there is no contribution
        if (nnzA_block == 0 || nnzB_block == 0) {
            continue;
        }

        // --- Local sparse multiplication for this k-block ---
        // Group A and B by their k index to efficiently form outer products
        std::map<int, std::vector<std::pair<int,int>>> A_by_k;
        for (int idx = 0; idx < nnzA_block; ++idx) {
            int k = A_k[idx];
            int i = A_i[idx];
            int v = A_val[idx];
            A_by_k[k].push_back({i, v}); // (row i, value)
        }

        std::map<int, std::vector<std::pair<int,int>>> B_by_k;
        for (int idx = 0; idx < nnzB_block; ++idx) {
            int k = B_k[idx];
            int j = B_j[idx];
            int v = B_val[idx];
            B_by_k[k].push_back({j, v}); // (col j, value)
        }

        // Walk both maps in order to find common k's
        auto itA = A_by_k.begin();
        auto itB = B_by_k.begin();
        while (itA != A_by_k.end() && itB != B_by_k.end()) {
            int kA = itA->first;
            int kB = itB->first;
            if (kA < kB) {
                ++itA;
            } else if (kA > kB) {
                ++itB;
            } else {
                // kA == kB: we have A(i,k) and B(k,j) entries for this k
                const auto &avec = itA->second;
                const auto &bvec = itB->second;
                for (const auto &ai : avec) {
                    int i    = ai.first;
                    int aval = ai.second;
                    for (const auto &bj : bvec) {
                        int j    = bj.first;
                        int bval = bj.second;
                        int prod = times(aval, bval); // times(a_ik, b_kj)

                        auto key = std::make_pair(i, j);
                        auto itC = C_map.find(key);
                        if (itC == C_map.end()) {
                            C_map[key] = prod;
                        } else {
                            itC->second = plus(itC->second, prod);
                        }
                    }
                }
                ++itA;
                ++itB;
            }
        }
    }

    // Convert map -> sparse (i,j,value) list for this rank's C block
    C.clear();
    C.reserve(C_map.size());
    for (const auto &kv : C_map) {
        C.push_back({kv.first, kv.second});
    }
}

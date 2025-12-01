#include <vector>
#include <map>
#include <algorithm>
#include <utility>
#include <iostream>
#include <mpi.h>
#include <cassert>
#include "functions.h"

void spgemm_2d(int m, int p, int n,
               std::vector<std::pair<std::pair<int,int>, int>> &A,
               std::vector<std::pair<std::pair<int,int>, int>> &B,
               std::vector<std::pair<std::pair<int,int>, int>> &C,
               std::function<int(int, int)> plus, std::function<int(int, int)> times,
               MPI_Comm row_comm, MPI_Comm col_comm)
{
    // We will overwrite C completely
    C.clear();

    // --- 1. Allgather A along each column communicator (col_comm) ---
    int col_size = 0;
    MPI_Comm_size(col_comm, &col_size);

    int local_nnz_A = static_cast<int>(A.size());

    // Gather the per-rank counts of A in this column
    std::vector<int> counts_A(col_size);
    MPI_Allgather(&local_nnz_A, 1, MPI_INT,
                  counts_A.data(), 1, MPI_INT, col_comm);

    std::vector<int> displs_A(col_size);
    displs_A[0] = 0;
    for (int i = 1; i < col_size; ++i) {
        displs_A[i] = displs_A[i - 1] + counts_A[i - 1];
    }
    int total_nnz_A = (col_size > 0) ? (displs_A[col_size - 1] + counts_A[col_size - 1]) : 0;

    // Flatten local A into separate arrays for row, col, and value
    std::vector<int> send_A_i(local_nnz_A), send_A_j(local_nnz_A), send_A_v(local_nnz_A);
    for (int idx = 0; idx < local_nnz_A; ++idx) {
        send_A_i[idx] = A[idx].first.first;   // row i
        send_A_j[idx] = A[idx].first.second;  // col k
        send_A_v[idx] = A[idx].second;        // value
    }

    std::vector<int> all_A_i(total_nnz_A), all_A_j(total_nnz_A), all_A_v(total_nnz_A);

    int *recv_A_i_ptr = total_nnz_A ? all_A_i.data() : nullptr;
    int *recv_A_j_ptr = total_nnz_A ? all_A_j.data() : nullptr;
    int *recv_A_v_ptr = total_nnz_A ? all_A_v.data() : nullptr;
    const int *send_A_i_ptr = local_nnz_A ? send_A_i.data() : nullptr;
    const int *send_A_j_ptr = local_nnz_A ? send_A_j.data() : nullptr;
    const int *send_A_v_ptr = local_nnz_A ? send_A_v.data() : nullptr;

    MPI_Allgatherv(send_A_i_ptr, local_nnz_A, MPI_INT,
                   recv_A_i_ptr, counts_A.data(), displs_A.data(), MPI_INT,
                   col_comm);
    MPI_Allgatherv(send_A_j_ptr, local_nnz_A, MPI_INT,
                   recv_A_j_ptr, counts_A.data(), displs_A.data(), MPI_INT,
                   col_comm);
    MPI_Allgatherv(send_A_v_ptr, local_nnz_A, MPI_INT,
                   recv_A_v_ptr, counts_A.data(), displs_A.data(), MPI_INT,
                   col_comm);

    // --- 2. Allgather B along each row communicator (row_comm) ---
    int row_size = 0;
    MPI_Comm_size(row_comm, &row_size);

    int local_nnz_B = static_cast<int>(B.size());

    std::vector<int> counts_B(row_size);
    MPI_Allgather(&local_nnz_B, 1, MPI_INT,
                  counts_B.data(), 1, MPI_INT, row_comm);

    std::vector<int> displs_B(row_size);
    displs_B[0] = 0;
    for (int i = 1; i < row_size; ++i) {
        displs_B[i] = displs_B[i - 1] + counts_B[i - 1];
    }
    int total_nnz_B = (row_size > 0) ? (displs_B[row_size - 1] + counts_B[row_size - 1]) : 0;

    std::vector<int> send_B_i(local_nnz_B), send_B_j(local_nnz_B), send_B_v(local_nnz_B);
    for (int idx = 0; idx < local_nnz_B; ++idx) {
        send_B_i[idx] = B[idx].first.first;   // row k
        send_B_j[idx] = B[idx].first.second;  // col j
        send_B_v[idx] = B[idx].second;        // value
    }

    std::vector<int> all_B_i(total_nnz_B), all_B_j(total_nnz_B), all_B_v(total_nnz_B);

    int *recv_B_i_ptr = total_nnz_B ? all_B_i.data() : nullptr;
    int *recv_B_j_ptr = total_nnz_B ? all_B_j.data() : nullptr;
    int *recv_B_v_ptr = total_nnz_B ? all_B_v.data() : nullptr;
    const int *send_B_i_ptr = local_nnz_B ? send_B_i.data() : nullptr;
    const int *send_B_j_ptr = local_nnz_B ? send_B_j.data() : nullptr;
    const int *send_B_v_ptr = local_nnz_B ? send_B_v.data() : nullptr;

    MPI_Allgatherv(send_B_i_ptr, local_nnz_B, MPI_INT,
                   recv_B_i_ptr, counts_B.data(), displs_B.data(), MPI_INT,
                   row_comm);
    MPI_Allgatherv(send_B_j_ptr, local_nnz_B, MPI_INT,
                   recv_B_j_ptr, counts_B.data(), displs_B.data(), MPI_INT,
                   row_comm);
    MPI_Allgatherv(send_B_v_ptr, local_nnz_B, MPI_INT,
                   recv_B_v_ptr, counts_B.data(), displs_B.data(), MPI_INT,
                   row_comm);

    // If one of the matrices is empty in this row/column, no work to do
    if (total_nnz_A == 0 || total_nnz_B == 0) {
        return;
    }

    // --- 3. Build A_by_k and B_by_k ---
    // A is m x p, so entries are (i, k)
    std::map<int, std::vector<std::pair<int,int>>> A_by_k; // k -> list of (i, A[i,k])
    for (int idx = 0; idx < total_nnz_A; ++idx) {
        int i = all_A_i[idx];
        int k = all_A_j[idx];
        int val = all_A_v[idx];
        A_by_k[k].emplace_back(i, val);
    }

    // B is p x n, so entries are (k, j)
    std::map<int, std::vector<std::pair<int,int>>> B_by_k; // k -> list of (j, B[k,j])
    for (int idx = 0; idx < total_nnz_B; ++idx) {
        int k = all_B_i[idx];
        int j = all_B_j[idx];
        int val = all_B_v[idx];
        B_by_k[k].emplace_back(j, val);
    }

    // --- 4. Local sparse GEMM: for each k, compute outer products ---
    // We accumulate into a sparse map of C entries keyed by (i,j)
    std::map<std::pair<int,int>, int> C_map;

    for (const auto &ak_entry : A_by_k) {
        int k = ak_entry.first;
        auto itB = B_by_k.find(k);
        if (itB == B_by_k.end()) {
            continue; // no matching B row for this k on this rank
        }

        const auto &avec = ak_entry.second;   // (i, A[i,k]) for this k
        const auto &bvec = itB->second;       // (j, B[k,j]) for this k

        for (const auto &aij : avec) {
            int i = aij.first;
            int a_val = aij.second;
            for (const auto &bkj : bvec) {
                int j = bkj.first;
                int b_val = bkj.second;

                int prod = times(a_val, b_val);
                auto key = std::make_pair(i, j);
                auto itC = C_map.find(key);
                if (itC == C_map.end()) {
                    C_map[key] = prod;
                } else {
                    itC->second = plus(itC->second, prod);
                }
            }
        }
    }

    // --- 5. Convert C_map back into the vector-of-triplets format ---
    C.reserve(C_map.size());
    for (const auto &entry : C_map) {
        const auto &idx = entry.first;
        int value = entry.second;
        C.push_back(std::make_pair(std::make_pair(idx.first, idx.second), value));
    }
    // std::map keeps keys ordered, so C is already sorted by (i,j).
}

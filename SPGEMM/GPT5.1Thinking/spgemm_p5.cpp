#include <vector>
#include <map>
#include <unordered_map>
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
               std::function<int(int, int)> plus,
               std::function<int(int, int)> times,
               MPI_Comm row_comm, MPI_Comm col_comm)
{
    // We don’t actually need the global dimensions here, but keep them
    // to match the interface and avoid unused-parameter warnings.
    (void)m;
    (void)p;
    (void)n;

    // Helper: allgather variable-length (i,j,val) triples in a communicator.
    auto allgather_triples =
        [](MPI_Comm comm,
           const std::vector<int> &send_i,
           const std::vector<int> &send_j,
           const std::vector<int> &send_val,
           std::vector<int> &recv_i,
           std::vector<int> &recv_j,
           std::vector<int> &recv_val)
        {
            int comm_size;
            MPI_Comm_size(comm, &comm_size);

            int local_nnz = static_cast<int>(send_i.size());

            // First gather counts
            std::vector<int> counts(comm_size);
            MPI_Allgather(&local_nnz, 1, MPI_INT,
                          counts.data(), 1, MPI_INT,
                          comm);

            // Build displacements
            std::vector<int> displs(comm_size, 0);
            for (int r = 1; r < comm_size; ++r) {
                displs[r] = displs[r - 1] + counts[r - 1];
            }
            int total = (comm_size == 0) ? 0 : displs[comm_size - 1] + counts[comm_size - 1];

            recv_i.resize(total);
            recv_j.resize(total);
            recv_val.resize(total);

            // Allgatherv for each of i, j, value
            MPI_Allgatherv(send_i.data(), local_nnz, MPI_INT,
                           recv_i.data(), counts.data(), displs.data(), MPI_INT,
                           comm);
            MPI_Allgatherv(send_j.data(), local_nnz, MPI_INT,
                           recv_j.data(), counts.data(), displs.data(), MPI_INT,
                           comm);
            MPI_Allgatherv(send_val.data(), local_nnz, MPI_INT,
                           recv_val.data(), counts.data(), displs.data(), MPI_INT,
                           comm);
        };

    // 1. Flatten local A and B into (i,j,val) arrays
    std::vector<int> A_i, A_j, A_val;
    std::vector<int> B_i, B_j, B_val;

    A_i.reserve(A.size());
    A_j.reserve(A.size());
    A_val.reserve(A.size());
    B_i.reserve(B.size());
    B_j.reserve(B.size());
    B_val.reserve(B.size());

    flatten_matrix(A, A_i, A_j, A_val);
    flatten_matrix(B, B_i, B_j, B_val);

    // 2. Replicate A along rows, B along columns of the 2-D grid
    std::vector<int> A_row_i, A_row_j, A_row_val;
    std::vector<int> B_col_i, B_col_j, B_col_val;

    // All A entries for this process *row* (across all process columns)
    allgather_triples(row_comm, A_i, A_j, A_val,
                      A_row_i, A_row_j, A_row_val);

    // All B entries for this process *column* (across all process rows)
    allgather_triples(col_comm, B_i, B_j, B_val,
                      B_col_i, B_col_j, B_col_val);

    // 3. Build a row-wise structure for B: for each k, we store (j, B[k,j])
    //    using global indices. This lets us quickly find all B entries that
    //    share a given inner index k.
    std::unordered_map<int, std::vector<std::pair<int,int>>> B_by_row;
    B_by_row.reserve(B_col_i.size());

    for (std::size_t idx = 0; idx < B_col_i.size(); ++idx) {
        int k  = B_col_i[idx];   // row index in B (inner dimension)
        int j  = B_col_j[idx];   // column index in B
        int bv = B_col_val[idx]; // value
        B_by_row[k].emplace_back(j, bv);
    }

    // 4. Local sparse multiplication: A_row (i,k) times B_by_row(k, j)
    //    We accumulate contributions into a sparse map keyed by (i,j).
    //    Use std::map so iteration is deterministic, but correctness does not
    //    require sorted output because the final gather + correctness_check
    //    re-sorts on rank 0.
    std::map<std::pair<int,int>, int> C_acc;

    for (std::size_t idx = 0; idx < A_row_i.size(); ++idx) {
        int i  = A_row_i[idx];   // row in A / C
        int k  = A_row_j[idx];   // inner index
        int av = A_row_val[idx]; // value A[i,k]

        auto it = B_by_row.find(k);
        if (it == B_by_row.end()) {
            continue; // no matching B row for this k
        }

        const auto &row_entries = it->second; // vector of (j, B[k,j])
        for (const auto &cj : row_entries) {
            int j   = cj.first;
            int bv  = cj.second;
            int prod = times(av, bv); // typically av * bv

            auto key = std::make_pair(i, j);
            auto cit = C_acc.find(key);
            if (cit == C_acc.end()) {
                // First contribution for this (i,j)
                C_acc.emplace(key, prod);
            } else {
                // Accumulate with the provided plus operation (typically +)
                cit->second = plus(cit->second, prod);
            }
        }
    }

    // 5. Move accumulated (i,j,val) into C in the required sparse format.
    C.clear();
    C.reserve(C_acc.size());
    for (const auto &kv : C_acc) {
        const auto &idx = kv.first;   // pair<i,j>
        int value       = kv.second;  // accumulated C[i,j]
        C.emplace_back(idx, value);
    }
}

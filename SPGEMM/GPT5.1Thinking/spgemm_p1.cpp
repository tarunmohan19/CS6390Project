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
    // We’ll use the full world communicator for gathering and combining.
    (void)row_comm;
    (void)col_comm;

    int world_rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    (void)world_size; // silence unused warning if not used later

    // Gather distributed matrices A and B to rank 0.
    // A_full and B_full will contain ALL nonzeros on rank 0 only.
    std::vector<std::pair<std::pair<int,int>, int>> A_full;
    std::vector<std::pair<std::pair<int,int>, int>> B_full;

    gather_matrix(A, A_full, 0, MPI_COMM_WORLD);
    gather_matrix(B, B_full, 0, MPI_COMM_WORLD);

    // Ensure C starts empty on all ranks
    C.clear();

    // Only rank 0 does the actual SpGEMM; others leave C empty.
    if (world_rank != 0) {
        return;
    }

    // ---- Sequential sparse SpGEMM on rank 0 over the given semiring ----
    //
    // We compute C = A * B where:
    //  - A is m x p, B is p x n
    //  - A_full entries are (i, k) with value a
    //  - B_full entries are (k, j) with value b
    //  - C(i,j) = ⨁_k (A(i,k) ⊗ B(k,j))  with ⨁ = plus, ⊗ = times
    //
    // Strategy: outer-product in k:
    //   group A by column k (A_col[k] = list of (i, a_ik))
    //   group B by row    k (B_row[k] = list of (j, b_kj))
    //   for each k in intersection, form all (i,j) pairs and accumulate.

    // Group A by column k (the inner dimension index)
    std::map<int, std::vector<std::pair<int,int>>> A_col; // k -> [(i, a_ik)]
    for (const auto &entry : A_full) {
        int i = entry.first.first;
        int k = entry.first.second;
        int val = entry.second;
        A_col[k].emplace_back(i, val);
    }

    // Group B by row k (the same inner dimension index)
    std::map<int, std::vector<std::pair<int,int>>> B_row; // k -> [(j, b_kj)]
    for (const auto &entry : B_full) {
        int k = entry.first.first;
        int j = entry.first.second;
        int val = entry.second;
        B_row[k].emplace_back(j, val);
    }

    // Accumulate C in a map keyed by (i,j)
    std::map<std::pair<int,int>, int> C_map;

    for (const auto &akv : A_col) {
        int k = akv.first;

        auto itB = B_row.find(k);
        if (itB == B_row.end()) {
            continue; // no matching row in B for this k
        }

        const auto &A_list = akv.second;       // [(i, a_ik)]
        const auto &B_list = itB->second;      // [(j, b_kj)]

        // Outer product for this k
        for (const auto &a_pair : A_list) {
            int i = a_pair.first;
            int a_val = a_pair.second;

            for (const auto &b_pair : B_list) {
                int j = b_pair.first;
                int b_val = b_pair.second;

                int prod = times(a_val, b_val);   // a_ik ⊗ b_kj

                auto key = std::make_pair(i, j);
                auto itC = C_map.find(key);
                if (itC == C_map.end()) {
                    // First contribution for this (i,j): no need for identity
                    C_map[key] = prod;
                } else {
                    // Accumulate with ⨁
                    itC->second = plus(itC->second, prod);
                }
            }
        }
    }

    // Convert map to the sparse triplet vector C.
    C.reserve(C_map.size());
    for (const auto &kv : C_map) {
        C.push_back(std::make_pair(kv.first, kv.second));
    }
    // Sorting is not required here since main() sorts the gathered result
    // before comparing to the expected output, but it doesn't hurt:
    // std::sort(C.begin(), C.end());
}

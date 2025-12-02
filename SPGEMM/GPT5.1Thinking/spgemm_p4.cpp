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
    // We will centralize the sparse GEMM on world rank 0.
    // 1. Gather all nonzeros of A and B from all ranks to world rank 0.
    // 2. Rank 0 performs serial SpGEMM (on the given semiring).
    // 3. Rank 0 writes the full result into C; other ranks leave C empty.

    (void)row_comm; // not used in this implementation
    (void)col_comm; // not used in this implementation

    int world_rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    const int root = 0;

    // Ensure C starts empty
    C.clear();

    // ---- Gather A ----
    int local_nnz_A = static_cast<int>(A.size());

    std::vector<int> recvcountsA;
    if (world_rank == root) {
        recvcountsA.resize(world_size);
    }

    MPI_Gather(&local_nnz_A, 1, MPI_INT,
               world_rank == root ? recvcountsA.data() : nullptr,
               1, MPI_INT, root, MPI_COMM_WORLD);

    // Flatten local A into three arrays: rows, cols, values
    std::vector<int> sendA_i(local_nnz_A);
    std::vector<int> sendA_j(local_nnz_A);
    std::vector<int> sendA_v(local_nnz_A);

    for (int idx = 0; idx < local_nnz_A; ++idx) {
        sendA_i[idx] = A[idx].first.first;

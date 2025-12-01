#include <vector>
#include <map>
#include <algorithm>
#include <utility>
#include <iostream>
#include <mpi.h>
#include <cassert>
#include <functional>

void spgemm_2d(int m, int p, int n,
               std::vector<std::pair<std::pair<int,int>, int>> &A,
               std::vector<std::pair<std::pair<int,int>, int>> &B,
               std::vector<std::pair<std::pair<int,int>, int>> &C,
               std::function<int(int, int)> plus, std::function<int(int, int)> times,
               MPI_Comm row_comm, MPI_Comm col_comm)
{
    int row_rank, col_rank, row_size, col_size;
    MPI_Comm_rank(row_comm, &row_rank);
    MPI_Comm_rank(col_comm, &col_rank);
    MPI_Comm_size(row_comm, &row_size);
    MPI_Comm_size(col_comm, &col_size);

    assert(row_size == col_size); // Must be a square process grid
    int q = row_size;

    // Organize local matrices by row/col for fast lookup
    std::map<int, std::vector<std::pair<int,int>>> localA_rows;
    std::map<int, std::vector<std::pair<int,int>>> localB_cols;

    for (auto &[idx, val] : A)
        localA_rows[idx.first].emplace_back(idx.second, val);

    for (auto &[idx, val] : B)
        localB_cols[idx.second].emplace_back(idx.first, val);

    // Local C accumulator
    std::map<std::pair<int,int>, int> localC;

    // For each stage, we broadcast a block along row and column
    std::vector<std::pair<std::pair<int,int>, int>> A_buf = A;
    std::vector<std::pair<std::pair<int,int>, int>> B_buf = B;

    for (int stage = 0; stage < q; ++stage) {
        int A_root = (row_rank + stage) % q;
        int B_root = (col_rank + stage) % q;

        // Broadcast A block along the row
        int A_count = static_cast<int>(A_buf.size());
        MPI_Bcast(&A_count, 1, MPI_INT, A_root, row_comm);
        if (row_rank != A_root)
            A_buf.resize(A_count);
        for (int i = 0; i < A_count; ++i) {
            int data[3];
            if (row_rank == A_root) {
                data[0] = A_buf[i].first.first;
                data[1] = A_buf[i].first.second;
                data[2] = A_buf[i].second;
            }
            MPI_Bcast(data, 3, MPI_INT, A_root, row_comm);
            if (row_rank != A_root) {
                A_buf[i] = {{data[0], data[1]}, data[2]};
            }
        }

        // Broadcast B block along the column
        int B_count = static_cast<int>(B_buf.size());
        MPI_Bcast(&B_count, 1, MPI_INT, B_root, col_comm);
        if (col_rank != B_root)
            B_buf.resize(B_count);
        for (int i = 0; i < B_count; ++i) {
            int data[3];
            if (col_rank == B_root) {
                data[0] = B_buf[i].first.first;
                data[1] = B_buf[i].first.second;
                data[2] = B_buf[i].second;
            }
            MPI_Bcast(data, 3, MPI_INT, B_root, col_comm);
            if (col_rank != B_root) {
                B_buf[i] = {{data[0], data[1]}, data[2]};
            }
        }

        // Multiply received blocks: C += A_buf * B_buf
        std::map<int, std::vector<std::pair<int,int>>> stageA_rows;
        std::map<int, std::vector<std::pair<int,int>>> stageB_cols;

        for (auto &[idx, val] : A_buf)
            stageA_rows[idx.first].emplace_back(idx.second, val);

        for (auto &[idx, val] : B_buf)
            stageB_cols[idx.second].emplace_back(idx.first, val);

        for (auto &[i, row_vec] : stageA_rows) {
            for (auto &[k, a_val] : row_vec) {
                if (stageB_cols.count(k)) {
                    for (auto &[j, b_val] : stageB_cols[k]) {
                        localC[{i,j}] = plus(localC[{i,j}], times(a_val, b_val));
                    }
                }
            }
        }
    }

    // Flatten map to vector
    for (auto &[idx, val] : localC) {
        if (val != 0) {
            C.push_back({idx, val});
        }
    }
}

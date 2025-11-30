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
    // TODO: Write your code here
    int proc_row, proc_col;
    MPI_Comm_rank(col_comm, &proc_row);
    MPI_Comm_rank(row_comm, &proc_col);

    int q;
    MPI_Comm_size(row_comm, &q);

    std::map<std::pair<int, int>, int> local_C;

    for (int k = 0; k < q; k++) {
        int A_block_count = (proc_col == k) ? A.size() : 0;
        int B_block_count = (proc_row == k) ? B.size() : 0;
        MPI_Request reqs[2];
        MPI_Ibcast(&A_block_count, 1, MPI_INT, k, row_comm, &reqs[0]);
        MPI_Ibcast(&B_block_count, 1, MPI_INT, k, col_comm, &reqs[1]);
        MPI_Waitall(2, reqs, MPI_STATUSES_IGNORE);

        std::vector<int> A_buffer, B_buffer;
        if (proc_col == k) {
            A_buffer.resize(A_block_count * 3);
            for (size_t i = 0; i < A.size(); i++) {
                A_buffer[3*i] = A[i].first.first;
                A_buffer[3*i+1] = A[i].first.second;
                A_buffer[3*i+2] = A[i].second;
            }
        }
        if (proc_row == k) {
            B_buffer.resize(B_block_count * 3);
            for (size_t i = 0; i < B.size(); i++) {
                B_buffer[3*i] = B[i].first.first;
                B_buffer[3*i+1] = B[i].first.second;
                B_buffer[3*i+2] = B[i].second;
            }
        }

        MPI_Request breqs[2];
        A_buffer.resize(A_block_count * 3);
        MPI_Ibcast(A_buffer.data(), A_block_count * 3, MPI_INT, k, row_comm, &breqs[0]);
        B_buffer.resize(B_block_count * 3);
        MPI_Ibcast(B_buffer.data(), B_block_count * 3, MPI_INT, k, col_comm, &breqs[1]);
        MPI_Waitall(2, breqs, MPI_STATUSES_IGNORE);

        std::vector<std::pair<std::pair<int,int>, int>> A_block(A_block_count);
        for (int i = 0; i < A_block_count; i++) {
            A_block[i] = {{A_buffer[3*i], A_buffer[3*i+1]}, A_buffer[3*i+2]};
        }

        std::map<int, std::vector<std::pair<int, int>>> B_map;
        for (int i = 0; i < B_block_count; i++) {
            B_map[B_buffer[3*i]].emplace_back(B_buffer[3*i+1],  B_buffer[3*i+2]);
        }

        std::vector<std::pair<std::pair<int, int>, int>> temp_C;
        for (const auto& a_entry : A_block) {
            int a_col = a_entry.first.second;
            auto b_it = B_map.find(a_col);
            if (b_it == B_map.end()) continue;

            for (const auto& b_entry : b_it->second) {
                int prod = times(a_entry.second, b_entry.second);
                temp_C.emplace_back(std::make_pair(a_entry.first.first, b_entry.first), prod);
            }
        }

        std::sort(temp_C.begin(), temp_C.end(), [](const auto& a, const auto& b) { return a.first < b.first; });
        
        if (!temp_C.empty()) {
            auto current = temp_C[0];
            for (size_t i = 1; i < temp_C.size(); ++i) {
                if (temp_C[i].first == current.first) {
                    current.second = plus(current.second, temp_C[i].second);
                } else {
                    auto it = local_C.find(current.first);
                    if (it != local_C.end()) {
                        it->second = plus(it->second, current.second);
                    } else {
                        local_C[current.first] = current.second;
                    }
                    current = temp_C[i];
                }
            }
            auto it = local_C.find(current.first);
            if (it != local_C.end()) {
                it->second = plus(it->second, current.second);
            } else {
                local_C[current.first] = current.second;
            }
        }
    }

    C.clear();
    C.reserve(local_C.size());
    for (const auto& entry : local_C) {
        C.emplace_back(entry.first, entry.second);
    }
}

void spgemm_2d(int m, int p, int n,
               std::vector<std::pair<std::pair<int,int>, int>> &A,
               std::vector<std::pair<std::pair<int,int>, int>> &B,
               std::vector<std::pair<std::pair<int,int>, int>> &C,
               std::function<int(int, int)> plus, std::function<int(int, int)> times,
               MPI_Comm row_comm, MPI_Comm col_comm)
{
    int rank_row, rank_col, row_size, col_size;
    MPI_Comm_rank(row_comm, &rank_row);
    MPI_Comm_rank(col_comm, &rank_col);
    MPI_Comm_size(row_comm, &row_size);
    MPI_Comm_size(col_comm, &col_size);
    assert(row_size == col_size);

    int q = row_size; // sqrt(P)
    
    // Map local A/B blocks for quick access
    std::map<std::pair<int,int>, int> local_A, local_B, local_C;
    for (auto &entry : A) local_A[entry.first] = entry.second;
    for (auto &entry : B) local_B[entry.first] = entry.second;

    // Buffers for broadcasts
    std::vector<std::pair<std::pair<int,int>, int>> A_buffer, B_buffer;

    for (int k = 0; k < q; ++k) {
        // Determine the rank that owns the block to broadcast
        int A_owner = k;
        int B_owner = k;

        // Prepare A block to broadcast along the row
        if (rank_col == A_owner) {
            A_buffer.clear();
            for (auto &[idx, val] : local_A) {
                if (idx.second / q == k) {  // column block index
                    A_buffer.push_back({idx, val});
                }
            }
        }
        // Broadcast along row_comm
        int A_size = static_cast<int>(A_buffer.size());
        MPI_Bcast(&A_size, 1, MPI_INT, A_owner, row_comm);
        if (rank_col != A_owner) A_buffer.resize(A_size);
        MPI_Bcast(A_buffer.data(), A_size * sizeof(std::pair<std::pair<int,int>, int>), MPI_BYTE, A_owner, row_comm);

        // Prepare B block to broadcast along the column
        if (rank_row == B_owner) {
            B_buffer.clear();
            for (auto &[idx, val] : local_B) {
                if (idx.first / q == k) {  // row block index
                    B_buffer.push_back({idx, val});
                }
            }
        }
        // Broadcast along col_comm
        int B_size = static_cast<int>(B_buffer.size());
        MPI_Bcast(&B_size, 1, MPI_INT, B_owner, col_comm);
        if (rank_row != B_owner) B_buffer.resize(B_size);
        MPI_Bcast(B_buffer.data(), B_size * sizeof(std::pair<std::pair<int,int>, int>), MPI_BYTE, B_owner, col_comm);

        // Multiply local A_buffer and B_buffer
        for (auto &[a_idx, a_val] : A_buffer) {
            int i = a_idx.first;
            int kA = a_idx.second;
            for (auto &[b_idx, b_val] : B_buffer) {
                int kB = b_idx.first;
                int j = b_idx.second;
                if (kA == kB) {
                    std::pair<int,int> C_idx = {i, j};
                    local_C[C_idx] = plus(local_C[C_idx], times(a_val, b_val));
                }
            }
        }
    }

    // Convert local_C map to vector
    C.clear();
    for (auto &[idx, val] : local_C) {
        if (val != 0) C.push_back({idx, val});
    }
}

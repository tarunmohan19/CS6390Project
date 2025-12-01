void spgemm_2d(int m, int p, int n,
               std::vector<std::pair<std::pair<int,int>, int>> &A,
               std::vector<std::pair<std::pair<int,int>, int>> &B,
               std::vector<std::pair<std::pair<int,int>, int>> &C,
               std::function<int(int, int)> plus, std::function<int(int, int)> times,
               MPI_Comm row_comm, MPI_Comm col_comm)
{
    int rank_row, rank_col, size_row, size_col;
    MPI_Comm_rank(row_comm, &rank_row);
    MPI_Comm_rank(col_comm, &rank_col);
    MPI_Comm_size(row_comm, &size_row);
    MPI_Comm_size(col_comm, &size_col);

    assert(size_row == size_col); // square 2D grid

    // Step 1: Convert local A and B to map-of-maps for fast access
    std::map<int, std::map<int, int>> local_A; // row -> (col -> val)
    std::map<int, std::map<int, int>> local_B; // row -> (col -> val)
    
    for (auto &[idx, val] : A) {
        local_A[idx.first][idx.second] = val;
    }
    for (auto &[idx, val] : B) {
        local_B[idx.first][idx.second] = val;
    }

    // Step 2: Initialize local C as map-of-maps
    std::map<int, std::map<int, int>> local_C;

    // Step 3: Broadcast blocks of A along rows, B along columns
    for (int k = 0; k < size_row; ++k) {
        // Determine root for this broadcast
        int A_root = k;
        int B_root = k;

        // Prepare buffers for A
        std::vector<int> send_A_idx1, send_A_idx2, send_A_val;
        std::vector<int> recv_A_idx1, recv_A_idx2, recv_A_val;

        if (rank_col == A_root) {
            for (auto &[row, colmap] : local_A) {
                for (auto &[col, val] : colmap) {
                    send_A_idx1.push_back(row);
                    send_A_idx2.push_back(col);
                    send_A_val.push_back(val);
                }
            }
        }

        // Broadcast A block along row
        int A_size = send_A_idx1.size();
        MPI_Bcast(&A_size, 1, MPI_INT, A_root, row_comm);
        recv_A_idx1.resize(A_size);
        recv_A_idx2.resize(A_size);
        recv_A_val.resize(A_size);

        if (rank_col == A_root) {
            MPI_Bcast(send_A_idx1.data(), A_size, MPI_INT, A_root, row_comm);
            MPI_Bcast(send_A_idx2.data(), A_size, MPI_INT, A_root, row_comm);
            MPI_Bcast(send_A_val.data(), A_size, MPI_INT, A_root, row_comm);
            recv_A_idx1 = send_A_idx1;
            recv_A_idx2 = send_A_idx2;
            recv_A_val = send_A_val;
        } else {
            MPI_Bcast(recv_A_idx1.data(), A_size, MPI_INT, A_root, row_comm);
            MPI_Bcast(recv_A_idx2.data(), A_size, MPI_INT, A_root, row_comm);
            MPI_Bcast(recv_A_val.data(), A_size, MPI_INT, A_root, row_comm);
        }

        // Build broadcasted A block
        std::map<int, std::map<int,int>> A_block;
        for (int i = 0; i < A_size; ++i) {
            A_block[recv_A_idx1[i]][recv_A_idx2[i]] = recv_A_val[i];
        }

        // Prepare buffers for B
        std::vector<int> send_B_idx1, send_B_idx2, send_B_val;
        std::vector<int> recv_B_idx1, recv_B_idx2, recv_B_val;

        if (rank_row == B_root) {
            for (auto &[row, colmap] : local_B) {
                for (auto &[col, val] : colmap) {
                    send_B_idx1.push_back(row);
                    send_B_idx2.push_back(col);
                    send_B_val.push_back(val);
                }
            }
        }

        // Broadcast B block along column
        int B_size = send_B_idx1.size();
        MPI_Bcast(&B_size, 1, MPI_INT, B_root, col_comm);
        recv_B_idx1.resize(B_size);
        recv_B_idx2.resize(B_size);
        recv_B_val.resize(B_size);

        if (rank_row == B_root) {
            MPI_Bcast(send_B_idx1.data(), B_size, MPI_INT, B_root, col_comm);
            MPI_Bcast(send_B_idx2.data(), B_size, MPI_INT, B_root, col_comm);
            MPI_Bcast(send_B_val.data(), B_size, MPI_INT, B_root, col_comm);
            recv_B_idx1 = send_B_idx1;
            recv_B_idx2 = send_B_idx2;
            recv_B_val = send_B_val;
        } else {
            MPI_Bcast(recv_B_idx1.data(), B_size, MPI_INT, B_root, col_comm);
            MPI_Bcast(recv_B_idx2.data(), B_size, MPI_INT, B_root, col_comm);
            MPI_Bcast(recv_B_val.data(), B_size, MPI_INT, B_root, col_comm);
        }

        // Build broadcasted B block
        std::map<int, std::map<int,int>> B_block;
        for (int i = 0; i < B_size; ++i) {
            B_block[recv_B_idx1[i]][recv_B_idx2[i]] = recv_B_val[i];
        }

        // Step 4: Multiply A_block * B_block locally
        for (auto &[i, row_map] : A_block) {
            for (auto &[k_idx, a_val] : row_map) {
                if (B_block.find(k_idx) != B_block.end()) {
                    for (auto &[j, b_val] : B_block[k_idx]) {
                        local_C[i][j] = plus(local_C[i][j], times(a_val, b_val));
                    }
                }
            }
        }
    }

    // Step 5: Flatten local_C to output vector
    C.clear();
    for (auto &[i, row_map] : local_C) {
        for (auto &[j, val] : row_map) {
            if (val != 0) {
                C.push_back({{i,j}, val});
            }
        }
    }
}

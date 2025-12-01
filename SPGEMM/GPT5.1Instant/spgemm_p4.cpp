void spgemm_2d(int m, int p, int n,
               std::vector<std::pair<std::pair<int,int>, int>> &A_local,
               std::vector<std::pair<std::pair<int,int>, int>> &B_local,
               std::vector<std::pair<std::pair<int,int>, int>> &C_local,
               std::function<int(int,int)> plus, std::function<int(int,int)> times,
               MPI_Comm row_comm, MPI_Comm col_comm)
{
    int rank_row, rank_col;
    int q_row, q_col;
    MPI_Comm_rank(row_comm, &rank_row);
    MPI_Comm_rank(col_comm, &rank_col);
    MPI_Comm_size(row_comm, &q_col);
    MPI_Comm_size(col_comm, &q_row);

    C_local.clear();

    // Loop over SUMMA steps
    for (int k = 0; k < q_row; ++k) {
        // Step 1: Broadcast A block along row
        std::vector<std::pair<std::pair<int,int>, int>> A_bcast;
        if (rank_col == k) A_bcast = A_local;
        // First serialize to send buffer
        int send_count = A_bcast.size() * 3; // i,j,value
        std::vector<int> sendbuf(send_count);
        for (int idx = 0; idx < (int)A_bcast.size(); ++idx) {
            sendbuf[idx*3 + 0] = A_bcast[idx].first.first;
            sendbuf[idx*3 + 1] = A_bcast[idx].first.second;
            sendbuf[idx*3 + 2] = A_bcast[idx].second;
        }

        // Broadcast size first
        int recv_count;
        if (rank_col == k) recv_count = send_count;
        MPI_Bcast(&recv_count, 1, MPI_INT, k, row_comm);

        if (rank_col != k) sendbuf.resize(recv_count);

        MPI_Bcast(sendbuf.data(), recv_count, MPI_INT, k, row_comm);

        // Deserialize
        std::vector<std::pair<std::pair<int,int>, int>> A_block;
        for (int idx = 0; idx < recv_count / 3; ++idx) {
            A_block.push_back({{sendbuf[idx*3], sendbuf[idx*3+1]}, sendbuf[idx*3+2]});
        }

        // Step 2: Broadcast B block along column
        std::vector<std::pair<std::pair<int,int>, int>> B_bcast;
        if (rank_row == k) B_bcast = B_local;

        int send_count_B = B_bcast.size() * 3;
        std::vector<int> sendbufB(send_count_B);
        for (int idx = 0; idx < (int)B_bcast.size(); ++idx) {
            sendbufB[idx*3+0] = B_bcast[idx].first.first;
            sendbufB[idx*3+1] = B_bcast[idx].first.second;
            sendbufB[idx*3+2] = B_bcast[idx].second;
        }

        int recv_count_B;
        if (rank_row == k) recv_count_B = send_count_B;
        MPI_Bcast(&recv_count_B, 1, MPI_INT, k, col_comm);

        if (rank_row != k) sendbufB.resize(recv_count_B);

        MPI_Bcast(sendbufB.data(), recv_count_B, MPI_INT, k, col_comm);

        // Deserialize
        std::vector<std::pair<std::pair<int,int>, int>> B_block;
        for (int idx = 0; idx < recv_count_B / 3; ++idx) {
            B_block.push_back({{sendbufB[idx*3], sendbufB[idx*3+1]}, sendbufB[idx*3+2]});
        }

        // Step 3: Local multiply and add to C_local
        auto C_step = local_spgemm(A_block, B_block, plus, times);
        // Merge C_step into C_local
        std::map<std::pair<int,int>, int> C_map;
        for (auto [idx, val] : C_local) C_map[idx] = val;
        for (auto [idx, val] : C_step) {
            C_map[idx] = C_map.find(idx) == C_map.end() ? val : plus(C_map[idx], val);
        }
        C_local.clear();
        for (auto &kv : C_map) C_local.push_back({kv.first, kv.second});
    }
}


std::vector<std::pair<std::pair<int,int>, int>> local_spgemm(
    const std::vector<std::pair<std::pair<int,int>, int>>& A_block,
    const std::vector<std::pair<std::pair<int,int>, int>>& B_block,
    std::function<int(int,int)> plus,
    std::function<int(int,int)> times)
{
    std::map<std::pair<int,int>, int> C_map;
    // Convert B_block to map for faster access by row
    std::map<int, std::vector<std::pair<int,int>>> B_by_row;
    for (auto [idx, val] : B_block) {
        B_by_row[idx.first].push_back({idx.second, val});
    }

    for (auto [idxA, valA] : A_block) {
        int i = idxA.first;
        int k = idxA.second;
        if (B_by_row.find(k) != B_by_row.end()) {
            for (auto [j, valB] : B_by_row[k]) {
                C_map[{i,j}] = C_map.find({i,j}) == C_map.end() 
                               ? times(valA, valB) 
                               : plus(C_map[{i,j}], times(valA, valB));
            }
        }
    }

    std::vector<std::pair<std::pair<int,int>, int>> C_local;
    for (auto &kv : C_map) {
        C_local.push_back({kv.first, kv.second});
    }
    return C_local;
}

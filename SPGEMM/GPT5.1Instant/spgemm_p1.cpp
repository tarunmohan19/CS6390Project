void spgemm_2d(int m, int p, int n,
               std::vector<std::pair<std::pair<int,int>, int>> &A,
               std::vector<std::pair<std::pair<int,int>, int>> &B,
               std::vector<std::pair<std::pair<int,int>, int>> &C,
               std::function<int(int, int)> plus, std::function<int(int, int)> times,
               MPI_Comm row_comm, MPI_Comm col_comm)
{
    int row_rank, row_size;
    int col_rank, col_size;
    MPI_Comm_rank(row_comm, &row_rank);
    MPI_Comm_size(row_comm, &row_size);
    MPI_Comm_rank(col_comm, &col_rank);
    MPI_Comm_size(col_comm, &col_size);

    assert(row_size == col_size); // square 2D grid

    int q = row_size; // sqrt(num_procs)

    // Step 1: Pack local A and B into maps for faster lookup
    // A_map: row -> vector of (col,value)
    std::map<int, std::vector<std::pair<int,int>>> A_map;
    for (auto &[idx, val] : A) {
        A_map[idx.first].emplace_back(idx.second, val);
    }

    std::map<int, std::vector<std::pair<int,int>>> B_map; // B is A_T locally
    for (auto &[idx, val] : B) {
        B_map[idx.first].emplace_back(idx.second, val);
    }

    // Step 2: Broadcast A along rows, B along columns
    for (int k = 0; k < q; ++k) {
        std::vector<std::pair<std::pair<int,int>, int>> A_block;
        std::vector<std::pair<std::pair<int,int>, int>> B_block;

        if (col_rank == k) {
            // Send local A to row
            A_block = A;
        }
        if (row_rank == k) {
            // Send local B to column
            B_block = B;
        }

        // Broadcast along row
        int A_count = A_block.size();
        MPI_Bcast(&A_count, 1, MPI_INT, k, row_comm);
        if (col_rank != k) A_block.resize(A_count);
        MPI_Bcast(A_block.data(), A_count * sizeof(std::pair<std::pair<int,int>, int>), MPI_BYTE, k, row_comm);

        // Broadcast along column
        int B_count = B_block.size();
        MPI_Bcast(&B_count, 1, MPI_INT, k, col_comm);
        if (row_rank != k) B_block.resize(B_count);
        MPI_Bcast(B_block.data(), B_count * sizeof(std::pair<std::pair<int,int>, int>), MPI_BYTE, k, col_comm);

        // Step 3: Multiply local blocks
        // Convert received blocks to maps
        std::map<int, std::vector<std::pair<int,int>>> A_blk_map;
        for (auto &[idx, val] : A_block) A_blk_map[idx.first].emplace_back(idx.second, val);

        std::map<int, std::vector<std::pair<int,int>>> B_blk_map;
        for (auto &[idx, val] : B_block) B_blk_map[idx.first].emplace_back(idx.second, val);

        // Multiply
        for (auto &[i, vecA] : A_blk_map) {
            for (auto &[kA, valA] : vecA) {
                if (B_blk_map.count(kA) == 0) continue;
                for (auto &[j, valB] : B_blk_map[kA]) {
                    C.push_back({{i, j}, times(valA, valB)});
                }
            }
        }
    }

    // Step 4: Reduce duplicate entries in C (sum duplicates)
    std::map<std::pair<int,int>, int> C_map;
    for (auto &[idx, val] : C) {
        C_map[idx] = C_map.count(idx) ? plus(C_map[idx], val) : val;
    }

    // Write back to C vector
    C.clear();
    for (auto &[idx, val] : C_map) {
        C.push_back({idx, val});
    }
}

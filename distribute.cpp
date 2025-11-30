#include <algorithm>
#include <vector>
#include <iostream>
#include <fstream>
#include <cassert>
#include <cmath>
#include <string>
#include <sstream>
#include "functions.h"

void distribute_matrix_2d(int m, int n, std::vector<std::pair<std::pair<int, int>, int>> &full_matrix,
                          std::vector<std::pair<std::pair<int, int>, int>> &local_matrix,
                          int root, MPI_Comm comm_2d)
{
    // TODO: Write your code here

    /** 
        Ed says that we can assume uniformly dist data
        so we can just use the data row and col to determine
        the correct processor to send the data to
    **/ 

    int rank, size;
    MPI_Comm_rank(comm_2d, &rank);
    MPI_Comm_size(comm_2d, &size);

    int ndims = 2;
    int dims[2], periods[2], my_coords[2];
    MPI_Cart_get(comm_2d, ndims, dims, periods, my_coords);

    // The root needs to figure out which data goes where and send it
    if (rank == root) {
        std::vector< std::vector<std::pair<std::pair<int, int>, int>> > distribution_lists(size);
        
        // Go through the full matrix
        // Determine where the data is supposed to go and add it to the list
        for (auto &entry : full_matrix) {
            int i = entry.first.first;
            int j = entry.first.second;
            int proc_row = (i * dims[0]) / m;
            int proc_col = (j * dims[1]) / n;
            int dest_rank;
            int dest_coords[2] = {proc_row, proc_col};
            MPI_Cart_rank(comm_2d, dest_coords, &dest_rank);
            distribution_lists[dest_rank].push_back(entry);
        }
        
        // send data to other processes
        for (int dest = 0; dest < size; dest++) {
            if (dest == root) {
                local_matrix = distribution_lists[root];
                continue;
            }
            int count = distribution_lists[dest].size();
            MPI_Send(&count, 1, MPI_INT, dest, 0, comm_2d);
            
            
            if (count > 0) {
                std::vector<int> buffer(count * 3);
                for (int k = 0; k < count; k++) {
                    buffer[3 * k]     = distribution_lists[dest][k].first.first;
                    buffer[3 * k + 1] = distribution_lists[dest][k].first.second;
                    buffer[3 * k + 2] = distribution_lists[dest][k].second;
                }
                MPI_Send(buffer.data(), count * 3, MPI_INT, dest, 0, comm_2d);
            }
        }
        // The root needs its own data
        
    }
    // The other processes need to recieve the correct data from the root
    else {
        int count;
        MPI_Recv(&count, 1, MPI_INT, root, 0, comm_2d, MPI_STATUS_IGNORE);
        
        if (count > 0) {
            std::vector<int> buffer(count * 3);
            MPI_Recv(buffer.data(), count * 3, MPI_INT, root, 0, comm_2d, MPI_STATUS_IGNORE);
            
            local_matrix.resize(count);
            for (int k = 0; k < count; k++) {
                int row = buffer[3 * k];
                int col = buffer[3 * k + 1];
                int val = buffer[3 * k + 2];
                local_matrix[k] = std::make_pair(std::make_pair(row, col), val);
            }
        }
    }
}

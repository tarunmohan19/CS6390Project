#include <vector>
#include <utility>
#include <array>
#include <mpi.h>
#include <cassert>
#include <iostream>
#include "functions.h"

void apsp(int n, std::vector<std::pair<std::pair<int, int>, int>> &graph,
          std::vector<std::pair<std::pair<int, int>, int>> &result,
          MPI_Comm row_comm, MPI_Comm col_comm)
{
    std::vector<std::pair<std::pair<int,int>, int>> L = graph;

    int max_iter = 1;
    while (max_iter<n){
        std::vector<std::pair<std::pair<int,int>, int>> L_tmp = std::move(L);
        spgemm_2d(n,n,n,L_tmp,L_tmp,L,
                  // TODO: Choose operation here
                  [](int a, int b){ return std::min(a,b); },
                  // TODO: Choose operation here
                  [](int a, int b){ return a+b; }, row_comm, col_comm);
        max_iter*=2;
    }
    result = L;
}

#include <mpi.h>
#include <iostream>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <sstream>
#include "pi.h"

// Command Line Option Processing
int find_arg_idx(int argc, char** argv, const char* option) {
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], option) == 0) {
            return i;
        }
    }
    return -1;
}

//Function to convert command line input (string) to integer
int stringToInt(const std::string& str) {
    std::istringstream iss(str);
    int result;
    iss >> result;
    return result;
}

int main(int argc, char* argv[]) {

    // Initialize MPI
    MPI_Init(&argc, &argv);

    // Define and initialize variables (Rank and Comm Size) for all the procs
    int rank, size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // Parse Arguments
    if (rank == 0 && argc < 3) {
        std::cerr << "Error: Missing number of points. Use -n <int> to specify." << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    if (find_arg_idx(argc, argv, "-h") >= 0) {
        if (rank == 0) {
            std::cout << "Options:" << std::endl;
            std::cout << "-h: see this help" << std::endl;
            std::cout << "-n <int>: set number of points for the estimation" << std::endl;
        }
        MPI_Finalize();
        return 0;
    }

    if (find_arg_idx(argc, argv, "-n") >= 0) {

        long int n = stringToInt(argv[2]);
        double start_time = MPI_Wtime();
        double result = pi_calc(n);
        double end_time = MPI_Wtime();
        
        if (rank == 0) {
            std::cout << "Estimated Pi: " << result << std::endl;
            std::cout << "Time: " << end_time - start_time << " seconds" << std::endl;
        }
        
        MPI_Finalize();
        return 0;
    }

    //MPI_Finalize
    MPI_Finalize();
    return 0;
}

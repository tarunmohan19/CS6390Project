#include <stdio.h>
#include <ctime>
#include <cstring>
#include <sstream>
#include <iomanip>
#include <iostream>
#include <mpi.h>


double pi_calc(long int n) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Divide points among processes
    long int local_n = n / size;
    long int remainder = n % size;

    // Last process takes the remainder
    if (rank == size - 1) {
        local_n += remainder;
    }

    // Seed random number generator differently for each process
    unsigned int seed = static_cast<unsigned int>(time(NULL)) + rank;
    long int local_count = 0;

    for (long int i = 0; i < local_n; ++i) {
        double x = static_cast<double>(rand_r(&seed)) / RAND_MAX;
        double y = static_cast<double>(rand_r(&seed)) / RAND_MAX;

        if (x * x + y * y <= 1.0) {
            ++local_count;
        }
    }

    long int global_count = 0;

    // Reduce local counts to rank 0
    MPI_Reduce(&local_count, &global_count, 1, MPI_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

    // Rank 0 calculates pi
    if (rank == 0) {
        return 4.0 * static_cast<double>(global_count) / static_cast<double>(n);
    } else {
        return 0.0;  // Other ranks return dummy value
    }
}

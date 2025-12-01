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

    // Each process handles roughly n/size points
    long int points_per_proc = n / size;
    long int local_count = 0;

    // Seed random number generator uniquely per process
    unsigned int seed = static_cast<unsigned int>(time(nullptr)) + rank;
    
    for (long int i = 0; i < points_per_proc; ++i) {
        double x = static_cast<double>(rand_r(&seed)) / RAND_MAX;
        double y = static_cast<double>(rand_r(&seed)) / RAND_MAX;
        if (x * x + y * y <= 1.0) {
            local_count++;
        }
    }

    long int total_count = 0;
    // Reduce local counts to rank 0
    MPI_Reduce(&local_count, &total_count, 1, MPI_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

    // Only rank 0 calculates and returns the final pi value
    double pi_estimate = 0.0;
    if (rank == 0) {
        pi_estimate = 4.0 * static_cast<double>(total_count) / static_cast<double>(n);
    }

    return pi_estimate;
}

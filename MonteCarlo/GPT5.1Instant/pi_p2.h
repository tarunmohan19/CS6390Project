#ifndef PI_H
#define PI_H

#include <mpi.h>
#include <cstdlib>
#include <ctime>
#include <cmath>

double pi_calc(long int n) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Divide work among processes
    long int points_per_proc = n / size;

    // Seed random generator uniquely for each process
    unsigned int seed = static_cast<unsigned int>(time(nullptr)) + rank * 1337;
    srand(seed);

    // Count points inside the unit circle for this process
    long int local_count = 0;
    for (long int i = 0; i < points_per_proc; ++i) {
        double x = static_cast<double>(rand()) / RAND_MAX;
        double y = static_cast<double>(rand()) / RAND_MAX;
        if (x * x + y * y <= 1.0) {
            ++local_count;
        }
    }

    // Reduce local counts to get the global total at rank 0
    long int global_count = 0;
    MPI_Reduce(&local_count, &global_count, 1, MPI_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

    // Rank 0 computes final pi value
    double pi_estimate = 0.0;
    if (rank == 0) {
        pi_estimate = 4.0 * static_cast<double>(global_count) / static_cast<double>(n);
    }

    // Broadcast the result to all processes (optional, useful if all ranks need it)
    MPI_Bcast(&pi_estimate, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    return pi_estimate;
}

#endif // PI_H

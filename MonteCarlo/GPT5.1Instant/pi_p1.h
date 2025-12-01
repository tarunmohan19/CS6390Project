#ifndef PI_H
#define PI_H

#include <mpi.h>
#include <cstdlib>
#include <ctime>

inline double pi_calc(long int n) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Seed random number generator differently for each process
    std::srand(static_cast<unsigned int>(std::time(nullptr)) + rank);

    // Each process computes a portion of the total points
    long int local_n = n / size;
    long int local_inside = 0;

    for (long int i = 0; i < local_n; ++i) {
        double x = static_cast<double>(std::rand()) / RAND_MAX;
        double y = static_cast<double>(std::rand()) / RAND_MAX;
        if (x * x + y * y <= 1.0) {
            local_inside++;
        }
    }

    // Reduce all local counts to get the total count in the circle
    long int total_inside = 0;
    MPI_Reduce(&local_inside, &total_inside, 1, MPI_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

    // Only rank 0 calculates and returns pi
    double pi = 0.0;
    if (rank == 0) {
        pi = 4.0 * static_cast<double>(total_inside) / static_cast<double>(n);
    }

    return pi;
}

#endif // PI_H

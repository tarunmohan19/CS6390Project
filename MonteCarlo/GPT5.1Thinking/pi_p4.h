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

    // Divide work as evenly as possible among ranks
    long int base = n / size;
    long int rem  = n % size;
    long int local_n = base + (rank < rem ? 1 : 0);

    // Seed RNG differently on each rank
    unsigned int seed = static_cast<unsigned int>(time(nullptr)) + rank * 1337u;
    srand(seed);

    long long local_hits = 0;

    for (long int i = 0; i < local_n; ++i) {
        double x = static_cast<double>(rand()) / RAND_MAX;
        double y = static_cast<double>(rand()) / RAND_MAX;

        if (x * x + y * y <= 1.0) {
            ++local_hits;
        }
    }

    // Reduce all local hit counts to rank 0
    long long global_hits = 0;
    MPI_Reduce(&local_hits, &global_hits, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

    // Rank 0 computes the final estimate of pi
    double pi_est = 0.0;
    if (rank == 0) {
        pi_est = 4.0 * static_cast<double>(global_hits) / static_cast<double>(n);
    }

    // Broadcast pi to all ranks so everyone returns the same value
    MPI_Bcast(&pi_est, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    return pi_est;
}

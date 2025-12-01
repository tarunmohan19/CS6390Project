#include <stdio.h>
#include <ctime>
#include <cstring>
#include <sstream>
#include <iomanip>
#include <iostream>
#include <random>   // <--- add this
#include <mpi.h>

double pi_calc(long int n) {

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Divide work among ranks, handling remainder
    long int base = n / size;
    long int rem  = n % size;
    long int local_n = base + (rank < rem ? 1 : 0);

    // Seed a fast RNG differently on each rank
    unsigned long seed =
        static_cast<unsigned long>(time(nullptr)) ^ (static_cast<unsigned long>(rank) << 16);
    std::mt19937_64 gen(seed);
    std::uniform_real_distribution<double> dist(0.0, 1.0);

    // Monte Carlo: count points inside quarter circle
    long long local_count = 0;
    for (long int i = 0; i < local_n; ++i) {
        double x = dist(gen);
        double y = dist(gen);
        if (x * x + y * y <= 1.0) {
            ++local_count;
        }
    }

    // Reduce local counts to a global count on rank 0
    long long global_count = 0;
    MPI_Reduce(&local_count, &global_count, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

    // Rank 0 computes the estimate of pi
    double pi_est = 0.0;
    if (rank == 0) {
        pi_est = 4.0 * static_cast<double>(global_count) / static_cast<double>(n);
    }

    // Broadcast pi_est so all ranks return the same value (safe & clean)
    MPI_Bcast(&pi_est, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    return pi_est;
}

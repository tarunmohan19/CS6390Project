#include <stdio.h>
#include <ctime>
#include <cstring>
#include <sstream>
#include <iomanip>
#include <iostream>
#include <random>   // <-- add this
#include <mpi.h>


double pi_calc(long int n) {

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Handle degenerate case
    if (n <= 0) {
        return 0.0;
    }

    // Divide work roughly evenly among ranks
    long long total_points = static_cast<long long>(n);
    long long base_points  = total_points / size;
    long long remainder    = total_points % size;

    // First 'remainder' ranks get one extra point
    long long local_points = base_points + (rank < remainder ? 1 : 0);

    // Seed RNG uniquely per rank
    unsigned long seed =
        static_cast<unsigned long>(std::time(nullptr)) +
        static_cast<unsigned long>(rank) * 1234567UL;

    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> dist(0.0, 1.0);

    // Monte Carlo: count points inside quarter circle
    long long local_in_circle = 0;
    for (long long i = 0; i < local_points; ++i) {
        double x = dist(rng);
        double y = dist(rng);
        if (x * x + y * y <= 1.0) {
            ++local_in_circle;
        }
    }

    // Sum counts across all processes
    long long global_in_circle = 0;
    MPI_Allreduce(&local_in_circle, &global_in_circle, 1,
                  MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);

    // Compute final estimate of pi
    double pi_est = 4.0 * static_cast<double>(global_in_circle)
                          / static_cast<double>(total_points);

    return pi_est;
}

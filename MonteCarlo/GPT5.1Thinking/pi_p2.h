#include <stdio.h>
#include <ctime>
#include <cstring>
#include <sstream>
#include <iomanip>
#include <iostream>
#include <random>   // <-- add this
#include <mpi.h>

double pi_calc(long int n) {

    // Handle degenerate case
    if (n <= 0) {
        return 0.0;
    }

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Distribute points as evenly as possible among ranks
    long long base = n / size;
    long long rem  = n % size;

    long long local_n = base + (rank < rem ? 1 : 0);

    // Deterministic, per-rank RNG seed for reproducibility
    // Same (n, size, rank) => same random sequence.
    std::uint64_t seed =
        0x9e3779b97f4a7c15ULL ^                        // arbitrary constant
        (static_cast<std::uint64_t>(rank) * 0x1000003ULL) ^
        static_cast<std::uint64_t>(n);

    std::mt19937_64 gen(seed);
    std::uniform_real_distribution<double> dist(0.0, 1.0);

    // Local Monte Carlo sampling
    long long local_hits = 0;
    for (long long i = 0; i < local_n; ++i) {
        double x = dist(gen);
        double y = dist(gen);
        if (x * x + y * y <= 1.0) {
            ++local_hits;
        }
    }

    // Reduce hit counts to rank 0
    long long global_hits = 0;
    MPI_Reduce(&local_hits, &global_hits, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

    // Compute and broadcast π estimate
    double pi_est = 0.0;
    if (rank == 0) {
        pi_est = 4.0 * static_cast<double>(global_hits) / static_cast<double>(n);
    }

    // Make result available on all ranks (harmless overhead, useful for generality)
    MPI_Bcast(&pi_est, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    return pi_est;
}

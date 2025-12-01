// pi.h
#pragma once

#include <stdio.h>
#include <ctime>
#include <cstring>
#include <sstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <mpi.h>

inline double pi_calc(long int n) {

    // Handle non-positive input safely
    if (n <= 0) {
        return 0.0;
    }

    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Distribute work roughly evenly across ranks
    long long total_samples = static_cast<long long>(n);
    long long base = total_samples / size;
    long long remainder = total_samples % size;
    long long local_samples = base + (rank < remainder ? 1 : 0);

    // Thread-safe RNG: local to this function / rank
    // Use a seed sequence mixing time, rank, and n
    std::seed_seq seed{
        static_cast<unsigned int>(std::time(nullptr)),
        static_cast<unsigned int>(rank),
        static_cast<unsigned int>(total_samples & 0xffffffffULL),
        static_cast<unsigned int>((total_samples >> 32) & 0xffffffffULL)
    };
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> dist(0.0, 1.0);

    // Local Monte Carlo loop
    long long local_hits = 0;
    for (long long i = 0; i < local_samples; ++i) {
        double x = dist(rng);
        double y = dist(rng);
        double r2 = x * x + y * y;
        if (r2 <= 1.0) {
            ++local_hits;
        }
    }

    // Reduce hits and samples across all ranks
    long long global_hits = 0;
    long long global_samples = 0;

    MPI_Allreduce(&local_hits, &global_hits, 1, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&local_samples, &global_samples, 1, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);

    long double pi_est = 0.0L;
    if (global_samples > 0) {
        pi_est = 4.0L * static_cast<long double>(global_hits) /
                 static_cast<long double>(global_samples);
    }

    return static_cast<double>(pi_est);
}

#ifndef PI_H
#define PI_H

#include <stdio.h>
#include <ctime>
#include <cstring>
#include <sstream>
#include <iomanip>
#include <iostream>
#include <mpi.h>
#include <random> // Required for high-quality RNG

/**
 * Calculates Pi using a Monte Carlo simulation distributed via MPI.
 * * @param n Total number of points to simulate across all processes.
 * @return The estimated value of Pi (only valid on Rank 0).
 */
double pi_calc(long int n) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // 1. Load Balancing
    // Divide the total work (n) among the processes.
    // If n is not divisible by size, distribute the remainder among the first few ranks.
    long int local_n = n / size;
    long int remainder = n % size;

    if (rank < remainder) {
        local_n++;
    }

    // 2. Random Number Generation Setup
    // We use std::mt19937 (Mersenne Twister) for better statistical properties than rand().
    // Seeding: 
    // - Use time(NULL) so runs at different times produce different results.
    // - Add (rank * constant) to ensure every process has a unique seed and non-overlapping sequence start.
    unsigned int seed = (unsigned int)time(NULL) + (unsigned int)(rank * 10000);
    std::mt19937 gen(seed);
    std::uniform_real_distribution<double> dis(0.0, 1.0);

    // 3. Monte Carlo Simulation
    // Generate local_n points and count how many fall within the unit circle.
    long int local_hits = 0;
    double x, y;

    for (long int i = 0; i < local_n; ++i) {
        x = dis(gen);
        y = dis(gen);
        
        // Check if point is inside unit circle (x^2 + y^2 <= 1)
        if ((x * x + y * y) <= 1.0) {
            local_hits++;
        }
    }

    // 4. Reduction
    // Sum all local_hits to the root process (Rank 0)
    long int global_hits = 0;
    MPI_Reduce(&local_hits, &global_hits, 1, MPI_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

    // 5. Final Calculation
    // Only Rank 0 computes the final estimation.
    double pi_estimate = 0.0;
    if (rank == 0) {
        // Pi = 4 * (Points inside circle / Total points)
        pi_estimate = 4.0 * (double)global_hits / (double)n;
    }

    return pi_estimate;
}

#endif
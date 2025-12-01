#ifndef PI_H
#define PI_H

#include <stdio.h>
#include <ctime>
#include <cstring>
#include <sstream>
#include <iomanip>
#include <iostream>
#include <random> // Added for modern C++ RNG
#include <mpi.h>

/**
 * Estimates Pi using a Monte Carlo simulation across multiple MPI processes.
 * * @param n The total number of random points to generate across all processes.
 * @return The estimated value of Pi (returned only on Rank 0, 0.0 on others).
 */
double pi_calc(long int n) {
    int rank, size;
    
    // Get current process rank and total number of processes
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Divide the work among processes
    // local_n is the number of points this specific process will simulate
    long int local_n = n / size;

    // Handle the remainder: Rank 0 takes the extra points
    // This ensures total samples == n exactly
    if (rank == 0) {
        local_n += n % size;
    }

    // Initialize Modern Random Number Generator
    // We combine random_device with rank to ensure every process 
    // has a unique seed, preventing identical number sequences.
    std::random_device rd;
    std::mt19937_64 gen(rd() + rank); 
    std::uniform_real_distribution<double> dist(0.0, 1.0);

    long int local_hits = 0;

    // Simulation Loop
    for (long int i = 0; i < local_n; ++i) {
        // Generate random point (x, y) in unit square [0,1] x [0,1]
        double x = dist(gen);
        double y = dist(gen);

        // Check if point is inside unit circle: x^2 + y^2 <= 1
        // We compare squared values to avoid the expensive sqrt() function
        if ((x * x + y * y) <= 1.0) {
            local_hits++;
        }
    }

    // Reduce all local_hits to a single global count on Rank 0
    long int global_hits = 0;
    
    // MPI_Reduce(send_buf, recv_buf, count, datatype, op, root, comm)
    MPI_Reduce(&local_hits, &global_hits, 1, MPI_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

    // Calculate Pi on Rank 0
    if (rank == 0) {
        // Pi approx = 4 * (points inside circle / total points)
        double pi_estimate = 4.0 * (static_cast<double>(global_hits) / static_cast<double>(n));
        return pi_estimate;
    }

    // Non-root ranks return 0.0 (or could return the estimate if MPI_Allreduce was used)
    return 0.0;
}

#endif // PI_H
#ifndef PI_H
#define PI_H

#include <mpi.h>
#include <random>
#include <iostream>

/**
 * Calculates Pi using a Monte Carlo simulation distributed across MPI processes.
 * * @param n The total number of points to simulate (across all processes).
 * @return The estimated value of Pi (valid on Rank 0, 0.0 on others).
 */
double pi_calc(long int n) {
    int rank, size;
    
    // Get MPI rank and size
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Calculate the number of points for this specific process
    // Note: We use integer division here. If n is not perfectly divisible by size,
    // we effectively simulate (local_n * size) points total.
    long int local_n = n / size;

    // Initialize Random Number Generator
    // Critical: Seed must be different for each process!
    // We combine a random device with the rank to ensure unique sequences.
    std::random_device rd;
    std::mt19937 gen(rd() + rank); 
    std::uniform_real_distribution<double> dist(0.0, 1.0);

    long int local_count = 0;

    // Monte Carlo Simulation Loop
    for (long int i = 0; i < local_n; ++i) {
        // Generate random (x, y) coordinates between 0.0 and 1.0
        double x = dist(gen);
        double y = dist(gen);

        // Check if the point is inside the unit circle
        // x^2 + y^2 <= r^2 (where r=1)
        if ((x * x) + (y * y) <= 1.0) {
            local_count++;
        }
    }

    // Reduce all local counts to the global count on Rank 0
    long int global_count = 0;
    
    // MPI_Reduce arguments: send_buf, recv_buf, count, datatype, operation, root, communicator
    MPI_Reduce(&local_count, &global_count, 1, MPI_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

    double pi_estimate = 0.0;

    if (rank == 0) {
        // Calculate Pi: 4 * (points_in_circle / total_points)
        // We use (local_n * size) as the denominator to account for any truncation 
        // that occurred during the integer division of n / size.
        long int total_points_simulated = local_n * size;
        
        if (total_points_simulated > 0) {
            pi_estimate = 4.0 * ((double)global_count / (double)total_points_simulated);
        }
    }

    return pi_estimate;
}

#endif // PI_H
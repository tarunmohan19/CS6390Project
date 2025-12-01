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

    // Divide points among processes
    long int points_per_proc = n / size;
    long int remainder = n % size;
    if (rank < remainder) {
        points_per_proc += 1; // distribute remainder
    }

    // Initialize thread-safe random generator
    unsigned int seed = static_cast<unsigned int>(time(nullptr)) + rank;
    long int local_hits = 0;

    for (long int i = 0; i < points_per_proc; ++i) {
        // Generate random (x, y) in [0,1)
        double x = static_cast<double>(rand_r(&seed)) / RAND_MAX;
        double y = static_cast<double>(rand_r(&seed)) / RAND_MAX;

        if (x * x + y * y <= 1.0) {
            local_hits++;
        }
    }

    // Reduce hits from all processes to rank 0
    long int total_hits = 0;
    MPI_Reduce(&local_hits, &total_hits, 1, MPI_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

    // Rank 0 calculates and returns π
    if (rank == 0) {
        return 4.0 * static_cast<double>(total_hits) / static_cast<double>(n);
    } else {
        return 0.0; // Other ranks return dummy value
    }
}

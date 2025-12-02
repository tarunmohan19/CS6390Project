#include <mpi.h>
#include <random>
#include <ctime>

double pi_calc(long int n) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // 1. Work Distribution
    // Calculate how many iterations this specific process (rank) handles.
    // We handle the remainder (n % size) by giving one extra point 
    // to the first few ranks to ensure exactly 'n' points are calculated.
    long int local_n = n / size;
    if (rank < n % size) {
        local_n++;
    }

    // 2. Random Number Generation
    // CRITICAL: Seed the generator uniquely for every rank. 
    // If we only use time(0), all nodes might generate the exact same "random" sequence.
    std::mt19937 gen(time(NULL) + (rank * 100000)); 
    std::uniform_real_distribution<double> dis(0.0, 1.0);

    // 3. Monte Carlo Simulation
    long int local_count = 0;
    double x, y;

    for (long int i = 0; i < local_n; ++i) {
        x = dis(gen);
        y = dis(gen);
        
        // Check if point is inside the unit circle (x^2 + y^2 <= 1)
        if (x * x + y * y <= 1.0) {
            local_count++;
        }
    }

    // 4. Reduction
    // Sum all local_counts into a global_count variable on Rank 0
    long int global_count = 0;
    MPI_Reduce(&local_count, &global_count, 1, MPI_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

    // 5. Final Calculation
    // Only Rank 0 needs to compute the return value.
    if (rank == 0) {
        return 4.0 * (double)global_count / (double)n;
    }

    // Non-root ranks return 0.0 (ignored by main)
    return 0.0;
}

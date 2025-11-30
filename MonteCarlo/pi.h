#include <stdio.h>
#include <ctime>
#include <cstring>
#include <sstream>
#include <iomanip>
#include <iostream>
#include <mpi.h>


double pi_calc(long int n) {
    
    // Write your code below
    ////////////////////////////////////////
    // there will be n random points
    // pi =  4 * (points in the circle / total points)

    // Each process will consume n/p points and then randomly plot them and then increment points in the circle
    // Total points in equal to n
    // We will reduce all processes points in the circle to rank 0 and then calculate pi


    ////////////////////////////////////////
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int p_in_cir = 0;
    int points_per_process = n / size;

    srand(time(NULL) + rank);

    for(int i = 0; i < points_per_process; i++) {
        double x = (double)rand() / RAND_MAX;
        double y = (double)rand() / RAND_MAX;
        if(x * x + y * y <= 1) {
            p_in_cir++;
        }
    }

    int total_p_in_cir = 0;
    MPI_Reduce(&p_in_cir, &total_p_in_cir, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    double pi = 4 * (double)total_p_in_cir / n;

    return pi;
}

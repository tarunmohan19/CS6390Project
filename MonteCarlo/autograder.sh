#!/bin/bash

#SBATCH --job-name=results
#SBATCH --output=results.out
#SBATCH --error=results.err
#SBATCH --nodes=1
#SBATCH --constraint=gold6226
#SBATCH --ntasks-per-node=24
#SBATCH --time=00:05:00
#SBATCH --exclusive

# Maximum time allowed for the script (in seconds)
MAX_EXECUTION_TIME=180

# Run the tests within the time limit
timeout $MAX_EXECUTION_TIME bash -c '

lscpu
echo "-------------------------------------"

# Run the tests
module load openmpi

make clean
make

# Correctness tests
echo "-------------------------------------"
echo "Correctness test"
echo "n = 1000000"
echo "# of processors: 1 and 16"
output1=$(mpirun -np 1 ./pi -n 1000000)
estimated_pi1=$(echo "$output1" | grep "Estimated Pi" | awk "{print \$3}")
output5=$(mpirun -np 16 ./pi -n 1000000)
estimated_pi2=$(echo "$output5" | grep "Estimated Pi" | awk "{print \$3}")

min_pi=3.13
max_pi=3.15

    if (( $(echo "$estimated_pi1 < $min_pi" | bc -l) )) || (( $(echo "$estimated_pi1 > $max_pi" | bc -l) )) || (( $(echo "$estimated_pi2 < $min_pi" | bc -l) )) || (( $(echo "$estimated_pi2 > $max_pi" | bc -l) )); then
        echo "FAILED: Estimated Pi out of range [$min_pi, $max_pi]"
	exit 1
    else
        echo "PASSED: Estimated Pi within range [$min_pi, $max_pi]"
    fi


echo "-------------------------------------"

# RunTime test
echo "RunTime test"
echo "n = 1000000000"
echo "# of processors: 8"
output2=$(mpirun -np 8 ./pi -n 1000000000)
time1=$(echo "$output2" | grep "Time" | awk "{print \$2}")
max_time=5.5
if (( $(echo "$time1 > $max_time" | bc -l) )); then
        echo "FAILED: Time ($time1 seconds) exceeds $max_time seconds"
    else
        echo "PASSED: Time ($time1 seconds) within $max_time seconds"
    fi

echo "-------------------------------------"

# Scalability tests
echo "Scalability test"
echo "n = 1000000000"
echo "# of processors: 16"
output4=$(mpirun -np 1 ./pi -n 1000000000)
time4=$(echo "$output4" | grep "Time" | awk "{print \$2}")

output3=$(mpirun -np 16 ./pi -n 1000000000)
time3=$(echo "$output3" | grep "Time" | awk "{print \$2}")
speedup=$(echo "$time4/$time3" | bc -l)
speedup_ref=15.5
if (( $(echo "$speedup < $speedup_ref" | bc -l) )); then
    echo "FAILED: Speedup ($speedup) should be at least $speedup_ref for 16 processors"
else
    echo "PASSED: Speedup ($speedup) is at least $speedup_ref for 16 processors"
fi
'

if [[ $? -eq 124 ]]; then
    echo "Error: Script execution exceeded $MAX_EXECUTION_TIME seconds."
    exit 1
fi

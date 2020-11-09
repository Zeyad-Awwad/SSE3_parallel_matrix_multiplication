# Parallel Matrix Multiplication with SSE3

The software in this repository uses CPU parallelism (SSE3) to perform matrix multiplication using 14 CPU registers with a 2x4 kernel and loop unrolling. The final performance exceeded 3.5 gflops/s, compared to only 0.24 gflops/s for the starter code.

The primary software is developed in C using the SSE3 library. A few external autotuning functions were implemented as Python scripts to iteratively edit certain parameters and run the C code. These were implemented for testing purposes and are not required to run the software. 

The benchmarking program can be called from the command line

./benchmark-blocked -n [N] [path]

Where [N] is the matrix size and the [path] is the working directory

The code was collaboratively developed with my project partner, Erin Cummings, as part of a parallel computation course. It was built upon a starter code shared by Prof Bryan Chin, which was originally developed by Prof Jim Demmel and updated by Prof Scott Baden.
http://www.cs.berkeley.edu/~knight/cs267/hw1.html
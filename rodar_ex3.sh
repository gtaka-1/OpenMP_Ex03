gcc -O2 -fopenmp ex3_matmul.c -o ex3
N=2048
./ex3 $N seq 1
for t in 1 2 3 4 5 6; do ./ex3 $N par $t; done

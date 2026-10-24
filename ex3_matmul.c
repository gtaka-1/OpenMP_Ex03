/* Nome: Gabriel Goes Taka - RA: 10723533
   Ex.3 - Multiplicacao de matrizes C = A x B (double, N x N)*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>

#define BS 32

/* Versao sequencial (com tiling) */
void matmul_seq(const double *A, const double *B, double *C, int N) {
    for (int ii = 0; ii < N; ii += BS)
     for (int kk = 0; kk < N; kk += BS)
      for (int jj = 0; jj < N; jj += BS)
       for (int i = ii; i < ii + BS; i++)
        for (int k = kk; k < kk + BS; k++) {
            double a = A[i * N + k];
            for (int j = jj; j < jj + BS; j++)
                C[i * N + j] += a * B[k * N + j];
        }
}

/* Versao paralela: mesmo algoritmo, blocos de linhas (ii) divididos entre threads */
void matmul_par(const double *A, const double *B, double *C, int N) {
    #pragma omp parallel for schedule(static)
    for (int ii = 0; ii < N; ii += BS)
     for (int kk = 0; kk < N; kk += BS)
      for (int jj = 0; jj < N; jj += BS)
       for (int i = ii; i < ii + BS; i++)
        for (int k = kk; k < kk + BS; k++) {
            double a = A[i * N + k];
            for (int j = jj; j < jj + BS; j++)
                C[i * N + j] += a * B[k * N + j];
        }
}

int main(int argc, char **argv) {
    int N = atoi(argv[1]); int par = strcmp(argv[2], "par") == 0;
    int nt = par ? atoi(argv[3]) : 1;
    omp_set_num_threads(nt);                 /* N deve ser multiplo de BS */
    double *A = malloc((size_t)N * N * sizeof(double));
    double *B = malloc((size_t)N * N * sizeof(double));
    double *C = malloc((size_t)N * N * sizeof(double));
    for (long i = 0; i < (long)N * N; i++) { A[i] = i % 10; B[i] = (i * 7) % 10; }

    double total = 0, check = 0;
    for (int e = 0; e < 3; e++) {
        memset(C, 0, (size_t)N * N * sizeof(double));
        double t = omp_get_wtime();
        if (par) matmul_par(A, B, C, N); else matmul_seq(A, B, C, N);
        total += omp_get_wtime() - t;
    }
    for (long i = 0; i < (long)N * N; i++) check += C[i];
    printf("%s N=%d threads=%d checksum=%.0f tempo_medio=%.6f s\n", par ? "PARALELA  " : "SEQUENCIAL", N, nt, check, total / 3);
    free(A); free(B); free(C); return 0;
}

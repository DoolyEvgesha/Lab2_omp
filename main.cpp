#include <stdio.h>
#include <omp.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>

double h;
double EPS;
int A;
int N;
int ITER_MAX;
int GRID_POW;

// Equation B4
static inline double f(double y) {
    return A * (y * y * y - y);
}

// it is a derivative
static inline double g_derivative(double y) {
    return A * (3 * y * y - 1);
}

static inline double numerov(double y_left, double y, double y_right) {
    return (f(y_left) + 10. * f(y) + f(y_right)) / 12.;
}

static double diff(const double *y1, const double *y2) {
    assert(y1);
    assert(y2);

    double res = 0;
#pragma omp parallel for reduction(+:res)
    for (int i = 0; i < N + 1; i++)
        res += fabs(y1[i] - y2[i]);
    return res;
}

// Method
static void forward(double *a, double *b, double *c, double *F,
                    int i, int start, int step);

static void backward(double *a, double *b, double *c, double *F, double *y,
                     int i, int start, int step);

static void cycle_reduction(double *a, double *b, double *c,
                            double *F, double *y /* out */);

// Initialization
static inline void set_beta(double *b, const double *y) {
    assert(b);
    assert(y);

#pragma omp parallel for
    for (int i = 1; i < N; i++) {
        b[i] = -2 - g_derivative(y[i]) * h * h;
    }
}

static inline void set_F(double *F, const double *y) {
    assert(F);
    assert(y);

#pragma omp parallel for
    for (int i = 1; i < N; i++) {
        F[i] = (numerov(y[i - 1], y[i], y[i + 1]) - y[i] * g_derivative(y[i])) * h * h;
    }
}

static inline void fillArray(double *arr, double val) {
    assert(arr);

#pragma omp parallel for
    for (int i = 0; i < N + 1; i++) {
        arr[i] = val;
    }
}

void scanFileParameters() {
    FILE *in_file = fopen("input.txt", "r");
    if (!in_file) {
        printf("Couldn't open the file\n");
        exit(EXIT_FAILURE);
    }
    fscanf(in_file, "%lg %lg %d %d %d", &h, &EPS, &A, &N, &ITER_MAX);
    GRID_POW = log2(N);
    printf("%lg %lg %d %d \n", h, EPS, A, N, ITER_MAX);
}

int main() {
    // change number of threads here
    omp_set_num_threads(8);
    scanFileParameters();
    auto *alpha = (double *) malloc((N + 1) * sizeof(double));
    auto *beta = (double *) malloc((N + 1) * sizeof(double));
    auto *gamma = (double *) malloc((N + 1) * sizeof(double));
    auto *y = (double *) malloc((N + 1) * sizeof(double));
    auto *F = (double *) malloc((N + 1) * sizeof(double));
    auto *next = (double *) malloc((N + 1) * sizeof(double));
    // To be vectorized
    fillArray(y, 0.1);
    fillArray(next, 0.1);

    int n_iters = 0;
    double res = 1.;

    double startTime = omp_get_wtime();
    while (res > EPS && n_iters++ < ITER_MAX) {
        set_beta(beta, y);
        set_F(F, y);
        fillArray(alpha, 1);
        fillArray(gamma, 1);

        cycle_reduction(alpha, beta, gamma, F, next);
        // граничные условия
        next[0] = sqrt(2);
        next[N] = sqrt(2);
        res = diff(y, next);

        double *swap = next;
        next = y;
        y = swap;
        // y contains answer on exit
    }
    double endTime = omp_get_wtime();
    printf("Time: %lg\n", endTime - startTime);

    char name[64] = {};
    sprintf(name, "../%d.csv", A);
    FILE *file = fopen(name, "w");
    if (!file) {
        perror("fopen");
        return EXIT_FAILURE;
    }
    for (int i = 0; i < N + 1; i++) {
        fprintf(file, "%lf, %lf\n ", (double) i / (N + 1) * 20 - 10, y[i]);
    }
//    fwrite(y, sizeof(*y), N + 1, file);
    fclose(file);

    free(alpha);
    free(beta);
    free(gamma);
    free(y);
    free(F);
    free(next);
    return 0;
}

// =========================================================

void forward(double *a, double *b, double *c, double *F,
             int i, int start, int step) {
    assert(a);
    assert(b);
    assert(c);
    assert(F);
    assert(i >= 0);
    assert(start >= 0);
    assert(step >= 0);

    int k = start * (i + 1);

    double alpha = -(a[k] * a[k - step]) / b[k - step];
    double gamma = -(c[k] * c[k + step]) / b[k + step];
    double beta = b[k]
                  - c[k - step] * a[k] / b[k - step]
                  - c[k] * a[k + step] / b[k + step];
    double f = F[k]
               - a[k] * F[k - step] / b[k - step]
               - c[k] * F[k + step] / b[k + step];

    a[k] = alpha;
    b[k] = beta;
    c[k] = gamma;
    F[k] = f;
}

void backward(double *a, double *b, double *c, double *F, double *y,
              int i, int start, int step) {
    assert(a);
    assert(b);
    assert(c);
    assert(F);
    assert(y);
    assert(i >= 0);
    assert(start >= 0);
    assert(step >= 0);

    int k = start * (2 * i + 1);
    double out = F[k] - c[k] * y[k + step] - a[k] * y[k - step];

    y[k] = out / b[k];
}

void cycle_reduction(double *a, double *b, double *c,
                     double *F, double *y /* out */) {
    assert(a);
    assert(b);
    assert(c);
    assert(F);
    assert(y);

    // Forward
    int start = 2;
    int step = 1;
    int nelem = N - 1;
    for (int p = 0; p < GRID_POW; p++) {
        nelem = (nelem - 1) / 2;
#pragma omp parallel for
        for (int i = 0; i < nelem; i++)
            forward(a, b, c, F, i, start, step);
        start <<= 1;
        step <<= 1;
    }

    // Backward
    start = N / 2;
    step = N / 2;
    nelem = 1;
    for (int p = 0; p < GRID_POW; p++) {
#pragma omp parallel for
        for (int i = 0; i < nelem; i++)
            backward(a, b, c, F, y, i, start, step);
        start >>= 1;
        step >>= 1;
        nelem <<= 1;
    }
}


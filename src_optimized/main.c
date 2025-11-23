/* OPTIMIZED VERSION - Hash: 28f428231572313c2e38491601c53bdfa6f09e745709c2644909cf3202c2a8ad */

#include "matrix.h"
#include "stats.h"
#include "utils.h"
#include "vector.h"
#include <stdio.h>
#include <stdlib.h>

/* Mark demo functions as static inline to allow compiler to optimize/inline them */
static void demo_matrix_operations(void)
{
    printf("=== Matrix Operations Demo ===\n\n");

    Matrix *a = matrix_create(3, 3);
    Matrix *b = matrix_create(3, 3);

    matrix_fill_random(a);
    matrix_fill_random(b);

    printf("Matrix A:\n");
    matrix_print(a);

    printf("\nMatrix B:\n");
    matrix_print(b);

    Matrix *sum = matrix_add(a, b);
    printf("\nA + B:\n");
    matrix_print(sum);

    Matrix *product = matrix_multiply(a, b);
    printf("\nA * B:\n");
    matrix_print(product);

    Matrix *transposed = matrix_transpose(a);
    printf("\nTranspose of A:\n");
    matrix_print(transposed);

    /* Free resources */
    matrix_free(a);
    matrix_free(b);
    matrix_free(sum);
    matrix_free(product);
    matrix_free(transposed);
}

static void demo_vector_operations(void)
{
    printf("\n=== Vector Operations Demo ===\n\n");

    Vector *v1 = vector_create(5);
    Vector *v2 = vector_create(5);

    vector_fill_random(v1);
    vector_fill_random(v2);

    printf("Vector 1: ");
    vector_print(v1);

    printf("Vector 2: ");
    vector_print(v2);

    const double dot = vector_dot(v1, v2);
    printf("\nDot product: %.3f\n", dot);

    Vector *sum = vector_add(v1, v2);
    printf("Sum: ");
    vector_print(sum);

    const double mag = vector_magnitude(v1);
    printf("Magnitude of v1: %.3f\n", mag);

    Vector *normalized = vector_normalize(v1);
    printf("Normalized v1: ");
    vector_print(normalized);

    /* Free resources */
    vector_free(v1);
    vector_free(v2);
    vector_free(sum);
    vector_free(normalized);
}

static void demo_statistics(void)
{
    printf("\n=== Statistics Demo ===\n\n");

    const size_t n = 100;
    double *data = (double *) malloc(n * sizeof(double));

    /* Generate random data */
    for (size_t i = 0; i < n; i++) {
        data[i] = utils_random_double(0.0, 100.0);
    }

    /* Compute statistics - const for values that won't change */
    const double mean = stats_mean(data, n);
    const double variance = stats_variance(data, n);
    const double stddev = stats_stddev(data, n);
    const double min = stats_min(data, n);
    const double max = stats_max(data, n);
    const double median = stats_median(data, n);

    printf("Dataset statistics (n=%zu):\n", n);
    printf("  Mean:     %.3f\n", mean);
    printf("  Median:   %.3f\n", median);
    printf("  Variance: %.3f\n", variance);
    printf("  Std Dev:  %.3f\n", stddev);
    printf("  Min:      %.3f\n", min);
    printf("  Max:      %.3f\n", max);

    free(data);
}

static void benchmark_matrix_multiply(void)
{
    printf("\n=== Matrix Multiply Benchmark ===\n\n");

    /* Use const array for sizes */
    const size_t sizes[] = {50, 100, 200};
    const int num_sizes = 3;

    for (int i = 0; i < num_sizes; i++) {
        const size_t size = sizes[i];

        Matrix *a = matrix_create(size, size);
        Matrix *b = matrix_create(size, size);
        matrix_fill_random(a);
        matrix_fill_random(b);

        Timer timer;
        timer_start(&timer);

        Matrix *result = matrix_multiply(a, b);

        timer_stop(&timer);
        const double elapsed = timer_elapsed_ms(&timer);

        printf("Matrix %zux%zu multiply: %.2f ms\n", size, size, elapsed);

        /* Free resources */
        matrix_free(a);
        matrix_free(b);
        matrix_free(result);
    }
}

int main(int argc, char *argv[])
{
    /* Suppress unused parameter warnings */
    (void) argc;
    (void) argv;

    /* Initialize random number generator once */
    utils_init_random();

    printf("C AI Optimizer Demo - Optimized Version\n");
    printf("========================================\n\n");

    /* Run demo functions */
    demo_matrix_operations();
    demo_vector_operations();
    demo_statistics();
    benchmark_matrix_multiply();

    printf("\n=== Done ===\n");

    return 0;
}

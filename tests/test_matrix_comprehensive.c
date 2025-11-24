#include "../include/matrix.h"
#include "../include/utils.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Test with large matrices to exercise AVX paths and cache behavior */
int test_large_matrix_multiply(void)
{
    const size_t N = 500; /* 500x500 = 250k elements */
    printf("  Testing large matrix multiply (%zux%zu)... ", N, N);
    fflush(stdout);

    Matrix *a = matrix_create(N, N);
    Matrix *b = matrix_create(N, N);

    if (a == NULL || b == NULL) {
        printf("[SKIP - allocation failed]\n");
        matrix_free(a);
        matrix_free(b);
        return 1; /* Skip but don't fail */
    }

    /* Fill with simple pattern for verification */
    for (size_t i = 0; i < N * N; i++) {
        a->data[i] = 1.0;
        b->data[i] = 1.0;
    }

    Matrix *result = matrix_multiply(a, b);

    int passed = 1;
    if (result != NULL) {
        /* Each element should be N (sum of N 1.0s) */
        const double expected = (double) N;
        for (size_t i = 0; i < N * N; i++) {
            if (!utils_double_equal(result->data[i], expected, 1e-6)) {
                printf("[FAIL - value mismatch at %zu: got %f, expected %f]\n", i, result->data[i],
                       expected);
                passed = 0;
                break;
            }
        }
        if (passed) {
            printf("[PASS]\n");
        }
    } else {
        printf("[FAIL - NULL result]\n");
        passed = 0;
    }

    matrix_free(a);
    matrix_free(b);
    matrix_free(result);

    return passed;
}

/* Test large matrix addition to exercise AVX and OpenMP */
int test_large_matrix_add(void)
{
    const size_t rows = 1000;
    const size_t cols = 500; /* 500k elements */
    printf("  Testing large matrix add (%zux%zu)... ", rows, cols);
    fflush(stdout);

    Matrix *a = matrix_create(rows, cols);
    Matrix *b = matrix_create(rows, cols);

    if (a == NULL || b == NULL) {
        printf("[SKIP - allocation failed]\n");
        matrix_free(a);
        matrix_free(b);
        return 1;
    }

    /* Fill with known values */
    for (size_t i = 0; i < rows * cols; i++) {
        a->data[i] = 2.5;
        b->data[i] = 3.5;
    }

    Matrix *result = matrix_add(a, b);

    int passed = 1;
    if (result != NULL) {
        const double expected = 6.0;
        for (size_t i = 0; i < rows * cols; i++) {
            if (!utils_double_equal(result->data[i], expected, 1e-10)) {
                printf("[FAIL - value mismatch]\n");
                passed = 0;
                break;
            }
        }
        if (passed) {
            printf("[PASS]\n");
        }
    } else {
        printf("[FAIL - NULL result]\n");
        passed = 0;
    }

    matrix_free(a);
    matrix_free(b);
    matrix_free(result);

    return passed;
}

/* Test aliasing: matrix_add(m, m) should work correctly */
int test_matrix_add_aliasing(void)
{
    printf("  Testing matrix_add aliasing (same pointer twice)... ");
    fflush(stdout);

    Matrix *m = matrix_create(100, 100);
    if (m == NULL) {
        printf("[SKIP - allocation failed]\n");
        return 1;
    }

    /* Fill with test values */
    for (size_t i = 0; i < 100 * 100; i++) {
        m->data[i] = 5.0;
    }

    /* This should double all values: m + m = 2*m */
    Matrix *result = matrix_add(m, m);

    int passed = 1;
    if (result != NULL) {
        const double expected = 10.0;
        for (size_t i = 0; i < 100 * 100; i++) {
            if (!utils_double_equal(result->data[i], expected, 1e-10)) {
                printf("[FAIL - aliasing broke correctness: got %f, expected %f]\n",
                       result->data[i], expected);
                passed = 0;
                break;
            }
        }
        /* Also verify original wasn't modified */
        if (passed && !utils_double_equal(m->data[0], 5.0, 1e-10)) {
            printf("[FAIL - aliasing modified input]\n");
            passed = 0;
        }
        if (passed) {
            printf("[PASS]\n");
        }
    } else {
        printf("[FAIL - NULL result]\n");
        passed = 0;
    }

    matrix_free(m);
    matrix_free(result);

    return passed;
}

/* Test in-place operations don't corrupt when aliased */
int test_matrix_scale_large(void)
{
    const size_t N = 1000 * 1000; /* 1M elements */
    printf("  Testing large matrix scale (1M elements)... ");
    fflush(stdout);

    Matrix *m = matrix_create(1000, 1000);
    if (m == NULL) {
        printf("[SKIP - allocation failed]\n");
        return 1;
    }

    /* Fill with test values */
    for (size_t i = 0; i < N; i++) {
        m->data[i] = 3.0;
    }

    matrix_scale(m, 2.0);

    int passed = 1;
    const double expected = 6.0;
    for (size_t i = 0; i < N; i++) {
        if (!utils_double_equal(m->data[i], expected, 1e-10)) {
            printf("[FAIL]\n");
            passed = 0;
            break;
        }
    }

    if (passed) {
        printf("[PASS]\n");
    }

    matrix_free(m);

    return passed;
}

/* Test non-square large matrices */
int test_large_nonsquare_multiply(void)
{
    const size_t M = 800;
    const size_t K = 600;
    const size_t N = 400;
    printf("  Testing large non-square multiply (%zux%zu * %zux%zu)... ", M, K, K, N);
    fflush(stdout);

    Matrix *a = matrix_create(M, K);
    Matrix *b = matrix_create(K, N);

    if (a == NULL || b == NULL) {
        printf("[SKIP - allocation failed]\n");
        matrix_free(a);
        matrix_free(b);
        return 1;
    }

    /* Simple test pattern */
    for (size_t i = 0; i < M * K; i++) {
        a->data[i] = 0.5;
    }
    for (size_t i = 0; i < K * N; i++) {
        b->data[i] = 0.5;
    }

    Matrix *result = matrix_multiply(a, b);

    int passed = 1;
    if (result != NULL && result->rows == M && result->cols == N) {
        /* Each element = K * (0.5 * 0.5) = K * 0.25 */
        const double expected = K * 0.25;
        for (size_t i = 0; i < M * N; i++) {
            if (!utils_double_equal(result->data[i], expected, 1e-6)) {
                printf("[FAIL - value mismatch]\n");
                passed = 0;
                break;
            }
        }
        if (passed) {
            printf("[PASS]\n");
        }
    } else {
        printf("[FAIL - wrong dimensions or NULL]\n");
        passed = 0;
    }

    matrix_free(a);
    matrix_free(b);
    matrix_free(result);

    return passed;
}

/* Test transpose with large matrix */
int test_large_transpose(void)
{
    const size_t rows = 800;
    const size_t cols = 600;
    printf("  Testing large transpose (%zux%zu)... ", rows, cols);
    fflush(stdout);

    Matrix *m = matrix_create(rows, cols);
    if (m == NULL) {
        printf("[SKIP - allocation failed]\n");
        return 1;
    }

    /* Fill with index pattern to verify correctness */
    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++) {
            m->data[i * cols + j] = i * 1000.0 + j;
        }
    }

    Matrix *t = matrix_transpose(m);

    int passed = 1;
    if (t != NULL && t->rows == cols && t->cols == rows) {
        /* Verify transpose: t[j,i] == m[i,j] */
        for (size_t i = 0; i < rows; i++) {
            for (size_t j = 0; j < cols; j++) {
                const double original = m->data[i * cols + j];
                const double transposed = t->data[j * rows + i];
                if (!utils_double_equal(original, transposed, 1e-10)) {
                    printf("[FAIL - incorrect transpose]\n");
                    passed = 0;
                    goto cleanup;
                }
            }
        }
        printf("[PASS]\n");
    } else {
        printf("[FAIL - wrong dimensions or NULL]\n");
        passed = 0;
    }

cleanup:
    matrix_free(m);
    matrix_free(t);

    return passed;
}

int run_comprehensive_matrix_tests(void)
{
    int passed = 0;
    int total = 0;

    printf("\n=== Comprehensive Matrix Tests (Large & Aliasing) ===\n");

    total++;
    if (test_large_matrix_multiply()) {
        passed++;
    }

    total++;
    if (test_large_matrix_add()) {
        passed++;
    }

    total++;
    if (test_matrix_add_aliasing()) {
        passed++;
    }

    total++;
    if (test_matrix_scale_large()) {
        passed++;
    }

    total++;
    if (test_large_nonsquare_multiply()) {
        passed++;
    }

    total++;
    if (test_large_transpose()) {
        passed++;
    }

    printf("  Comprehensive: %d/%d tests passed\n\n", passed, total);

    return passed == total;
}

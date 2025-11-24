/* OPTIMIZED VERSION - Hash: 09d67339a79f88253b7ee7ff711439e7cfde9035e5c30cf3946fd560b9abfb66 */

#include "matrix.h"
#include "utils.h"
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef __x86_64__
#include <immintrin.h>
#endif

Matrix *matrix_create(size_t rows, size_t cols)
{
    Matrix *m = (Matrix *) malloc(sizeof(Matrix));
    if (m == NULL) {
        return NULL;
    }

    m->rows = rows;
    m->cols = cols;

    /* Use aligned allocation for SIMD operations */
#ifdef __x86_64__
    m->data = (double *) aligned_alloc(32, rows * cols * sizeof(double));
#else
    m->data = (double *) malloc(rows * cols * sizeof(double));
#endif

    if (m->data == NULL) {
        free(m);
        return NULL;
    }

    /* Zero-initialize memory */
    memset(m->data, 0, rows * cols * sizeof(double));

    return m;
}

void matrix_free(Matrix *m)
{
    if (m != NULL) {
        if (m->data != NULL) {
            free(m->data);
        }
        free(m);
    }
}

Matrix *matrix_multiply(const Matrix *a, const Matrix *b)
{
    if (a == NULL || b == NULL || a->cols != b->rows) {
        return NULL;
    }

    Matrix *result = matrix_create(a->rows, b->cols);
    if (result == NULL) {
        return NULL;
    }

    /* OPTIMIZATION STRATEGY:
     * 1. OpenMP parallelization across output rows
     * 2. i-k-j loop order for better cache locality (streams through B's rows)
     * 3. AVX vectorization on the innermost j-loop
     * 4. Cache blocking to fit working set in L1/L2 cache
     *
     * NOTE: No restrict pointers - API must support aliasing cases */

    const size_t block_size = 64;

#pragma omp parallel for schedule(dynamic, 8)
    for (size_t ii = 0; ii < a->rows; ii += block_size) {
        size_t i_end = (ii + block_size < a->rows) ? ii + block_size : a->rows;

        for (size_t kk = 0; kk < a->cols; kk += block_size) {
            size_t k_end = (kk + block_size < a->cols) ? kk + block_size : a->cols;

            for (size_t i = ii; i < i_end; i++) {
                for (size_t k = kk; k < k_end; k++) {
                    const double a_ik = a->data[i * a->cols + k];
                    const double *b_row = &b->data[k * b->cols];
                    double *result_row = &result->data[i * result->cols];

                    size_t j = 0;
#ifdef __AVX__
                    /* Vectorize across B's row (contiguous in memory) */
                    const __m256d a_broadcast = _mm256_set1_pd(a_ik);

                    for (; j + 3 < b->cols; j += 4) {
                        __m256d b_vec = _mm256_loadu_pd(&b_row[j]);
                        __m256d result_vec = _mm256_loadu_pd(&result_row[j]);
                        result_vec = _mm256_fmadd_pd(a_broadcast, b_vec, result_vec);
                        _mm256_storeu_pd(&result_row[j], result_vec);
                    }
#endif
                    /* Scalar cleanup loop */
                    for (; j < b->cols; j++) {
                        result_row[j] += a_ik * b_row[j];
                    }
                }
            }
        }
    }

    return result;
}

Matrix *matrix_add(const Matrix *a, const Matrix *b)
{
    if (a == NULL || b == NULL || a->rows != b->rows || a->cols != b->cols) {
        return NULL;
    }

    Matrix *result = matrix_create(a->rows, a->cols);
    if (result == NULL) {
        return NULL;
    }

    const size_t total = a->rows * a->cols;

#pragma omp parallel
    {
        size_t i = 0;
        /* Distribute work across threads */
        const int tid = omp_get_thread_num();
        const int nthreads = omp_get_num_threads();
        const size_t chunk = total / nthreads;
        i = tid * chunk;
        const size_t i_end = (tid == nthreads - 1) ? total : (i + chunk);

#ifdef __AVX__
        /* AVX vectorized addition - processes 4 doubles at once */
        for (; i + 3 < i_end; i += 4) {
            __m256d a_vec = _mm256_loadu_pd(&a->data[i]);
            __m256d b_vec = _mm256_loadu_pd(&b->data[i]);
            __m256d result_vec = _mm256_add_pd(a_vec, b_vec);
            _mm256_storeu_pd(&result->data[i], result_vec);
        }
#endif
        /* Scalar cleanup */
        for (; i < i_end; i++) {
            result->data[i] = a->data[i] + b->data[i];
        }
    }

    return result;
}

Matrix *matrix_transpose(const Matrix *m)
{
    if (m == NULL) {
        return NULL;
    }

    Matrix *result = matrix_create(m->cols, m->rows);
    if (result == NULL) {
        return NULL;
    }

    /* Cache-blocked transpose for better memory locality */
    const size_t block_size = 32;

    for (size_t ii = 0; ii < m->rows; ii += block_size) {
        size_t i_end = (ii + block_size < m->rows) ? ii + block_size : m->rows;

        for (size_t jj = 0; jj < m->cols; jj += block_size) {
            size_t j_end = (jj + block_size < m->cols) ? jj + block_size : m->cols;

            for (size_t i = ii; i < i_end; i++) {
                for (size_t j = jj; j < j_end; j++) {
                    result->data[j * result->cols + i] = m->data[i * m->cols + j];
                }
            }
        }
    }

    return result;
}

void matrix_scale(Matrix *m, double scalar)
{
    if (m == NULL) {
        return;
    }

    const size_t total = m->rows * m->cols;

#pragma omp parallel
    {
        size_t i = 0;
        const int tid = omp_get_thread_num();
        const int nthreads = omp_get_num_threads();
        const size_t chunk = total / nthreads;
        i = tid * chunk;
        const size_t i_end = (tid == nthreads - 1) ? total : (i + chunk);

#ifdef __AVX__
        /* AVX vectorized scaling */
        const __m256d scalar_vec = _mm256_set1_pd(scalar);
        for (; i + 3 < i_end; i += 4) {
            __m256d data_vec = _mm256_loadu_pd(&m->data[i]);
            __m256d result_vec = _mm256_mul_pd(data_vec, scalar_vec);
            _mm256_storeu_pd(&m->data[i], result_vec);
        }
#endif
        /* Scalar cleanup */
        for (; i < i_end; i++) {
            m->data[i] *= scalar;
        }
    }
}

void matrix_print(const Matrix *m)
{
    if (m == NULL) {
        printf("NULL matrix\n");
        return;
    }

    printf("Matrix (%zu x %zu):\n", m->rows, m->cols);
    for (size_t i = 0; i < m->rows; i++) {
        for (size_t j = 0; j < m->cols; j++) {
            printf("%8.3f ", m->data[i * m->cols + j]);
        }
        printf("\n");
    }
}

void matrix_fill_random(Matrix *m)
{
    if (m == NULL) {
        return;
    }

    const size_t total = m->rows * m->cols;
    /* Sequential access pattern for better cache performance */
    for (size_t i = 0; i < total; i++) {
        m->data[i] = utils_random_double(-10.0, 10.0);
    }
}

int matrix_equal(const Matrix *a, const Matrix *b, double epsilon)
{
    if (a == NULL || b == NULL) {
        return 0;
    }

    if (a->rows != b->rows || a->cols != b->cols) {
        return 0;
    }

    const size_t total = a->rows * a->cols;
    /* Early exit optimization - fail fast on first mismatch */
    for (size_t i = 0; i < total; i++) {
        if (!utils_double_equal(a->data[i], b->data[i], epsilon)) {
            return 0;
        }
    }

    return 1;
}

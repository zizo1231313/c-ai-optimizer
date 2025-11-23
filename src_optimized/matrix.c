/* OPTIMIZED VERSION - Hash: 649b7d34881950df5feb1560621cf1b3369bd6609c7eb26942a6137c6d61e582 */

#include "matrix.h"
#include "utils.h"
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

    /* Cache-blocking matrix multiplication with SIMD optimization
     * This implementation improves cache locality and uses AVX for vectorization */
    const size_t block_size = 64; /* Tune based on cache size */

    for (size_t i = 0; i < a->rows; i++) {
        for (size_t jj = 0; jj < b->cols; jj += block_size) {
            size_t j_end = (jj + block_size < b->cols) ? jj + block_size : b->cols;

            for (size_t kk = 0; kk < a->cols; kk += block_size) {
                size_t k_end = (kk + block_size < a->cols) ? kk + block_size : a->cols;

                for (size_t j = jj; j < j_end; j++) {
                    double sum = result->data[i * result->cols + j];

#ifdef __AVX__
                    /* AVX vectorized inner loop - processes 4 doubles at once */
                    size_t k = kk;
                    __m256d sum_vec = _mm256_setzero_pd();

                    /* Process 4 elements at a time */
                    for (; k + 3 < k_end; k += 4) {
                        __m256d a_vec = _mm256_loadu_pd(&a->data[i * a->cols + k]);
                        __m256d b_vec = _mm256_set_pd(
                            b->data[(k + 3) * b->cols + j], b->data[(k + 2) * b->cols + j],
                            b->data[(k + 1) * b->cols + j], b->data[k * b->cols + j]);
                        sum_vec = _mm256_fmadd_pd(a_vec, b_vec, sum_vec);
                    }

                    /* Horizontal sum of AVX register */
                    __m128d sum_high = _mm256_extractf128_pd(sum_vec, 1);
                    __m128d sum_low = _mm256_castpd256_pd128(sum_vec);
                    __m128d sum128 = _mm_add_pd(sum_low, sum_high);
                    __m128d sum64 = _mm_hadd_pd(sum128, sum128);
                    sum += _mm_cvtsd_f64(sum64);

                    /* Handle remaining elements */
                    for (; k < k_end; k++) {
                        sum += a->data[i * a->cols + k] * b->data[k * b->cols + j];
                    }
#else
                    /* Fallback scalar implementation with loop unrolling */
                    size_t k = kk;

                    /* Unroll by 4 for better instruction-level parallelism */
                    for (; k + 3 < k_end; k += 4) {
                        sum += a->data[i * a->cols + k] * b->data[k * b->cols + j];
                        sum += a->data[i * a->cols + k + 1] * b->data[(k + 1) * b->cols + j];
                        sum += a->data[i * a->cols + k + 2] * b->data[(k + 2) * b->cols + j];
                        sum += a->data[i * a->cols + k + 3] * b->data[(k + 3) * b->cols + j];
                    }

                    /* Handle remaining elements */
                    for (; k < k_end; k++) {
                        sum += a->data[i * a->cols + k] * b->data[k * b->cols + j];
                    }
#endif

                    result->data[i * result->cols + j] = sum;
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

#ifdef __AVX__
    /* AVX vectorized addition - processes 4 doubles at once */
    size_t i = 0;
    for (; i + 3 < total; i += 4) {
        __m256d a_vec = _mm256_loadu_pd(&a->data[i]);
        __m256d b_vec = _mm256_loadu_pd(&b->data[i]);
        __m256d result_vec = _mm256_add_pd(a_vec, b_vec);
        _mm256_storeu_pd(&result->data[i], result_vec);
    }

    /* Handle remaining elements */
    for (; i < total; i++) {
        result->data[i] = a->data[i] + b->data[i];
    }
#else
    /* Unrolled loop for better performance */
    size_t i = 0;
    for (; i + 3 < total; i += 4) {
        result->data[i] = a->data[i] + b->data[i];
        result->data[i + 1] = a->data[i + 1] + b->data[i + 1];
        result->data[i + 2] = a->data[i + 2] + b->data[i + 2];
        result->data[i + 3] = a->data[i + 3] + b->data[i + 3];
    }

    /* Handle remaining elements */
    for (; i < total; i++) {
        result->data[i] = a->data[i] + b->data[i];
    }
#endif

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

#ifdef __AVX__
    /* AVX vectorized scaling */
    __m256d scalar_vec = _mm256_set1_pd(scalar);
    size_t i = 0;

    for (; i + 3 < total; i += 4) {
        __m256d data_vec = _mm256_loadu_pd(&m->data[i]);
        __m256d result_vec = _mm256_mul_pd(data_vec, scalar_vec);
        _mm256_storeu_pd(&m->data[i], result_vec);
    }

    /* Handle remaining elements */
    for (; i < total; i++) {
        m->data[i] *= scalar;
    }
#else
    /* Unrolled loop */
    size_t i = 0;
    for (; i + 3 < total; i += 4) {
        m->data[i] *= scalar;
        m->data[i + 1] *= scalar;
        m->data[i + 2] *= scalar;
        m->data[i + 3] *= scalar;
    }

    /* Handle remaining elements */
    for (; i < total; i++) {
        m->data[i] *= scalar;
    }
#endif
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

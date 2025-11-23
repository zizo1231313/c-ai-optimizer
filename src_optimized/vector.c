/* OPTIMIZED VERSION - Hash: ae63df7c5d62de6e14f896f32e7072bdd7d683208c7ffbd7c53943aadd179af3 */

#include "vector.h"
#include "utils.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef __x86_64__
#include <immintrin.h>
#endif

Vector *vector_create(size_t size)
{
    Vector *v = (Vector *) malloc(sizeof(Vector));
    if (v == NULL) {
        return NULL;
    }

    v->size = size;

    /* Use aligned allocation for SIMD operations */
#ifdef __x86_64__
    v->data = (double *) aligned_alloc(32, size * sizeof(double));
#else
    v->data = (double *) malloc(size * sizeof(double));
#endif

    if (v->data == NULL) {
        free(v);
        return NULL;
    }

    /* Zero-initialize memory */
    memset(v->data, 0, size * sizeof(double));

    return v;
}

void vector_free(Vector *v)
{
    if (v != NULL) {
        if (v->data != NULL) {
            free(v->data);
        }
        free(v);
    }
}

double vector_dot(const Vector *a, const Vector *b)
{
    if (a == NULL || b == NULL || a->size != b->size) {
        return 0.0;
    }

    double result = 0.0;

#ifdef __AVX__
    /* AVX vectorized dot product - processes 4 doubles at once */
    __m256d sum_vec = _mm256_setzero_pd();
    size_t i = 0;

    for (; i + 3 < a->size; i += 4) {
        __m256d a_vec = _mm256_loadu_pd(&a->data[i]);
        __m256d b_vec = _mm256_loadu_pd(&b->data[i]);
        sum_vec = _mm256_fmadd_pd(a_vec, b_vec, sum_vec);
    }

    /* Horizontal sum of AVX register */
    __m128d sum_high = _mm256_extractf128_pd(sum_vec, 1);
    __m128d sum_low = _mm256_castpd256_pd128(sum_vec);
    __m128d sum128 = _mm_add_pd(sum_low, sum_high);
    __m128d sum64 = _mm_hadd_pd(sum128, sum128);
    result = _mm_cvtsd_f64(sum64);

    /* Handle remaining elements */
    for (; i < a->size; i++) {
        result += a->data[i] * b->data[i];
    }
#else
    /* Loop unrolling for better instruction-level parallelism */
    size_t i = 0;
    double sum0 = 0.0, sum1 = 0.0, sum2 = 0.0, sum3 = 0.0;

    /* Unroll by 4 to reduce loop overhead and enable parallel execution */
    for (; i + 3 < a->size; i += 4) {
        sum0 += a->data[i] * b->data[i];
        sum1 += a->data[i + 1] * b->data[i + 1];
        sum2 += a->data[i + 2] * b->data[i + 2];
        sum3 += a->data[i + 3] * b->data[i + 3];
    }

    result = sum0 + sum1 + sum2 + sum3;

    /* Handle remaining elements */
    for (; i < a->size; i++) {
        result += a->data[i] * b->data[i];
    }
#endif

    return result;
}

Vector *vector_add(const Vector *a, const Vector *b)
{
    if (a == NULL || b == NULL || a->size != b->size) {
        return NULL;
    }

    Vector *result = vector_create(a->size);
    if (result == NULL) {
        return NULL;
    }

#ifdef __AVX__
    /* AVX vectorized addition */
    size_t i = 0;
    for (; i + 3 < a->size; i += 4) {
        __m256d a_vec = _mm256_loadu_pd(&a->data[i]);
        __m256d b_vec = _mm256_loadu_pd(&b->data[i]);
        __m256d result_vec = _mm256_add_pd(a_vec, b_vec);
        _mm256_storeu_pd(&result->data[i], result_vec);
    }

    /* Handle remaining elements */
    for (; i < a->size; i++) {
        result->data[i] = a->data[i] + b->data[i];
    }
#else
    /* Unrolled loop */
    size_t i = 0;
    for (; i + 3 < a->size; i += 4) {
        result->data[i] = a->data[i] + b->data[i];
        result->data[i + 1] = a->data[i + 1] + b->data[i + 1];
        result->data[i + 2] = a->data[i + 2] + b->data[i + 2];
        result->data[i + 3] = a->data[i + 3] + b->data[i + 3];
    }

    /* Handle remaining elements */
    for (; i < a->size; i++) {
        result->data[i] = a->data[i] + b->data[i];
    }
#endif

    return result;
}

Vector *vector_scale(const Vector *v, double scalar)
{
    if (v == NULL) {
        return NULL;
    }

    Vector *result = vector_create(v->size);
    if (result == NULL) {
        return NULL;
    }

#ifdef __AVX__
    /* AVX vectorized scaling */
    __m256d scalar_vec = _mm256_set1_pd(scalar);
    size_t i = 0;

    for (; i + 3 < v->size; i += 4) {
        __m256d data_vec = _mm256_loadu_pd(&v->data[i]);
        __m256d result_vec = _mm256_mul_pd(data_vec, scalar_vec);
        _mm256_storeu_pd(&result->data[i], result_vec);
    }

    /* Handle remaining elements */
    for (; i < v->size; i++) {
        result->data[i] = v->data[i] * scalar;
    }
#else
    /* Unrolled loop */
    size_t i = 0;
    for (; i + 3 < v->size; i += 4) {
        result->data[i] = v->data[i] * scalar;
        result->data[i + 1] = v->data[i + 1] * scalar;
        result->data[i + 2] = v->data[i + 2] * scalar;
        result->data[i + 3] = v->data[i + 3] * scalar;
    }

    /* Handle remaining elements */
    for (; i < v->size; i++) {
        result->data[i] = v->data[i] * scalar;
    }
#endif

    return result;
}

double vector_magnitude(const Vector *v)
{
    if (v == NULL) {
        return 0.0;
    }

    double sum = 0.0;

#ifdef __AVX__
    /* AVX vectorized magnitude computation */
    __m256d sum_vec = _mm256_setzero_pd();
    size_t i = 0;

    for (; i + 3 < v->size; i += 4) {
        __m256d data_vec = _mm256_loadu_pd(&v->data[i]);
        sum_vec = _mm256_fmadd_pd(data_vec, data_vec, sum_vec);
    }

    /* Horizontal sum */
    __m128d sum_high = _mm256_extractf128_pd(sum_vec, 1);
    __m128d sum_low = _mm256_castpd256_pd128(sum_vec);
    __m128d sum128 = _mm_add_pd(sum_low, sum_high);
    __m128d sum64 = _mm_hadd_pd(sum128, sum128);
    sum = _mm_cvtsd_f64(sum64);

    /* Handle remaining elements */
    for (; i < v->size; i++) {
        sum += v->data[i] * v->data[i];
    }
#else
    /* Loop unrolling with multiple accumulators */
    size_t i = 0;
    double sum0 = 0.0, sum1 = 0.0, sum2 = 0.0, sum3 = 0.0;

    for (; i + 3 < v->size; i += 4) {
        sum0 += v->data[i] * v->data[i];
        sum1 += v->data[i + 1] * v->data[i + 1];
        sum2 += v->data[i + 2] * v->data[i + 2];
        sum3 += v->data[i + 3] * v->data[i + 3];
    }

    sum = sum0 + sum1 + sum2 + sum3;

    /* Handle remaining elements */
    for (; i < v->size; i++) {
        sum += v->data[i] * v->data[i];
    }
#endif

    return sqrt(sum);
}

Vector *vector_normalize(const Vector *v)
{
    if (v == NULL) {
        return NULL;
    }

    /* Inline magnitude computation to avoid redundant calculation */
    double mag = vector_magnitude(v);
    if (mag < 1e-10) {
        return NULL;
    }

    /* Use scale function which is already optimized */
    return vector_scale(v, 1.0 / mag);
}

void vector_print(const Vector *v)
{
    if (v == NULL) {
        printf("NULL vector\n");
        return;
    }

    printf("Vector (%zu): [", v->size);
    for (size_t i = 0; i < v->size; i++) {
        printf("%.3f", v->data[i]);
        if (i < v->size - 1) {
            printf(", ");
        }
    }
    printf("]\n");
}

void vector_fill_random(Vector *v)
{
    if (v == NULL) {
        return;
    }

    /* Sequential access for better cache performance */
    for (size_t i = 0; i < v->size; i++) {
        v->data[i] = utils_random_double(-10.0, 10.0);
    }
}

int vector_equal(const Vector *a, const Vector *b, double epsilon)
{
    if (a == NULL || b == NULL) {
        return 0;
    }

    if (a->size != b->size) {
        return 0;
    }

    /* Early exit on first mismatch */
    for (size_t i = 0; i < a->size; i++) {
        if (!utils_double_equal(a->data[i], b->data[i], epsilon)) {
            return 0;
        }
    }

    return 1;
}

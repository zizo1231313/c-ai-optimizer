/* OPTIMIZED VERSION - Hash: 42d088a79c469f63b47795142a865e1b415bdfd646ecbaa9237162b19781236b */

#include "stats.h"
#include "utils.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

#ifdef __x86_64__
#include <immintrin.h>
#endif

double stats_mean(const double *data, size_t n)
{
    if (data == NULL || n == 0) {
        return 0.0;
    }

    double sum = 0.0;

#ifdef __AVX__
    /* AVX vectorized summation */
    __m256d sum_vec = _mm256_setzero_pd();
    size_t i = 0;

    for (; i + 3 < n; i += 4) {
        __m256d data_vec = _mm256_loadu_pd(&data[i]);
        sum_vec = _mm256_add_pd(sum_vec, data_vec);
    }

    /* Horizontal sum */
    __m128d sum_high = _mm256_extractf128_pd(sum_vec, 1);
    __m128d sum_low = _mm256_castpd256_pd128(sum_vec);
    __m128d sum128 = _mm_add_pd(sum_low, sum_high);
    __m128d sum64 = _mm_hadd_pd(sum128, sum128);
    sum = _mm_cvtsd_f64(sum64);

    /* Handle remaining elements */
    for (; i < n; i++) {
        sum += data[i];
    }
#else
    /* Unrolled loop with multiple accumulators to reduce dependencies */
    size_t i = 0;
    double sum0 = 0.0, sum1 = 0.0, sum2 = 0.0, sum3 = 0.0;

    for (; i + 3 < n; i += 4) {
        sum0 += data[i];
        sum1 += data[i + 1];
        sum2 += data[i + 2];
        sum3 += data[i + 3];
    }

    sum = sum0 + sum1 + sum2 + sum3;

    /* Handle remaining elements */
    for (; i < n; i++) {
        sum += data[i];
    }
#endif

    return sum / (double) n;
}

double stats_variance(const double *data, size_t n)
{
    if (data == NULL || n == 0) {
        return 0.0;
    }

    /* Compute mean once and reuse */
    const double mean = stats_mean(data, n);
    double sum_sq_diff = 0.0;

#ifdef __AVX__
    /* AVX vectorized variance computation */
    __m256d mean_vec = _mm256_set1_pd(mean);
    __m256d sum_vec = _mm256_setzero_pd();
    size_t i = 0;

    for (; i + 3 < n; i += 4) {
        __m256d data_vec = _mm256_loadu_pd(&data[i]);
        __m256d diff = _mm256_sub_pd(data_vec, mean_vec);
        sum_vec = _mm256_fmadd_pd(diff, diff, sum_vec);
    }

    /* Horizontal sum */
    __m128d sum_high = _mm256_extractf128_pd(sum_vec, 1);
    __m128d sum_low = _mm256_castpd256_pd128(sum_vec);
    __m128d sum128 = _mm_add_pd(sum_low, sum_high);
    __m128d sum64 = _mm_hadd_pd(sum128, sum128);
    sum_sq_diff = _mm_cvtsd_f64(sum64);

    /* Handle remaining elements */
    for (; i < n; i++) {
        double diff = data[i] - mean;
        sum_sq_diff += diff * diff;
    }
#else
    /* Unrolled loop with multiple accumulators */
    size_t i = 0;
    double sum0 = 0.0, sum1 = 0.0, sum2 = 0.0, sum3 = 0.0;

    for (; i + 3 < n; i += 4) {
        double diff0 = data[i] - mean;
        double diff1 = data[i + 1] - mean;
        double diff2 = data[i + 2] - mean;
        double diff3 = data[i + 3] - mean;

        sum0 += diff0 * diff0;
        sum1 += diff1 * diff1;
        sum2 += diff2 * diff2;
        sum3 += diff3 * diff3;
    }

    sum_sq_diff = sum0 + sum1 + sum2 + sum3;

    /* Handle remaining elements */
    for (; i < n; i++) {
        double diff = data[i] - mean;
        sum_sq_diff += diff * diff;
    }
#endif

    return sum_sq_diff / (double) n;
}

/* Inline small function to reduce call overhead */
static inline double stats_stddev_impl(const double *data, size_t n)
{
    return sqrt(stats_variance(data, n));
}

double stats_stddev(const double *data, size_t n)
{
    return stats_stddev_impl(data, n);
}

double stats_min(const double *data, size_t n)
{
    if (data == NULL || n == 0) {
        return 0.0;
    }

    double min_val = data[0];

#ifdef __AVX__
    /* AVX vectorized min computation */
    __m256d min_vec = _mm256_set1_pd(min_val);
    size_t i = 1;

    for (; i + 3 < n; i += 4) {
        __m256d data_vec = _mm256_loadu_pd(&data[i]);
        min_vec = _mm256_min_pd(min_vec, data_vec);
    }

    /* Extract minimum from vector */
    double min_array[4];
    _mm256_storeu_pd(min_array, min_vec);
    min_val = min_array[0];
    for (int j = 1; j < 4; j++) {
        if (min_array[j] < min_val) {
            min_val = min_array[j];
        }
    }

    /* Handle remaining elements */
    for (; i < n; i++) {
        if (data[i] < min_val) {
            min_val = data[i];
        }
    }
#else
    /* Unrolled loop for better branch prediction */
    size_t i = 1;
    for (; i + 3 < n; i += 4) {
        if (data[i] < min_val)
            min_val = data[i];
        if (data[i + 1] < min_val)
            min_val = data[i + 1];
        if (data[i + 2] < min_val)
            min_val = data[i + 2];
        if (data[i + 3] < min_val)
            min_val = data[i + 3];
    }

    /* Handle remaining elements */
    for (; i < n; i++) {
        if (data[i] < min_val) {
            min_val = data[i];
        }
    }
#endif

    return min_val;
}

double stats_max(const double *data, size_t n)
{
    if (data == NULL || n == 0) {
        return 0.0;
    }

    double max_val = data[0];

#ifdef __AVX__
    /* AVX vectorized max computation */
    __m256d max_vec = _mm256_set1_pd(max_val);
    size_t i = 1;

    for (; i + 3 < n; i += 4) {
        __m256d data_vec = _mm256_loadu_pd(&data[i]);
        max_vec = _mm256_max_pd(max_vec, data_vec);
    }

    /* Extract maximum from vector */
    double max_array[4];
    _mm256_storeu_pd(max_array, max_vec);
    max_val = max_array[0];
    for (int j = 1; j < 4; j++) {
        if (max_array[j] > max_val) {
            max_val = max_array[j];
        }
    }

    /* Handle remaining elements */
    for (; i < n; i++) {
        if (data[i] > max_val) {
            max_val = data[i];
        }
    }
#else
    /* Unrolled loop */
    size_t i = 1;
    for (; i + 3 < n; i += 4) {
        if (data[i] > max_val)
            max_val = data[i];
        if (data[i + 1] > max_val)
            max_val = data[i + 1];
        if (data[i + 2] > max_val)
            max_val = data[i + 2];
        if (data[i + 3] > max_val)
            max_val = data[i + 3];
    }

    /* Handle remaining elements */
    for (; i < n; i++) {
        if (data[i] > max_val) {
            max_val = data[i];
        }
    }
#endif

    return max_val;
}

static int compare_doubles(const void *a, const void *b)
{
    double diff = *(const double *) a - *(const double *) b;
    /* Branchless comparison for better performance */
    return (diff > 0) - (diff < 0);
}

void stats_sort(double *data, size_t n)
{
    if (data == NULL || n == 0) {
        return;
    }

    /* Use qsort - already highly optimized in standard library */
    qsort(data, n, sizeof(double), compare_doubles);
}

double stats_median(double *data, size_t n)
{
    if (data == NULL || n == 0) {
        return 0.0;
    }

    /* Allocate temporary array */
    double *temp = (double *) malloc(n * sizeof(double));
    if (temp == NULL) {
        return 0.0;
    }

    /* Use memcpy for efficient copying */
    memcpy(temp, data, n * sizeof(double));
    stats_sort(temp, n);

    double median;
    if (n % 2 == 0) {
        /* Average of two middle elements for even-sized arrays */
        median = (temp[n / 2 - 1] + temp[n / 2]) * 0.5;
    } else {
        median = temp[n / 2];
    }

    free(temp);
    return median;
}

double stats_correlation(const double *x, const double *y, size_t n)
{
    if (x == NULL || y == NULL || n == 0) {
        return 0.0;
    }

    /* Compute means once */
    const double mean_x = stats_mean(x, n);
    const double mean_y = stats_mean(y, n);

    double sum_xy = 0.0;
    double sum_x_sq = 0.0;
    double sum_y_sq = 0.0;

#ifdef __AVX__
    /* AVX vectorized correlation computation */
    __m256d mean_x_vec = _mm256_set1_pd(mean_x);
    __m256d mean_y_vec = _mm256_set1_pd(mean_y);
    __m256d sum_xy_vec = _mm256_setzero_pd();
    __m256d sum_x_sq_vec = _mm256_setzero_pd();
    __m256d sum_y_sq_vec = _mm256_setzero_pd();
    size_t i = 0;

    for (; i + 3 < n; i += 4) {
        __m256d x_vec = _mm256_loadu_pd(&x[i]);
        __m256d y_vec = _mm256_loadu_pd(&y[i]);
        __m256d dx = _mm256_sub_pd(x_vec, mean_x_vec);
        __m256d dy = _mm256_sub_pd(y_vec, mean_y_vec);

        sum_xy_vec = _mm256_fmadd_pd(dx, dy, sum_xy_vec);
        sum_x_sq_vec = _mm256_fmadd_pd(dx, dx, sum_x_sq_vec);
        sum_y_sq_vec = _mm256_fmadd_pd(dy, dy, sum_y_sq_vec);
    }

    /* Horizontal sum for all three accumulators */
    __m128d xy_high = _mm256_extractf128_pd(sum_xy_vec, 1);
    __m128d xy_low = _mm256_castpd256_pd128(sum_xy_vec);
    __m128d xy_128 = _mm_add_pd(xy_low, xy_high);
    __m128d xy_64 = _mm_hadd_pd(xy_128, xy_128);
    sum_xy = _mm_cvtsd_f64(xy_64);

    __m128d x_sq_high = _mm256_extractf128_pd(sum_x_sq_vec, 1);
    __m128d x_sq_low = _mm256_castpd256_pd128(sum_x_sq_vec);
    __m128d x_sq_128 = _mm_add_pd(x_sq_low, x_sq_high);
    __m128d x_sq_64 = _mm_hadd_pd(x_sq_128, x_sq_128);
    sum_x_sq = _mm_cvtsd_f64(x_sq_64);

    __m128d y_sq_high = _mm256_extractf128_pd(sum_y_sq_vec, 1);
    __m128d y_sq_low = _mm256_castpd256_pd128(sum_y_sq_vec);
    __m128d y_sq_128 = _mm_add_pd(y_sq_low, y_sq_high);
    __m128d y_sq_64 = _mm_hadd_pd(y_sq_128, y_sq_128);
    sum_y_sq = _mm_cvtsd_f64(y_sq_64);

    /* Handle remaining elements */
    for (; i < n; i++) {
        double dx = x[i] - mean_x;
        double dy = y[i] - mean_y;

        sum_xy += dx * dy;
        sum_x_sq += dx * dx;
        sum_y_sq += dy * dy;
    }
#else
    /* Unrolled loop with multiple accumulators */
    size_t i = 0;
    double xy0 = 0.0, xy1 = 0.0, xy2 = 0.0, xy3 = 0.0;
    double x_sq0 = 0.0, x_sq1 = 0.0, x_sq2 = 0.0, x_sq3 = 0.0;
    double y_sq0 = 0.0, y_sq1 = 0.0, y_sq2 = 0.0, y_sq3 = 0.0;

    for (; i + 3 < n; i += 4) {
        double dx0 = x[i] - mean_x;
        double dy0 = y[i] - mean_y;
        double dx1 = x[i + 1] - mean_x;
        double dy1 = y[i + 1] - mean_y;
        double dx2 = x[i + 2] - mean_x;
        double dy2 = y[i + 2] - mean_y;
        double dx3 = x[i + 3] - mean_x;
        double dy3 = y[i + 3] - mean_y;

        xy0 += dx0 * dy0;
        xy1 += dx1 * dy1;
        xy2 += dx2 * dy2;
        xy3 += dx3 * dy3;

        x_sq0 += dx0 * dx0;
        x_sq1 += dx1 * dx1;
        x_sq2 += dx2 * dx2;
        x_sq3 += dx3 * dx3;

        y_sq0 += dy0 * dy0;
        y_sq1 += dy1 * dy1;
        y_sq2 += dy2 * dy2;
        y_sq3 += dy3 * dy3;
    }

    sum_xy = xy0 + xy1 + xy2 + xy3;
    sum_x_sq = x_sq0 + x_sq1 + x_sq2 + x_sq3;
    sum_y_sq = y_sq0 + y_sq1 + y_sq2 + y_sq3;

    /* Handle remaining elements */
    for (; i < n; i++) {
        double dx = x[i] - mean_x;
        double dy = y[i] - mean_y;

        sum_xy += dx * dy;
        sum_x_sq += dx * dx;
        sum_y_sq += dy * dy;
    }
#endif

    /* Compute denominator */
    double denominator = sqrt(sum_x_sq * sum_y_sq);
    if (denominator < 1e-10) {
        return 0.0;
    }

    return sum_xy / denominator;
}

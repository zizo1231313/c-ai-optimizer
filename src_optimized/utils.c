/* OPTIMIZED VERSION - Hash: c19c4d95af869f12a2b8a6c73a8d88a37fd52d1c2b80e62a1c2fd9e8008b0ad9 */

#include "utils.h"
#include <math.h>
#include <stdlib.h>
#include <sys/time.h>
#include <time.h>

void utils_init_random(void)
{
    srand((unsigned int) time(NULL));
}

double utils_random_double(double min, double max)
{
    /* Use const for parameters to allow better optimization */
    const double random = (double) rand() / (double) RAND_MAX;
    return min + random * (max - min);
}

/* Inline small utility functions to eliminate function call overhead */
static inline double utils_abs_impl(double x)
{
    /* Use fabs from math.h for better performance */
    return fabs(x);
}

double utils_abs(double x)
{
    return utils_abs_impl(x);
}

/* Mark as inline for better performance */
static inline int utils_double_equal_impl(double a, double b, double epsilon)
{
    /* Optimized comparison using fabs */
    return fabs(a - b) < epsilon;
}

int utils_double_equal(double a, double b, double epsilon)
{
    return utils_double_equal_impl(a, b, epsilon);
}

/* Inline timer functions to reduce call overhead */
static inline void timer_start_impl(Timer *restrict t)
{
    struct timeval tv;
    gettimeofday(&tv, NULL);
    /* Combine multiplication and addition for better performance */
    t->start_time = (long long) tv.tv_sec * 1000000LL + (long long) tv.tv_usec;
}

void timer_start(Timer *t)
{
    if (t == NULL) {
        return;
    }
    timer_start_impl(t);
}

static inline void timer_stop_impl(Timer *restrict t)
{
    struct timeval tv;
    gettimeofday(&tv, NULL);
    t->end_time = (long long) tv.tv_sec * 1000000LL + (long long) tv.tv_usec;
}

void timer_stop(Timer *t)
{
    if (t == NULL) {
        return;
    }
    timer_stop_impl(t);
}

/* Inline and optimize elapsed time calculation */
static inline double timer_elapsed_ms_impl(const Timer *restrict t)
{
    const long long diff = t->end_time - t->start_time;
    /* Use multiplication instead of division for better performance */
    return (double) diff * 0.001;
}

double timer_elapsed_ms(const Timer *t)
{
    if (t == NULL) {
        return 0.0;
    }
    return timer_elapsed_ms_impl(t);
}

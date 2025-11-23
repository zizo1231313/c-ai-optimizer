#include <stdio.h>
#include "../include/utils.h"

int run_matrix_tests(void);
int run_vector_tests(void);
int run_stats_tests(void);

int main(int argc, char *argv[]) {
    (void)argc;
    (void)argv;

    int all_passed = 1;

    printf("===========================================\n");
    printf("  C AI Optimizer - Test Suite\n");
    printf("===========================================\n\n");

    all_passed &= run_matrix_tests();
    all_passed &= run_vector_tests();
    all_passed &= run_stats_tests();

    printf("===========================================\n");
    if (all_passed) {
        printf("  ALL TESTS PASSED!\n");
        printf("===========================================\n");
        return 0;
    } else {
        printf("  SOME TESTS FAILED\n");
        printf("===========================================\n");
        return 1;
    }
}

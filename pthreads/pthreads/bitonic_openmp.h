#ifndef BITONIC_OMP_H
#define BITONIC_OMP_h
#include <omp.h>
#include "bitonic.h"

/** OpenMP **/
void parallel_sort_openmp();
void *bitonic_sort_openmp(void *arguments);
void *bitonic_merge_openmp(void *arguments);
#endif

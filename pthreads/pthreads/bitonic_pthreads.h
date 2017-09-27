#ifndef BITONIC_PTHREADS
#define BITONIC_PTHREADS
#include <pthread.h>
#include "bitonic.h"
void parallel_sort_pthreads(void);
void *bitonic_sort_pthreads(void *arguments);
void *bitonic_merge_pthreads(void *arguments);
#endif

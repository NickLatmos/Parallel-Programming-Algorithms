#include "bitonic_openmp.h"
#include "bitonic_pthreads.h"
#include "bitonic_cilk.h"
#include "bitonic.h"
int main(int argc, char **argv) {
  if (argc != 3 || (atoi(argv[2]) > 8)) {
      printf("Usage: %s q\n  where n=2^q is problem size (power of two)\n",
             argv[0]);
      if (atoi(argv[2]) > 8) printf("Too many threads!");
      exit(1);
    }
  int q = atoi(argv[1]);
  int p = atoi(argv[2]);
  N = 1<<q;
  a = (int *) malloc(N * sizeof(int));
  int *temp_a = (int *) malloc(N * sizeof(int));
  THREAD_LEVELS = p;  //e.x. p = 5 then num_of_threads = 2^5 = 32
  PROBLEM_SIZE = q;   //2^PROBLEM_SIZE
  omp_set_nested(1);
  omp_set_num_threads(1<<p);


  /**qsort**/
  init();
  temp_a[0:N]=a[0:N];
  gettimeofday( &startwtime,NULL);
  qsort( a, N, sizeof( int ),asc);
  gettimeofday( &endwtime,NULL);
  execute_time = (double)( ( endwtime.tv_usec - startwtime.tv_usec ) / 1.0e6 + endwtime.tv_sec - startwtime.tv_sec );
  printf( "Qsort wall clock time = %f\n", execute_time );
  test();

  /**serial recursive bitonic sort**/
  a[0:N]=temp_a[0:N];
  gettimeofday(&startwtime,NULL);
  sort();
  gettimeofday(&endwtime,NULL);
  execute_time = (double)( ( endwtime.tv_usec - startwtime.tv_usec ) / 1.0e6 + endwtime.tv_sec - startwtime.tv_sec );
  printf( "Serial Bitonic Sort clock time = %f\n", execute_time );
  test();

  /**imperative bitonic sort**/
  a[0:N]=temp_a[0:N];
  gettimeofday(&startwtime,NULL);
  impBitonicSort();
  gettimeofday(&endwtime,NULL);
  execute_time = (double)( ( endwtime.tv_usec - startwtime.tv_usec ) / 1.0e6 + endwtime.tv_sec - startwtime.tv_sec );
  printf( "Imperative Bitonic Sort clock time = %f\n", execute_time );
  test();

  /**Parallel recursive (+ qsort) bitonic sort**/
  a[0:N]=temp_a[0:N];
  gettimeofday(&startwtime,NULL);
  parallel_sort_pthreads();
  gettimeofday(&endwtime,NULL);
  execute_time = (double)( ( endwtime.tv_usec - startwtime.tv_usec ) / 1.0e6 + endwtime.tv_sec - startwtime.tv_sec );
  printf( "Parallel recursive bitonic  with qsort and %i threads clock time = %f\n", 1 << atoi( argv[ 2 ] ), execute_time );
  test();

  /**Parallel recursive sort with OpemMP**/
  a[0:N]=temp_a[0:N];
  gettimeofday(&startwtime,NULL);
  parallel_sort_openmp();
  gettimeofday(&endwtime,NULL);
  execute_time = (double)( ( endwtime.tv_usec - startwtime.tv_usec ) / 1.0e6 + endwtime.tv_sec - startwtime.tv_sec );
  printf( "Recursive Bitonic Sort with OpemMP clock time = %f\n", execute_time );
  test();
  printf("\n");


  /**Imperative Bitonic Sort with Cilk**/
  a[0:N]=temp_a[0:N];
  gettimeofday(&startwtime,NULL);
  bitonic_sort_cilk();
  gettimeofday(&endwtime,NULL);
  execute_time = (double)( ( endwtime.tv_usec - startwtime.tv_usec ) / 1.0e6 + endwtime.tv_sec - startwtime.tv_sec );
  printf( "Imperative Bitonic Sort with Cilk clock time = %f\n", execute_time );
  test();


  return 0;
}

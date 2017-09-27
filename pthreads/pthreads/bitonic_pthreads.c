#include "bitonic_pthreads.h"

void parallel_sort_pthreads() {
  struct data args={0,N,ASCENDING,PROBLEM_SIZE};
  bitonic_sort_pthreads(&args);
}

void *bitonic_sort_pthreads(void *arguments) {
  struct data *args=(struct data *)arguments;
  if (args->cnt>1) {
      int k=args->cnt/2;
      if(PROBLEM_SIZE <= THREAD_LEVELS || args->problem_size <= PROBLEM_SIZE - THREAD_LEVELS)
        {
          //If we have reached a point where it's not possible to create more
          //threads, then we call the function qsort in serial. We can use
          //recBitonicSort instead of qsort.
          qsort( a + args->lo, k, sizeof( int ), asc );
          qsort( a + ( args->lo + k ) , k, sizeof( int ), desc );
        }
      else {
          pthread_t thread1;
          struct data args1={args->lo, k, ASCENDING, args->problem_size-1};
          struct data args2={args->lo + k, k, DESCENDING, args->problem_size-1};
          pthread_create(&thread1,NULL,bitonic_sort_pthreads,&args1);
          bitonic_sort_pthreads(&args2);
          pthread_join(thread1,NULL);
          //The current thread stops until its subthread finishes its work.
          //When these threads finish we will have a bitonic sequence which has
          //to be sorted in asending order.
        }
      struct data args3={args->lo,args->cnt,args->dir,args->problem_size};
      bitonic_merge_pthreads(&args3);
    }
  return 0;
}

void *bitonic_merge_pthreads(void *arguments) {
  struct data *args=(struct data *)arguments;
  if (args->cnt>1) {
      int k=args->cnt/2;
      int i;
      for (i=args->lo; i<args->lo+k; i++)
        compare(i, i+k, args->dir);
      if(PROBLEM_SIZE <= THREAD_LEVELS || args->problem_size <= PROBLEM_SIZE - THREAD_LEVELS ){
          bitonicMerge(args->lo,k,args->dir);
          bitonicMerge(args->lo+k,k,args->dir);
        }
      else {
          pthread_t thread1;
          //If there are available threads then create 1 in order to run
          //bitonicMerge in Parallel.
          struct data args1={args->lo, k, args->dir, args->problem_size - 1};
          struct data args2={args->lo + k, k, args->dir, args->problem_size - 1};
          pthread_create(&thread1,NULL,bitonic_merge_pthreads,&args1);
          bitonic_merge_pthreads(&args2);
          pthread_join(thread1,NULL);
        }
    }
  return 0;
}





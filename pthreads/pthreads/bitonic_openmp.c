#include "bitonic_openmp.h"
void parallel_sort_openmp(){
  struct data args={0,N,ASCENDING,PROBLEM_SIZE};
  omp_set_nested(1);
  omp_set_num_threads(1<<THREAD_LEVELS);
#pragma omp parallel
#pragma omp single nowait

  {
   #pragma omp taskgroup
    {
      bitonic_sort_openmp(&args);
    }
  }
}

void *bitonic_sort_openmp(void *arguments){
  struct data *args=(struct data *)arguments;
  if (args->cnt>1) {
      int k=args->cnt/2;
      if(PROBLEM_SIZE <= THREAD_LEVELS || args->problem_size <= PROBLEM_SIZE - THREAD_LEVELS ){
          qsort( a + args->lo, k, sizeof( int ), asc );                 //If we have reached a point where it's not possible to create more threads, then we call the function qsort in serial
          qsort( a + ( args->lo + k ) , k, sizeof( int ), desc );       //We can use recBitonicSort instead of qsort
        }
      else {
          struct data args1={args->lo, k, ASCENDING, args->problem_size - 1};
          struct data args2={args->lo + k, k, DESCENDING, args->problem_size - 1};
#pragma omp taskgroup
          {
#pragma omp task

            {
              int tid1 = omp_get_thread_num();

             // printf("Hello World from section 1 = %d\n", tid1);

              bitonic_sort_openmp(&args1);

            }
#pragma omp task
            {
              int tid2 = omp_get_thread_num();
             // printf("Hello World section 2= %d\n", tid2);

              bitonic_sort_openmp(&args2);

            }

          }

          //When these threads finish we will have a bitonic sequence which has to be sorted in asending order.
        }

      struct data args3={args->lo,args->cnt,args->dir,args->problem_size};
#pragma omp task
      {
        bitonic_merge_openmp(&args3);
      }
    }

return 0;
}

void *bitonic_merge_openmp(void *arguments) {
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

          struct data args1={args->lo, k, args->dir, args->problem_size - 1};
          struct data args2={args->lo + k, k, args->dir, args->problem_size - 1};
#pragma omp taskgroup
          {
#pragma omp task
            {

              bitonic_merge_openmp(&args1);
            }
#pragma omp task
            {
              bitonic_merge_openmp(&args2);
            }

          }
        }
    }
  return 0;
}




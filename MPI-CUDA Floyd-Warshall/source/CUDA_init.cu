#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "kernel_C.h"

/*
 * This function is used to make sure cuda is ready 
 * before the real execution of the program.
 */
void initializeCuda()
{
  int *d_k;

  //allocate device memory for source array  
  if( cudaSuccess != cudaMalloc((void **)&d_k,sizeof(int) )) {
    printf("ERROR DURING ALLOCATING MEMORY for k\n");
    exit(1);
  }     
  cudaFree(d_k);
  printf("CUDA is initialized\n");
}


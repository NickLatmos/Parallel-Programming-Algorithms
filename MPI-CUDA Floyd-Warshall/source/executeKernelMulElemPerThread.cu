#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "header.h"
#include "kernel_C.h"
#define NUM_OF_THREADS_PER_BLOCK 64  // 1,2,4,8,16,32,64

extern int world_size;  //number of processes

__global__ void kernelMulElemPerThread(float *A, float *row_k, int *n, int *k);

__global__ void kernelMulElemPerThread(float *A, float *row_k, int *n, int *k) 
{ 
  __shared__ int id_i_k;
  __shared__ float a_i_k;  //This variable is the same for every element in a row. 
  __shared__ int num_of_cells; 
  num_of_cells = 128/blockDim.x;

  int x = threadIdx.x * num_of_cells + blockIdx.x * blockDim.x * num_of_cells;   
  int y = threadIdx.y + blockIdx.y * blockDim.y;        //threadIdx.y = 0
  int id = x + y * gridDim.x * blockDim.x * num_of_cells;   //id holds the first entry coordinates for each thread
  
  id_i_k = (*k) + y * gridDim.x * blockDim.x * num_of_cells;  //id for A(i,k)       
  a_i_k = A[id_i_k];                                       

  for(int i = 0; i < num_of_cells; i++){       
    if( A[id] > a_i_k + row_k[x] )            
      A[id] = a_i_k + row_k[x];           
    id++;                                 
    x++;                                 
  }                                       
}

/*
 * This function calls the device in order to execute the Warshall-Floyd Algorithm
 * It also uses shared memory but one thread is responsible for more than one entries.
 */
void executeKernelMulElemPerThread(int n, int number_of_rows , int world_rank, float *testArray)
{
  int k, root, *d_n, *d_k, step = 0;
  float *d_A, *row_k, *d_row_k;
  cudaError_t cudaError;

  //128 threads per block => n/128 blocks in y-axis and number_of_rows = n/world_size blocks in x-axis => n/128 * n/world_size * 128 = n^2/world_size threads per process => OK! 
  dim3 dimBlock( NUM_OF_THREADS_PER_BLOCK , 1 );  //NUM_OF_THREADS_PER_BLOCK,1
  dim3 dimGrid( n/128 , number_of_rows ); // n/128, number_of_rows, Here it's 128 cells per block !not threads!

  //allocate device memory for source array  
  if( cudaSuccess != cudaMalloc((void **)&d_A, n*n/world_size*sizeof(float) )) {
    printf("ERROR DURING ALLOCATING MEMORY for d_A\n");
    exit(1);
  }     
  
  //allocate device memory   
  if( cudaSuccess != cudaMalloc((void **)&d_row_k, n*sizeof(float)) ){
    printf("ERROR DURING ALLOCATING MEMORY for d_row\n");
    exit(1);
  }

  //allocate device memory
  if( cudaSuccess != cudaMalloc((void **)&d_n, sizeof(int)) ){
    printf("ERROR DURING ALLOCATING MEMORY for d_n\n");
    exit(1);
  }

  //allocate device memory   
  if( cudaSuccess != cudaMalloc((void **)&d_k, sizeof(int)) ){
    printf("ERROR DURING ALLOCATING MEMORY for d_k\n");
    exit(1);
  }

  // Copy inputs to the device
  cudaMemcpy(d_A, testArray, n*n/world_size*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_n, &n, sizeof(int), cudaMemcpyHostToDevice);

  row_k = (float *)malloc(n*sizeof(float));

  //call kernel
  for(k = 0; k < n; k++){

    cudaMemcpy(d_k, &k, sizeof(int), cudaMemcpyHostToDevice);

    //The process that own's the k'th row must broadcast this row to everyone else
    root = k/number_of_rows;

    //Save the row in an array. Only for root
    if(world_rank == root){                                           
      for(int j = 0; j < n; j++)                                    
        row_k[j] = testArray[step * n + j]; //linearArray[k*n + j] = A(k,j) which is the entry I must send                                                                
      step++;
    }

    //Receive the row 
    receiveRow(row_k, n, root);
    cudaMemcpy(d_row_k, row_k, n*sizeof(float), cudaMemcpyHostToDevice); //We must save it in the device memory as it will be used there

    kernelMulElemPerThread<<<dimGrid,dimBlock>>>(d_A, d_row_k, d_n, d_k);
    //cudaThreadSynchronize();

    //Copy result back to host 
    cudaMemcpy(testArray, d_A, n*n/world_size*sizeof(float), cudaMemcpyDeviceToHost);

    //Write to the graphics card the new values. 
    /*if(k != n-1)
      cudaMemcpy(d_A, testArray, n*n/world_size*sizeof(float), cudaMemcpyHostToDevice);*/
  }

  cudaError = cudaGetLastError();
  if (cudaSuccess != cudaError){
    printf("  cudaGetLastError() returned %d: %s\n", cudaError, cudaGetErrorString(cudaError));
    exit(1);
  }

  //free the memory
  cudaFree(d_A);
  cudaFree(d_row_k);
  cudaFree(d_k);
  cudaFree(d_n);
}
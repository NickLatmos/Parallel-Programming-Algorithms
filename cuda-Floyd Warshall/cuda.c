/*Created by Nick Latmos
This program solves the all pair shortest path problem 
using Warshall Floyd algorithm, which is implemented in parallel
using the Nvidia graphics card.*/

#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <math.h>
#include <cuda.h>

float **A;                        //A is the prototype
float *linearArray,*copiedArray; //linearArray will be the array to process

struct timeval startwtime, endwtime;
double seq_time;

void executeKernelWithoutSharedMemory(int n);
void executeKernelWithSharedMemory(int n);
void executekernelMulElemPerThread(int n);
void makeAdjacency(int n,float p,int w);
void warshallFloydAlgorithm(int n);
void initialization(int n);
void printA(int n);
void printLinearArray(int n);
void printEverything(int n);
void resetLinearArray(int n);
void checkWithSerial(int n);  
void checkResults(int n);
void copySerialArray(int n);      
__global__ void kernelWithoutSharedMemory(float *A, int n, int k); 
__global__ void kernelWithSharedMemory(float *A, int n, int k);  
__global__ void kernelMulElemPerThread(float *A, int n, int k); 

/** the main program **/ 
int main( int argc, char **argv ) {

  if (argc != 4 ) {
    printf( "Usage: %s q t\n  .\n", argv[ 0 ] );
    exit( 1 );
  }
  
  int n = atoi(argv[1]);
  float p = (float) atof(argv[2]);
  int w = atoi(argv[3]);

  n = (int) pow(2.0,n);  // n = 2^n

  //allocates memory for tables A,linearArray,copiedArray      
  initialization(n);                  
 
  //creates the initial graph                    
  makeAdjacency(n,p,w);                     

  //Serial algorithm
  gettimeofday( &startwtime, NULL );
  warshallFloydAlgorithm(n);
  gettimeofday( &endwtime, NULL );
  seq_time = (double)( ( endwtime.tv_usec - startwtime.tv_usec ) / 1.0e6 + endwtime.tv_sec - startwtime.tv_sec );
  printf("Time of the serial algorithm: %f seconds\n",seq_time );
  copySerialArray(n); //copies linearArray to copiedArray
  resetLinearArray(n); //resets the linearArray to use it again from another function

  //Call Kernel without shared memory
  gettimeofday( &startwtime, NULL );
  executeKernelWithoutSharedMemory(n);
  gettimeofday( &endwtime, NULL );
  seq_time = (double)( ( endwtime.tv_usec - startwtime.tv_usec ) / 1.0e6 + endwtime.tv_sec - startwtime.tv_sec );
  printf("Time of the kernel without shared memory: %f seconds\n",seq_time );

  //Call Kernel with shared memory
  gettimeofday( &startwtime, NULL );
  executeKernelWithSharedMemory(n);
  gettimeofday( &endwtime, NULL );
  seq_time = (double)( ( endwtime.tv_usec - startwtime.tv_usec ) / 1.0e6 + endwtime.tv_sec - startwtime.tv_sec );
  printf("Time of the kernel with shared memory: %f seconds\n",seq_time );

  //Call kernel with shared memory and multiple cells per thread
  gettimeofday( &startwtime, NULL );
  executekernelMulElemPerThread(n);
  gettimeofday( &endwtime, NULL );
  seq_time = (double)( ( endwtime.tv_usec - startwtime.tv_usec ) / 1.0e6 + endwtime.tv_sec - startwtime.tv_sec );
  printf("Time of the kernel with shared memory and multiple cells per thread: %f seconds\n",seq_time );

  return 0;
}

/*Initializes tables A by allocating memory for them*/
void initialization(int n)
{
  int z;
  linearArray = (float *)malloc((n*n)*sizeof(float));
  copiedArray = (float *)malloc((n*n)*sizeof(float));

  A = (float **)malloc(n*sizeof(float *));
  for(z = 0; z < n;z++)
    A[z] = (float *)malloc(n*sizeof(float));
}

/*This function creates a graph G(V,E).*/
void makeAdjacency(int n,float p,int w)
{
  int i,j;
  float a;

  srand(time(NULL)); // randomize seed

  for(i = 0; i < n; i++){
    for(j = 0; j < n;j++)
      A[i][j] = 0;
      linearArray[i*n + j] = 0;
  }

  for(i = 0; i < n; i++){
    for(j = 0; j < n; j++){
      if( (a = (float) rand()/(float) RAND_MAX) > p)
        A[i][j] = INFINITY;
      else
        A[i][j] = a*w;
      linearArray[i*n+j] = A[i][j];
      if(i == j)
        linearArray[i*n + j] = 0;
    }
    A[i][i] = 0;
  }
  printf("The graph has been created\n");
}

/*Copies the linearArray in copiedArray*/
void copySerialArray(int n)
{
  int i,j;
  for(i = 0; i < n; i++){
    for(j = 0; j < n; j++)
      copiedArray[i*n + j] = linearArray[i*n + j];  //row order
  }
}

/*Equalizes linearArray to A*/
void resetLinearArray(int n)
{
  int i,j;
  
  for(i = 0; i < n; i++){
    for(j = 0; j < n; j++)
      linearArray[i*n + j] = A[i][j];
  }
}

/*Prints the elements of the table*/
void printLinearArray(int n)
{ 
  int i;
  for(i = 0; i < n*n; i++)
      printf("linearArray[%d] = %f\n",i,linearArray[i]);
}

void printEverything(int n)
{
  int i,j;
  for(i = 0; i < n; i++){
    for(j = 0; j < n; j++){
      printf("A[%d][%d] = %f and linearArray[%d] = %f and cpAr[%d] = %f\n",i,j,A[i][j], (i*n+j) ,linearArray[i*n+j],(i*n+j), copiedArray[i*n+j] );
    }
  }
}

void printA(int n)
{
  int i,j;
  for(i = 0; i < n; i++){
    for(j = 0; j < n; j++)
      printf("A[%d][%d] = %f\n",i,j,A[i][j]);
  }
}

/*Checks if linearArray is equal with the starting Array. Not really useful... I used it for testing*/
void checkWithSerial(int n)
{
  int i,j,flag = 0;
  for(i = 0; i < n; i++){
    for(j = 0; j < n; j++){
      if(A[i][j] == linearArray[i*n + j])
        continue;
      else
        flag = 1;
    }
  }
  if(flag == 0)
    printf("A and linearArray are [EQUAL]\n");
  else
    printf("A and lineararray are [DIFFERENT]\n");
}

/*Implementation of Warshall-Floyd's algorithm for table A */
/*It finds the minimum distance between each (i,j) pair of table linearArray (A)*/
void warshallFloydAlgorithm(int n)
{
  int k,i,j;

  for(k = 0; k < n; k++){
    for(i = 0; i < n; i++){
      for(j = 0; j < n; j++){
        if(linearArray[i*n + j] > linearArray[i*n + k] + linearArray[k*n + j] ) 
          linearArray[i*n + j] = linearArray[i*n + k] + linearArray[k*n + j];
      }
    }
  }
}

/*Checks if the array we found is equal to that from the serial algorithm*/
void checkResults(int n)
{
  int i,j,flag = 0;
  for(i = 0; i < n; i++){
    for(j = 0; j < n; j++){
      if(linearArray[i*n + j] == copiedArray[i*n + j] )
        continue;
      else{
        flag = 1;
        //printf("flag = 1 ! We are at %d index = %f\n",(i*n+j),linearArray[i*n + j] - copiedArray[i*n + j] );
      }
    }
  }
  if(flag == 0)
    printf("[SUCCEDDED]\n");
  else
    printf("[FAILED]\n");
}

/*This function a simple implementation of the Warshall Floyd algorithm.
The blocksize and the Gridsize don't play a major role*/
__global__ void kernelWithoutSharedMemory(float *A, int n, int k) 
{ 
  //thread indexing in row major order
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  
  if(row < n && col < n){
    if( A[n*row + col] > A[n*row + k] + A[k*n + col] ) 
      A[n*row + col] = A[n*row + k] + A[k*n + col]; 
    //__syncthreads(); // all the threads for this k must finish their job in order to move on and create the next matrix
  }
}

/*This function is different from the function kernelWithoutSharedMemory,
due to the fact that this one uses shared memoryfor the element A(i,k), 
because it is used from every thread in a particular block. 
Also it is important to give the right dimensions in order to work correctly.
I give blocksize = 1x128 and Grid = N/128xN*/
__global__ void kernelWithSharedMemory(float *A, int n, int k) 
{ 
  __shared__ float arrayElement;

  //thread indexing in row major order
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y;
  
  if(row < n && col < n){
    arrayElement = A[n*row + k];
    if( A[n*row + col] > arrayElement + A[k*n + col] ) 
      A[n*row + col] =  arrayElement + A[k*n + col]; 
    //__syncthreads(); // all the threads for this k must finish their job in order to move on and create the next matrix
  }
}


/*In order for this function to run correctly it is essential that we give the correct 
dimensions in the blocks and the Grid. I use block = 128x1 and Grid = N/128x1. In this 
way I create N threads, equal to the number of elements in a row. Each thread will 
be responsible for every column in table A*/
__global__ void kernelMulElemPerThread(float *A, int n, int k) 
{ 
  __shared__ float arrayElement;
  int col = blockIdx.x * blockDim.x + threadIdx.x;  //The column of the table A varries. It takes values from 0-127 for each block.
  int row;                                          //The row is constant for every thread in every block due to the blockSize and GridDimension we gave.
  
  if(col < n){
    for(row = 0; row < n; row++){                   //Each thread will calculate the whole column of array A (row=0,1,2...N-1)
      arrayElement = A[n*row + k];
      if( A[n*row + col] > arrayElement + A[k*n + col] ) //every thread will be in the same row, even the threads from different blocks. 
        //This means that we have to increase the row by 1 for every loop until row = n-1. Then k increasses and we repeat this process until k = n-1.
        A[n*row + col] =  arrayElement + A[k*n + col]; 
      //__syncthreads(); // all the threads for this k must finish their job in order to move on. After this command all the threads will be transferred in row = row + 1.
    }
  }
}

/*Allocates memory for the device, copies the linearArray to the device,
calls the kernel in the device and finally checks the results.
Special ATTENTION to the description above the kernelWithoutSharedMemory*/
void executeKernelWithoutSharedMemory(int n)
{
  int k;
  float *d_A;
  
  dim3 dimBlock( 128, 1 );
  dim3 dimGrid( n/128, n ); 

  //allocate device memory for source array  
  if( cudaSuccess != cudaMalloc((void **)&d_A,(n*n)*sizeof(float)) ){
    printf("ERROR DURING ALLOCATIONG MEMORY\n");
    exit(1);
  }     
  
  // Copy inputs to the device
  cudaMemcpy(d_A, linearArray, (n*n)*sizeof(float), cudaMemcpyHostToDevice);
 
  //call kernel
  for(k = 0; k < n; k++){
    kernelWithoutSharedMemory<<<dimGrid,dimBlock>>>(d_A,n,k);
    cudaThreadSynchronize();
  }

  /*if (cudaSuccess != cudaGetLastError())
    printf( "Error!\n" );
 */
  //copy result back to host
  cudaMemcpy(linearArray, d_A, (n*n)*sizeof(float), cudaMemcpyDeviceToHost);
  
  //free the memory
  cudaFree(d_A);

  printf("Results without shared memory: ");
  checkResults(n);
  resetLinearArray(n);
}

/*Allocates memory for the device, copies the linearArray to the device,
calls the kernel in the device and finally checks the results.
Special ATTENTION to the description above the kernelWithSharedMemory*/
void executeKernelWithSharedMemory(int n)
{
  int k;
  float *d_A;

  dim3 dimBlock( 128, 1 );
  dim3 dimGrid( n/dimBlock.x, n ); 

  //allocate device memory for source array  
  if( cudaSuccess != cudaMalloc((void **)&d_A,(n*n)*sizeof(float)) ){
    printf("ERROR DURING ALLOCATIONG MEMORY\n");
    exit(1);
  }
  
  // Copy inputs to the device
  cudaMemcpy(d_A, linearArray, (n*n)*sizeof(float), cudaMemcpyHostToDevice);
 
 //call kernel
  for(k = 0; k < n; k++){
    kernelWithSharedMemory<<<dimGrid,dimBlock>>>(d_A,n,k);
    cudaThreadSynchronize();
  }

  if (cudaSuccess != cudaGetLastError() )
    printf( "Error!\n" );
 
  //copy result back to host
  cudaMemcpy(linearArray, d_A, (n*n)*sizeof(float), cudaMemcpyDeviceToHost);
  
  //free the memory
  cudaFree(d_A);

  printf("Results with shared memory: ");
  checkResults(n);
  resetLinearArray(n);
}

/*Allocates memory for the device, copies the linearArray to the device,
calls the kernel in the device and finally checks the results.
Special ATTENTION to the description above the kernelMulElemPerThread*/
void executekernelMulElemPerThread(int n)
{
  int k;
  float *d_A;

  dim3 dimBlock( 128 , 1 );
  dim3 dimGrid( n/dimBlock.x, 1 ); 

  //allocate device memory for source array  
  if( cudaSuccess != cudaMalloc((void **)&d_A,(n*n)*sizeof(float)) ){
    printf("ERROR DURING ALLOCATIONG MEMORY\n");
    exit(1);
  }
  
  // Copy inputs to the device
  cudaMemcpy(d_A, linearArray, (n*n)*sizeof(float), cudaMemcpyHostToDevice);
 
 //call kernel
  for(k = 0; k < n; k++){
    kernelMulElemPerThread<<<n/128,128>>>(d_A,n,k);
    cudaThreadSynchronize();
  }

  if (cudaSuccess != cudaGetLastError() )
    printf( "Error!\n" );
 
  //copy result back to host
  cudaMemcpy(linearArray, d_A, (n*n)*sizeof(float), cudaMemcpyDeviceToHost);
  
  //free the memory
  cudaFree(d_A);

  printf("Results with shared memory and multiple cells per thread: ");
  checkResults(n);
  resetLinearArray(n);
}
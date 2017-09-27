/*Global indicates a function that runs on the device and is called by the host*/

#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <math.h>
#include <cuda.h>

float **A;          //A is the prototype
float *linearArray; //linearArray will be the array to process

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
void resetLinearArray(int n);
void checkResults(int n);
int checkResults1(int n);
__global__ void kernelWithoutSharedMemory(float *A, int *n);
__global__ void kernelWithSharedMemory(float *A, int *n); 
__global__ void kernelMulElemPerThread(float *A, int *n) ;

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

  //allocates memory for tables A,B.      
  initialization(n);                  
 
  //creates the initial graph                     
  makeAdjacency(n,p,w);                     
  
  //Serial algorithm
  /*gettimeofday( &startwtime, NULL );
  warshallFloydAlgorithm(n);
  gettimeofday( &endwtime, NULL );
  seq_time = (double)( ( endwtime.tv_usec - startwtime.tv_usec ) / 1.0e6 + endwtime.tv_sec - startwtime.tv_sec );
  checkResults(n);
  printf("Time of the serial algorithm: %f seconds\n",seq_time );
  resetLinearArray(n);*/


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

/*Initializes tables A,B by allocating memory for them*/
void initialization(int n)
{
  int z;
  linearArray = (float *)malloc((n*n)*sizeof(float));

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

void printA(int n)
{
  int i,j;
  for(i = 0; i < n; i++){
    for(j = 0; j < n; j++)
      printf("A[%d][%d] = %f\n",i,j,A[i][j]);
  }
}

/*Implementation of Warshall-Floyd's algorithm for table A */
/*It finds the minimum distance between each (i,j) pair of table A*/
void warshallFloydAlgorithm(int n)
{
  int k,i,j;

  for(k = 0; k < n; k++){
    for(i = 0; i < n; i++){
      for(j = 0; j < n; j++){
        if(linearArray[i*n + j] > linearArray[i*n + k] + linearArray[k*i + j] ) 
          linearArray[i*n + j] = linearArray[i*n + k] + linearArray[k*i + j];
 	    }
	  }
  }
}

/*Checks if the diastances which where found are smaller than the initial values of table A*/
void checkResults(int n)
{
  int i,j;
  int flag = 0;

  printf("Checking ...\n");
  for(i = 0; i < n; i++){
  	for(j = 0; j < n; j++){
  	  if(linearArray[i*n + j] <= A[i][j])
  		  continue;
  	  else
  		  flag = 1;
    }
  }
  if(flag == 1)
    printf("[FAILED]\n");
  else{
    if(checkResults1(n) == 0)
      printf("All components are equal, probably FAILED or the initial matrix is already the best possible\n");
    else
      printf("SUCCEDDED\n");
  }
}

/*Checks if the diastances which where found are smaller than the initial values of table A*/
int checkResults1(int n)
{
  int i,j;
  for(i = 0; i < n; i++){
    for(j = 0; j < n; j++){
      if(A[i][j] == linearArray[i*n + j])
        continue;
      else
        return 1;
    }
  }
  return 0;
}

/*This function a simple implementation of the Warshall Floyd algorithm.
The blocksize and the Gridsize don't play a major role*/
__global__ void kernelWithoutSharedMemory(float *A, int *n) 
{ 
  int k = 0;
  int N = *n;

  //thread indexing in row major order
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  
  if(row < N && col < N){
    for(k = 0; k < N; k++){
      if( A[N*row + col] > A[N*row + k] + A[k*row +col] ) //A[i][j] = A[j + i*N], A[i][k] = A[k + N*row] ?? not sure, A[k][j] = A[col + N*k] ??     
        A[N*row + col] = A[N*row + k] + A[k*row +col]; 
      __syncthreads(); // all the threads for this k must finish their job in order to move on and create the next matrix
    }
  }
}

/*This function is different from the function kernelWithoutSharedMemory,
due to the fact that this one uses shared memory for the variable k and
for the element A(i,k), because it is used from every thread in a particular block. 
Also it is important to give the right dimension in order to work correctly.
I give blocksize = 1x128 and Grid = NxN/128*/
__global__ void kernelWithSharedMemory(float *A, int *n) 
{ 
  __shared__ int k;
  __shared__ float arrayElement;
  int N = *n;

  //thread indexing in row major order
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  
  if(row < N && col < N){
    for(k = 0; k < N; k++){
      arrayElement = A[N*row + k];
      if( A[N*row + col] > arrayElement + A[k*row +col] ) //A[i][j]
        A[N*row + col] =  arrayElement + A[k*row +col]; 
      __syncthreads(); // all the threads for this k must finish their job in order to move on and create the next matrix
    }
  }
}


/*In order for this function to run correctly it is essential that we give the correct 
dimensions in the blocks and the Grid. I use block = 1x128 and Grid = 1xN/128. In this 
way I create N threads, equal to the number of elements of a row.Each thread will 
be responsible for every column in table A*/
__global__ void kernelMulElemPerThread(float *A, int *n) 
{ 
  __shared__ int k;
  __shared__ float arrayElement;
  int N = *n;

  int col = blockIdx.y * blockDim.y + threadIdx.y;  //The column of the table A varries. It takes values from 0-127 for each block.
  int row;                                          //The row is constant for every thread in every block due to the blockSize and GridDimension we gave.
  
  if(col < N){
    for(k = 0; k < N; k++){
      for(row = 0; row < N; row++){                   //Each thread will calculate the whole column of array A (row=0,1,2...N-1)
        arrayElement = A[N*col + k];
        if( A[N*col + row] > arrayElement + A[k*col + row] ) //every thread will be in the same row, even the threads from different blocks. 
          //This means that we have to increase the row by 1 for every loop until row = n-1. Then k increasses and we repeat this process until k = n-1.
          A[N*col + row] =  arrayElement + A[k*col + row]; 
        __syncthreads(); // all the threads for this k must finish their job in order to move on. After this command all the threads will be transferred in row = row + 1.
      }
    }
  }
}

/*Allocates memory for the device, copies the linearArray to the device,
calls the kernel in the device and finally checks the results.
Special ATTENTION to the description above the kernelWithoutSharedMemory*/
void executeKernelWithoutSharedMemory(int n)
{
  int *d_n;
  float *d_A;

  int NUMBER_OF_THREADS = 8;  
  int blocksize = NUMBER_OF_THREADS;
  dim3 dimBlock( blocksize, blocksize );
  dim3 dimGrid( n/dimBlock.x, n/dimBlock.y ); 

  //allocate device memory for source array  
  if( cudaSuccess != cudaMalloc((void **)&d_A,(n*n)*sizeof(float)) ){
    printf("ERROR DURING ALLOCATIONG MEMORY\n");
    exit(1);
  }
  
  //allocate device memory for variable n 
  if( cudaSuccess != cudaMalloc((void **)&d_n, sizeof(int)) ){        
    printf("ERROR DURING ALLOCATIONG MEMORY\n");
    exit(1);
  }
  
  // Copy inputs to the device
  cudaMemcpy(d_A, linearArray, (n*n)*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_n, &n, sizeof(int), cudaMemcpyHostToDevice);
 
 //call kernel
  kernelWithoutSharedMemory<<<dimGrid,dimBlock>>>(d_A,d_n);

  if (cudaSuccess != cudaGetLastError())
    printf( "Error!\n" );
 
  //copy result back to host
  cudaMemcpy(linearArray, d_A, (n*n)*sizeof(float), cudaMemcpyDeviceToHost);
  
  //free the memory
  cudaFree(d_A);
  cudaFree(d_n);

  checkResults(n);
  resetLinearArray(n);
}

/*Allocates memory for the device, copies the linearArray to the device,
calls the kernel in the device and finally checks the results.
Special ATTENTION to the description above the kernelWithSharedMemory*/
void executeKernelWithSharedMemory(int n)
{
  int *d_n;
  float *d_A;

  int NUMBER_OF_THREADS = 128;  
  int blocksize = NUMBER_OF_THREADS;
  dim3 dimBlock( 1, blocksize );
  dim3 dimGrid( n, n/dimBlock.y ); 

  //allocate device memory for source array  
  if( cudaSuccess != cudaMalloc((void **)&d_A,(n*n)*sizeof(float)) ){
    printf("ERROR DURING ALLOCATIONG MEMORY\n");
    exit(1);
  }
  
  //allocate device memory for variable n 
  if( cudaSuccess != cudaMalloc((void **)&d_n, sizeof(int)) ){        
    printf("ERROR DURING ALLOCATIONG MEMORY\n");
    exit(1);
  }
  
  // Copy inputs to the device
  cudaMemcpy(d_A, linearArray, (n*n)*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_n, &n, sizeof(int), cudaMemcpyHostToDevice);
 
 //call kernel
  kernelWithSharedMemory<<<dimGrid,dimBlock>>>(d_A,d_n);

  if (cudaSuccess != cudaGetLastError())
    printf( "Error!\n" );
 
  //copy result back to host
  cudaMemcpy(linearArray, d_A, (n*n)*sizeof(float), cudaMemcpyDeviceToHost);
  
  //free the memory
  cudaFree(d_A);
  cudaFree(d_n);

  checkResults(n);
  resetLinearArray(n);
}

/*Allocates memory for the device, copies the linearArray to the device,
calls the kernel in the device and finally checks the results.
Special ATTENTION to the description above the kernelMulElemPerThread*/
void executekernelMulElemPerThread(int n)
{
  int *d_n;
  float *d_A;

  int NUMBER_OF_THREADS = 128;  
  int blocksize = NUMBER_OF_THREADS;
  dim3 dimBlock( 1, blocksize );
  dim3 dimGrid( 1, n/dimBlock.y ); 

  //allocate device memory for source array  
  if( cudaSuccess != cudaMalloc((void **)&d_A,(n*n)*sizeof(float)) ){
    printf("ERROR DURING ALLOCATIONG MEMORY\n");
    exit(1);
  }

  //allocate device memory for variable n 
  if( cudaSuccess != cudaMalloc((void **)&d_n, sizeof(int)) ){        
    printf("ERROR DURING ALLOCATIONG MEMORY\n");
    exit(1);
  }
  
  // Copy inputs to the device
  cudaMemcpy(d_A, linearArray, (n*n)*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_n, &n, sizeof(int), cudaMemcpyHostToDevice);
 
 //call kernel
  kernelMulElemPerThread<<<dimGrid,dimBlock>>>(d_A,d_n);

  if (cudaSuccess != cudaGetLastError())
    printf( "Error!\n" );
 
  //copy result back to host
  cudaMemcpy(linearArray, d_A, (n*n)*sizeof(float), cudaMemcpyDeviceToHost);
  
  //free the memory
  cudaFree(d_A);
  cudaFree(d_n);

  checkResults(n);
  resetLinearArray(n);
}



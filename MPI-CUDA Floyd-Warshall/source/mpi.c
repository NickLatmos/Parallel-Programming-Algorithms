#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <sys/time.h>
#include <mpi.h>
#include "header.h"
#include "kernel.h"

int world_size;
float **A;                   
float *copiedArray;

struct timeval startwtime, endwtime;
double seq_time;

int main(int argc, char** argv) {
 
  if (argc != 5 ) {
    printf("Insert four parameters [n][p][w][kernel type] \n", argv[ 0 ] );
    printf("kernel type 0: Kernel without shared memory \n");
    printf("kernel type 1: Kernel with shared memory \n");
    printf("kernel type 2: Kernel with multiple cells per thread\n");
    exit(1);
  }

  int n = atoi(argv[1]);
  float p = (float) atof(argv[2]);
  int w = atoi(argv[3]);
  int kernel_type = atoi(argv[4]); 
  n = (int) pow(2.0,n);  // n = 2^n

  // Initialize the MPI environment
  MPI_Init(NULL, NULL);
  // Get the number of processes
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  int number_of_rows = n/world_size;
  // Get the rank of the process
  int world_rank;	
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

  if(world_size == 1){
  	printf("The number of processes must be greater than 1.\n");
  	printf("To check the the algorithm with one process please refer to project No3.\n");
  	exit(1);
  }

  float *testArray = (float *)malloc(n*n/world_size*sizeof(float));

  if(world_rank == 0){
  	//allocates memory for tables A,linearArray,copiedArray      
	initialization(n);                  

	//creates the initial graph                    
	makeAdjacency(n,p,w); 

	//copy A[][] to copiedArray to use it for checking
	copySerialArray(n);

    //send a specific block to each process except for process 0
    sendBlocksToProcesses(n,number_of_rows);

    for(int i = 0; i < number_of_rows; i++){
      for(int j = 0; j < n; j++)
    	testArray[i*n + j] = A[i][j];    
    }

  }else
    MPI_Recv(testArray, (n*n/world_size) , MPI_FLOAT , 0 , 0, MPI_COMM_WORLD ,MPI_STATUS_IGNORE); 

  if(world_rank == 0)
    initializeCuda();
  switch(kernel_type){
  	case 0:
  	  gettimeofday( &startwtime, NULL );
  	  executeKernelWithoutSharedMemory(n, number_of_rows, world_rank, testArray);
  	  buildFinalArray(world_rank, n, testArray, number_of_rows);
  	  gettimeofday( &endwtime, NULL );
  	  seq_time = (double)( ( endwtime.tv_usec - startwtime.tv_usec ) / 1.0e6 + endwtime.tv_sec - startwtime.tv_sec );
      if(!world_rank)
        printf("Time of the kernel without shared memory and %d processes: %f seconds\n",world_size, seq_time );
  	  break;

  	case 1:
  	  gettimeofday( &startwtime, NULL );
  	  executeKernelWithSharedMemory(n, number_of_rows, world_rank, testArray);
  	  buildFinalArray(world_rank, n, testArray, number_of_rows);
  	  gettimeofday( &endwtime, NULL );
  	  seq_time = (double)( ( endwtime.tv_usec - startwtime.tv_usec ) / 1.0e6 + endwtime.tv_sec - startwtime.tv_sec );
  	  if(!world_rank)
        printf("Time of the kernel with shared memory and %d processes: %f seconds\n",world_size, seq_time );
  	  break;

  	case 2:
  	  gettimeofday( &startwtime, NULL );
  	  executeKernelMulElemPerThread(n, number_of_rows, world_rank, testArray);
  	  buildFinalArray(world_rank, n, testArray, number_of_rows);
  	  gettimeofday( &endwtime, NULL );
  	  seq_time = (double)( ( endwtime.tv_usec - startwtime.tv_usec ) / 1.0e6 + endwtime.tv_sec - startwtime.tv_sec );
  	  if(!world_rank)
        printf("Time of the kernel with multiple cells per thread and %d processes: %f seconds\n",world_size, seq_time );
  	  break;

  	default:
  	  printf("Invalid kernel type\n");
  	  exit(1);
  	  break;
  }

  //Serial Implementation and Testing
  if(world_rank == 0){
  	gettimeofday( &startwtime, NULL );
  	warshallFloydAlgorithm(n);
  	gettimeofday( &endwtime, NULL );
  	seq_time = (double)( ( endwtime.tv_usec - startwtime.tv_usec ) / 1.0e6 + endwtime.tv_sec - startwtime.tv_sec );
  	printf("Time of serial Warshall-Floyd: %f seconds\n", seq_time );
  	checkResults(n);
  }

  MPI_Finalize();
}

/*
 * This function sends blocks of the original array
 * to the other processes
 */
void sendBlocksToProcesses(int n, int number_of_rows)
{
  float **blockArray;

  //blockArray[i] contains the entries for each process i except for process 0
  blockArray = (float **)malloc((world_size -1) *sizeof(float *)); 
  for(int i = 0; i < world_size - 1; i++)
    blockArray[i] = (float *)malloc(n*n/world_size*sizeof(float));  

  for(int l = 0; l < world_size - 1; l++){
    for(int i = 0; i < number_of_rows; i++){
      for(int j = 0; j < n; j++)
        blockArray[l][i*n + j] = A[i + (l+1) * number_of_rows][j];
    }
  }

  for(int i = 0; i < world_size - 1; i++)
    MPI_Send(blockArray[i] , (n*n/world_size) , MPI_FLOAT , (i + 1) ,  0 , MPI_COMM_WORLD); 
	
  free(blockArray);
}

/*
 * Initializes A and copiedArray
 */
void initialization(int n)
{
  int z; 
  copiedArray = (float *)malloc((n*n)*sizeof(float));

  A = (float **)malloc(n*sizeof(float *));
  for(z = 0; z < n;z++)
    A[z] = (float *)malloc(n*sizeof(float));
}

/*
 * This function creates a graph G(V,E).
 */
void makeAdjacency(int n,float p,int w)
{
  int i,j;
  float a;

  srand(time(NULL)); // randomize seed

  for(i = 0; i < n; i++){
    for(j = 0; j < n;j++)
      A[i][j] = 0;   
  }

  for(i = 0; i < n; i++){
    for(j = 0; j < n; j++){
      if( (a = (float) rand()/(float) RAND_MAX) > p)
        A[i][j] = INFINITY;
      else
        A[i][j] = a*w;
    }
    A[i][i] = 0;
  }
  printf("The graph has been created\n");
}

/*
 * Prints the elements of the testArray
 */
void printTestArray(int n, float *testArray)
{ 
  int i;
  for(i = 0; i < n*n/world_size; i++)
      printf("testArray[%d] = %f\n",i,testArray[i]);
}

/*
 * Prints the elements from copiedArray
 */
void printCopiedArray(int n)
{ 
  int i;
  for(i = 0; i < n; i++){
    for(int j = 0; j < n; j++)
      printf("copiedArray[%d][%d] = %f\n",i,j,copiedArray[i*n + j]);
  }
}

/*
 * Checks if the array we found is equal to that from the serial algorithm
 */
void checkResults(int n)
{
  int i,j,flag = 0;
  printf("PLease wait, Checking results...\n");
  for(i = 0; i < n; i++){
    for(j = 0; j < n; j++){
      if(A[i][j] == copiedArray[i*n + j] )
        continue;
      else{
        flag = 1;
        printf("FAULT! We are at index %d \n",(i*n+j));
        printf("A[%d][%d] = %f, copiedArray[%d] = %f\n", i, j, A[i][j], i*n+j , copiedArray[i*n + j] );
        //exit(1);
      }
    }
  }
  if(flag == 0)
    printf("[SUCCEDDED]\n");
  else
    printf("[FAILED]\n");
}

/*
 * Implementation of Warshall-Floyd's algorithm for copiedArray 
 * It finds the minimum distance between each (i,j) pair of table copiedArray (A)
 */
void warshallFloydAlgorithm(int n)
{
  int k,i,j;

  for(k = 0; k < n; k++){
    for(i = 0; i < n; i++){
      for(j = 0; j < n; j++){
        if(copiedArray[i*n + j] > copiedArray[i*n + k] + copiedArray[k*n + j] ) 
          copiedArray[i*n + j] = copiedArray[i*n + k] + copiedArray[k*n + j];
 	  }
	}
	//printf("Warshall-Floyd step k = %d, results:\n",k );
	//printArrayFromWarshallFloyd(n);
  }
}

/*
 * Copies the matrix A[][] in copiedArray
 */
void copySerialArray(int n)
{
  int i,j;
  for(i = 0; i < n; i++){
    for(j = 0; j < n; j++)
      copiedArray[i*n + j] = A[i][j]; //row order
  }
}

/*
 * Prints matrix A
 */
void printMatrix_A(int n)
{
  for(int i = 0; i < n; i++){
  	for(int j = 0; j < n; j++)
  	  printf("A[%d][%d] = %f\n",i,j,A[i][j] );
  }
}

/*
 * This functions builds the final array after the parallel implementation of
 * Warshall Floyd Algorithm
 */
void buildFinalArray(int world_rank,int n,float *testArray, int number_of_rows)
{
  MPI_Status status;

  if(world_rank != 0){
  	//Send the piece of the final array to process 0 
  	MPI_Send(testArray , (n*n/world_size) , MPI_FLOAT , 0 ,  0 , MPI_COMM_WORLD); 
  }else{

  	/*Fill the values from process 0 to matrix A*/
  	for(int i = 0; i < number_of_rows; i++){
  	  for(int j = 0; j < n; j++)
  	  	A[i][j] = testArray[i * n + j]; 	  
  	}

  	/*Receive the array pieces from world_size - 1 (except for 0) processes*/
  	for(int l = 0; l < world_size - 1; l++){
  	  MPI_Recv(testArray, (n*n/world_size) , MPI_FLOAT , MPI_ANY_SOURCE , 0, MPI_COMM_WORLD ,&status);
  	  
  	  int source = status.MPI_SOURCE; 
  	  int k = 0;

  	  for(int i = source * number_of_rows; i < (source + 1) * number_of_rows; i++ ){
  	  	for(int j = 0; j < n; j++)
  	  	  A[i][j] = testArray[k * n + j];  //A[][] holds the final values
  	  	k++;
  	  }
  	}
  }
}
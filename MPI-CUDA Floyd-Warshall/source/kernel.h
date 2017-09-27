void executeKernelWithoutSharedMemory(int n, int number_of_rows , int world_rank, float *testArray);
void executeKernelWithSharedMemory(int n, int number_of_rows , int world_rank, float *testArray);
void executeKernelMulElemPerThread(int n, int number_of_rows , int world_rank, float *testArray);
void initializeCuda();
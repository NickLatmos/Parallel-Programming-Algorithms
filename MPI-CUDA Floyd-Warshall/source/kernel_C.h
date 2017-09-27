extern "C" void executeKernelWithoutSharedMemory(int n, int number_of_rows , int world_rank, float *testArray);
extern "C" void executeKernelWithSharedMemory(int n, int number_of_rows , int world_rank, float *testArray);
extern "C" void executeKernelMulElemPerThread(int n, int number_of_rows , int world_rank, float *testArray);
extern "C" void receiveRow(float *row_k, int n, int root);
extern "C" void initializeCuda();
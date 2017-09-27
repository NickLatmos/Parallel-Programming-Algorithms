extern float *linearArray; //linearArray will be the array to process
extern int world_size;  //number of processes

void makeAdjacency(int n,float p,int w);
void warshallFloydAlgorithm(int n);
void initialization(int n);
void initializationBlock(int n);
void checkResults(int n);
void copySerialArray(int n);
void printEverything(int n,int world_rank);
void sendBlocksToProcesses(int n, int number_of_rows);
void printMatrix_A(int n);
void buildFinalArray(int world_rank,int n,float *testArray, int number_of_rows);
void printCopiedArray(int n);
void printTestArray(int n, float *testArray);


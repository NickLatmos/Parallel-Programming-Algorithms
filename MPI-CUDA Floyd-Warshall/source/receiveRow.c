#include <mpi.h>

/*
 * This function receives the an array from process with world rank = root
 * or sends this array if the process is the onw with world_rank = root
 */
void receiveRow(float *row_k, int n, int root)
{
  MPI_Bcast (row_k, n, MPI_FLOAT, root, MPI_COMM_WORLD); //Even the sender receives this message
}
all: program

program: out1 out2 out3 out4 out5 out6 project3
	mpicc out1 out2 out3 out4 out5 out6 -o program -L/usr/local/cuda/lib64 -lcudart -lm

out1: mpi.c
	mpicc -c -std=c99 mpi.c -o out1
out2: kernelWithoutSharedMemory.cu
	nvcc -c kernelWithoutSharedMemory.cu -o out2
out4: kernelWithSharedMemory.cu
	nvcc -c kernelWithSharedMemory.cu -o out4
out5: executeKernelMulElemPerThread.cu
	nvcc -c executeKernelMulElemPerThread.cu -o out5
out3: receiveRow.c
	mpicc -c -std=c99 receiveRow.c -o out3
out6: CUDA_init.cu
	nvcc -c CUDA_init.cu -o out6
project3: project3.cu
	nvcc project3.cu -o project3 
clean:
	rm out1 out2 out3 out4 out5 out6 program project3
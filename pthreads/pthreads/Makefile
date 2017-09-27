CC=icc
CFLAGS=-O3 -ansi-alias -ipo -openmp -lpthread -c -ffat-lto-objects
LFLAGS=-O3 -ansi-alias -ipo -openmp -lpthread -ffat-lto-objects
OBJS= main.o bitonic_pthreads.o bitonic_openmp.o bitonic_cilk.o bitonic.o 
all: parallel_bitonic
	
parallel_bitonic: $(OBJS)
	$(CC) $(OBJS) $(LFLAGS) -o parallel_bitonic
main.o: main.c bitonic_pthreads.h bitonic_openmp.h bitonic.h
	$(CC) $(CFLAGS) main.c 
bitonic_pthreads.o: bitonic_pthreads.h bitonic_pthreads.c bitonic.h
	$(CC) $(CFLAGS) bitonic_pthreads.c
bitonic_openmp.o: bitonic_openmp.h bitonic_openmp.c bitonic.h
	$(CC) $(CFLAGS) bitonic_openmp.c 
bitonic_cilk.o: bitonic_cilk.h bitonic_cilk.c bitonic.h
	$(CC) $(CFLAGS) bitonic_cilk.c 
bitonic.o: bitonic.h bitonic.c
	$(CC) $(CFLAGS) bitonic.c
clean:
	rm *.o

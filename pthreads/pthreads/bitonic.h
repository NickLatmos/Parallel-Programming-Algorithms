#ifndef BITONIC_H
#define BITONIC_H
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <time.h>


struct data {
  int lo;
  int cnt;
  int dir;
  int problem_size;
};
struct timeval startwtime, endwtime;
double execute_time;
extern const int ASCENDING; // = 1;
extern const int DESCENDING; //= 0;

int N;          // data array size
int *a;         // data array to be sorted

int THREAD_LEVELS;
int PROBLEM_SIZE;





void init(void);
void print(void);
void sort(void);
void test(void);
inline void exchange(int i, int j);
void compare(int i, int j, int dir);
void bitonicMerge(int lo, int cnt, int dir);
void recBitonicSort(int lo, int cnt, int dir);
void impBitonicSort(void);

int asc( const void *p1, const void *p2 );
int desc( const void *p1, const void *p2 );

#endif

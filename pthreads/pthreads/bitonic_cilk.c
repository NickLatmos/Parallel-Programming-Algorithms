#include "bitonic_cilk.h"

void bitonic_sort_cilk(){
  int i,j,k=0;
  for (k = 2; k <= N; k *= 2 ) {
      for (j=k>>1; j>0; j=j>>1) {
          cilk_for (i=0; i<N; i++) {
            int ij=i^j;
            if ((ij)>i) {
                if ((i&k)==0 && a[i] > a[ij]) {
                    exchange(i,ij);
                  }
                if ((i&k)!=0 && a[i] < a[ij]){
                    exchange(i,ij);
                  }
              }
          }
        }
    }
}

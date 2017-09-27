/*This program solves the k nearest neighbours problem using MPI
(in this implementation it solves it for 1 neighbour).
!IMPORTANT! This code does not absolutely right for a big number of processes
and there have to be some changes.
Created by Nick Latmos*/

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <sys/time.h>
#include <mpi.h>
#include <stddef.h>

int NQ,NC,P,n,m,k;
int processes_per_plain,plains_per_process,number_of_processes_in_a_column;
int boxes_in_plain;
int number_of_boxes;
int world_size;

struct timeval startwtime, endwtime;
double seq_time;

MPI_Datatype mpi_point_type;

typedef struct point
{
    double x,y,z; 
    double min_distance;
}points_str;

struct box{
    points_str *points_insideBox;
    points_str *points_insideBox_in_C;
    struct neighbour_box *neighbours;
    double box_coordinates[3];
    double end_coordinates[3];
    int number_of_points;
    int number_of_points_in_C;
    int id;
    int number_of_neighbours;
    int process_id;
    int *neighbour_boxes_id;
};

struct neighbour_box{
  double coor[3];
  int process;
  int number_of_points_in_C;
  points_str *points_insideBox_in_C;
};

/*The functions are described later in the code*/
double uniform_distribution(double a, double b);
struct box *insertPointsInBox(struct point *nq,struct point *nc,struct box *boxes);
struct box *createBoxes(int world_rank);
struct box *SerialFindNeighbourId(struct box *boxes);
struct box *SerialFindNearestPoints(struct box *boxes);
struct box *findNearestPoints(struct box *boxes, int world_rank);
struct box *findNeighbourId(struct box *boxes,int world_rank);
void printBoxCoordinates(struct box *boxes);
void printPointsInsideTheBox(struct box *boxes);
void printNumberOfPointsInBox(struct box *boxes);
void printNumberOfNeighbours(struct box *boxes);
void printNearestPoints(struct box *boxes);
void printNeighourBoxProcess(struct box *boxes);
void receiver(struct box *boxes,int world_rank);
void createPoints(struct point *nq,struct point *nc,struct box *boxes);
void print_coordinates(struct point *nq,struct point *nc);

void main(int argc,char **argv){
  
  if(argc != 7)
  {
    printf("Usage bla bla");
    exit(1);
  }

    // Initialize the MPI environment
    MPI_Init(NULL, NULL);
    // Get the number of processes
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    // Get the rank of the process
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

  NQ = atoi(argv[1]);
  NC = atoi(argv[2]);
  P = atoi(argv[3]);
  n = atoi(argv[4]);
  m = atoi(argv[5]);
  k = atoi(argv[6]);
  
  NQ = pow(2,NQ);  
  NC = pow(2,NC);
  P = pow(2,P);
  n = pow(2,n);  //e.x 2^4 if n = 4
  m = pow(2,m);
  k = pow(2,k);
  
  number_of_boxes = n*m*k; // 1/(1/n*m*k) = n*m*k

  boxes_in_plain = m*k; // x=0,1,2,3...
  number_of_boxes = number_of_boxes/P;  //number of boxes per process
  processes_per_plain = (int) boxes_in_plain/number_of_boxes; 
  number_of_processes_in_a_column = k/number_of_boxes;
  
  NQ = NQ/P; //P processings with NQ/P points each process
  NC = NC/P; //P processings with NC/P points each process
  
  /*MPI struct data type*/
  const int count = 4; 
  int blocklengths[4] = {1,1,1,1};
    MPI_Datatype types[4] = {MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE};
    //MPI_Datatype mpi_point_type;
    MPI_Aint offsets[4];

  offsets[0] = offsetof(points_str, x);  //#include <stddef.h>,else it will give errors
    offsets[1] = offsetof(points_str, y);
    offsets[2] = offsetof(points_str, z);
    offsets[3] = offsetof(points_str, min_distance);
    MPI_Type_create_struct(count, blocklengths, offsets, types, &mpi_point_type);
    MPI_Type_commit(&mpi_point_type);

  points_str *nq,*nc;
  nq = (points_str *)malloc(NQ*sizeof(points_str));
  nc = (points_str *)malloc(NC*sizeof(points_str));
  if(nq == NULL || nc == NULL){
    printf("Not enough memory, exit");
    exit(1);
  }

  srand(time(NULL)); // randomize seed  

    struct box *boxes = createBoxes(world_rank);  //creates boxes

    createPoints(nq,nc,boxes);  //creates points
   
    gettimeofday( &startwtime, NULL );
    if(P != 1)
      boxes = findNeighbourId(boxes,world_rank);  //finds neighbouring boxes and the process they belong to
    else
      boxes = SerialFindNeighbourId(boxes);    //finds the neighbouring boxes and their ids in serial
 
    boxes = insertPointsInBox(nq,nc,boxes);   //inserts points in all boxes

    if(P != 1){
    boxes = findNearestPoints(boxes,world_rank);  //finds the minimum distance for each point. If no point is found in C then we set min_distance = -1;
    receiver(boxes,world_rank);           //creates a receiver
    //printf("after the receiver function world_rank = %d \n",world_rank);
    }
  else
    boxes = SerialFindNearestPoints(boxes);   //finds the minimum distance for each point in serial. If no point is found in C then we set min_distance = -1;

  //When it reaches this point it means that the work is done, no need to call MPI_BARRIER, because the receiver works like a ring, which means that when the 
  //last process is finished then we can continue the execution of the program.
  if(world_rank == 0){
    gettimeofday( &endwtime, NULL );
    seq_time = (double)( ( endwtime.tv_usec - startwtime.tv_usec ) / 1.0e6 + endwtime.tv_sec - startwtime.tv_sec );
    printf("Time with %d processes and %d points : = %f seconds\n",P,NQ*P, seq_time );
    printf("n = %d m = %d k = %d\n",n,m,k );
  }

    // Finalize the MPI environment.
    MPI_Finalize();
}

/*This function returns a double between a and b*/
double uniform_distribution(double a, double b) {
    double random = ((double) rand()) / (double) RAND_MAX;
    double diff = b - a;
    double r = random * diff;
    return a + r;
}

/*This function creates points in boxes which belong to a specific process and they are distributed uniformly*/
void createPoints(struct point *nq,struct point *nc,struct box *boxes){
  int i;
  for(i = 0;i < NQ;i++){
    nq[i].x = uniform_distribution(boxes[0].box_coordinates[0], boxes[number_of_boxes-1].end_coordinates[0]);
    nq[i].y = uniform_distribution(boxes[0].box_coordinates[1], boxes[number_of_boxes-1].end_coordinates[1]);
    nq[i].z = uniform_distribution(boxes[0].box_coordinates[2], boxes[number_of_boxes-1].end_coordinates[2]);
  }
  
  for(i = 0;i < NC;i++){
    nc[i].x = uniform_distribution(boxes[0].box_coordinates[0], boxes[number_of_boxes-1].end_coordinates[0]);
    nc[i].y = uniform_distribution(boxes[0].box_coordinates[1], boxes[number_of_boxes-1].end_coordinates[1]);
    nc[i].z = uniform_distribution(boxes[0].box_coordinates[2], boxes[number_of_boxes-1].end_coordinates[2]);
  }
}

/*This function prints the coordinates of all points in unit cube*/
void print_coordinates(struct point *nq,struct point *nc){
  int i;
  for(i = 0;i < NQ;i++){
    printf("point in Q %d coordinates : %lf %lf %lf\n",i,nq[i].x, nq[i].y ,nq[i].z);  
  }
  for(i = 0;i < NC;i++){
    printf("point in C  %d coordinates : %lf %lf %lf\n",i,nc[i].x, nc[i].y ,nc[i].z);
  }
}

/*This function inserts all the points of Q and C in the boxes created before */
struct box *insertPointsInBox(struct point *nq,struct point *nc,struct box *boxes){
    int counter = -1;
    double i,j,l;
    double nn = pow(n,-1);  //nn = 1/n
    double mm = pow(m,-1);
    double kk = pow(k,-1);
    int ii;
    for(i = boxes[0].box_coordinates[0]; i <= boxes[number_of_boxes-1].box_coordinates[0]; i+=nn){
      for(j = boxes[0].box_coordinates[1]; j <= boxes[number_of_boxes-1].box_coordinates[1]; j+=mm){
        for(l = boxes[0].box_coordinates[2]; l <= boxes[number_of_boxes-1].box_coordinates[2]; l+=kk){
        counter++;
        
          for(ii = 0; ii < NQ; ii++){
            if(nq[ii].x >= i && nq[ii].x < i+nn){
              if(nq[ii].y >= j && nq[ii].y < j+mm){
                if(nq[ii].z >= l && nq[ii].z < l+mm){
                  boxes[counter].number_of_points++;
                  if(boxes[counter].number_of_points == 1)
                    boxes[counter].points_insideBox = (struct point *)malloc(boxes[counter].number_of_points*sizeof(struct point));
                  else
                    boxes[counter].points_insideBox = (struct point *)realloc(boxes[counter].points_insideBox,boxes[counter].number_of_points*sizeof(struct point));
                  boxes[counter].points_insideBox[boxes[counter].number_of_points - 1] = nq[ii];
                }
                else
                  continue;
              }
              else
                continue;
            }
            else
              continue;
          }

          for(ii = 0; ii < NC; ii++){
            if(nc[ii].x >= i && nc[ii].x < i+nn){
              if(nc[ii].y >= j && nc[ii].y < j+mm){
                if(nc[ii].z >= l && nc[ii].z < l+mm){
                  boxes[counter].number_of_points_in_C++;
                  if(boxes[counter].number_of_points_in_C == 1)
                    boxes[counter].points_insideBox_in_C = (struct point *)malloc(boxes[counter].number_of_points_in_C*sizeof(struct point));
                  else
                    boxes[counter].points_insideBox_in_C = (struct point *)realloc(boxes[counter].points_insideBox_in_C,boxes[counter].number_of_points_in_C*sizeof(struct point));
                  boxes[counter].points_insideBox_in_C[boxes[counter].number_of_points_in_C - 1] = nc[ii];
                }
                else
                  continue;
              }
              else
                continue;
            }
            else
              continue;
          }
        }
      }
    }
    return boxes;
}

/*This function creates the boxes in unit cube*/
struct box *createBoxes(int world_rank){

  struct box *boxes = (struct box *)malloc(number_of_boxes*sizeof(struct box));
  if(boxes == NULL){
    printf("Not enough memory for the creation of boxes\n");
    exit(1);
  }
    double i,j,l;
    double temp;
    double nn = pow(n,-1);
    double mm = pow(m,-1);
    double kk = pow(k,-1);
    int ii = -1;
    int counter;
    int pos1;
    int y;

    if(processes_per_plain != 0 && P != 1){
      pos1 = (int) world_rank/processes_per_plain; 
      if(processes_per_plain == 1){
        for(j = 0;j < 1; j+=mm){
          for(l = 0; l < 1; l+=kk){
            ii++;
            /*Every process can be defined by box[0] which we will use as its start point in unit cube*/
            boxes[ii].id = ii;
            boxes[ii].box_coordinates[0] = pos1*nn;
            boxes[ii].box_coordinates[1] = j;
            boxes[ii].box_coordinates[2] = l;
            boxes[ii].number_of_points = 0;
            boxes[ii].number_of_points_in_C = 0;
            boxes[ii].number_of_neighbours = 0;
            boxes[ii].process_id = world_rank;
            if(ii == number_of_boxes - 1){
              boxes[ii].end_coordinates[0] = pos1*nn + nn;
              boxes[ii].end_coordinates[1] = j+mm;
              boxes[ii].end_coordinates[2] = l+kk;
            }
          }
        }
      }else{          //processes_per_plain > 1
        if(k == number_of_boxes){
          y = (int) world_rank%processes_per_plain;
          for(y*mm; y < (y+1)*mm; y+=mm){
            for(l = 0; l < 1; l+=kk){
              ii++;
              /*Every process can be defined by box[0] which we will use as its start point in unit cube*/
              boxes[ii].id = ii;
              boxes[ii].box_coordinates[0] = pos1*nn;
              boxes[ii].box_coordinates[1] = y;
              boxes[ii].box_coordinates[2] = l;
              boxes[ii].number_of_points = 0;
              boxes[ii].number_of_points_in_C = 0;
              boxes[ii].number_of_neighbours = 0;
              boxes[ii].process_id = world_rank;
              if(ii == number_of_boxes - 1){
                boxes[ii].end_coordinates[0] = pos1*nn + nn;
                boxes[ii].end_coordinates[1] = y+mm;
                boxes[ii].end_coordinates[2] = l+kk;
              }
            }
          }
        }else if(k < number_of_boxes){
          j = 1;
          while(j*k < number_of_boxes)
            j++;
          j = j*mm;
          y = (int) world_rank%processes_per_plain;
          for(temp = y*j; temp < (y+1)*j; temp+=mm){
            for(l = 0; l < 1; l+=kk){
            ii++;
            /*Every process can be defined by box[0] which we will use as its start point in unit cube*/
            boxes[ii].id = ii;
            boxes[ii].box_coordinates[0] = pos1*nn;
            boxes[ii].box_coordinates[1] = temp;
            boxes[ii].box_coordinates[2] = l;
              boxes[ii].number_of_points = 0;
              boxes[ii].number_of_points_in_C = 0;
              boxes[ii].number_of_neighbours = 0;
              boxes[ii].process_id = world_rank;
              if(ii == number_of_boxes - 1){
                boxes[ii].end_coordinates[0] = pos1*nn + nn;
                boxes[ii].end_coordinates[1] = temp+mm;
                boxes[ii].end_coordinates[2] = l+kk;
              }
            }
          }
        }else
          printf("I thought z > number_of_boxes was impossible\n");
      } 
    }
    else if(P != 1 ){
      plains_per_process = n/P;
      pos1 = world_rank;
      for(i = pos1*nn*plains_per_process; i < (pos1+1)*nn*plains_per_process; i+=nn ){
        for(j = 0; j < 1; j+=mm){
        for(l = 0; l < 1; l+=kk){
          ii++;
          /*Every process can be defined by box[0] which we will use as its start point in unit cube*/
          boxes[ii].id = ii;
          boxes[ii].box_coordinates[0] = i;
          boxes[ii].box_coordinates[1] = j;
          boxes[ii].box_coordinates[2] = l;
            boxes[ii].number_of_points = 0;
            boxes[ii].number_of_points_in_C = 0;
            boxes[ii].number_of_neighbours = 0;
            boxes[ii].process_id = world_rank;
            if(ii == number_of_boxes - 1){
            boxes[ii].end_coordinates[0] = i+nn;
            boxes[ii].end_coordinates[1] = j+mm;
              boxes[ii].end_coordinates[2] = l+kk;
            }
        }
      }
      }
    }else{            //P == 1, serial algorithm
      for(i = 0;i < 1; i+=nn){
      for(j = 0;j < 1; j+=mm){
        for(l = 0;l < 1; l+=kk){
           ii++;
           boxes[ii].id = ii; 
           boxes[ii].box_coordinates[0] = i;
           boxes[ii].box_coordinates[1] = j;
           boxes[ii].box_coordinates[2] = l;
           boxes[ii].number_of_points = 0;
           boxes[ii].number_of_points_in_C = 0;
           boxes[ii].number_of_neighbours = 0;
              if(ii == number_of_boxes - 1){
            boxes[ii].end_coordinates[0] = i+nn;
            boxes[ii].end_coordinates[1] = j+mm;
              boxes[ii].end_coordinates[2] = l+kk;
          }
        }
      }
    }
    }

    return boxes;

}

void printNeighourBoxProcess(struct box *boxes){
  int i,j;
  for(i = 0; i < number_of_boxes; i++){
    for(j = 0; j < boxes[i].number_of_neighbours; j++){
      if(i == 255)
        printf("a = %lf b = %lf c = %lf\n",boxes[i].neighbours[j].coor[0],boxes[i].neighbours[j].coor[1],boxes[i].neighbours[j].coor[2] );
      printf("box[%d] neighbour %d rank  = %d\n",i,j,boxes[i].neighbours[j].process);
    }
  }
}

/*This function prints the coordinates of each box. We assume that the coordinates of a box are the coordinates of its back left corner*/
void printBoxCoordinates(struct box *boxes){
    int ii;
  for(ii = 0;ii < number_of_boxes; ii++){
    printf("coordinates of box[%d] = %lf  %lf  %lf\n",ii,boxes[ii].box_coordinates[0], boxes[ii].box_coordinates[1], boxes[ii].box_coordinates[2]); 
  }
}

/*This function prints the number of points that are inside of each box*/ 
void printNumberOfPointsInBox(struct box *boxes){
    int ii;
  for(ii = 0;ii < number_of_boxes; ii++){
    printf("number of points that belong in Q in box[%d] = %d\n",ii,boxes[ii].number_of_points);  
  }
  for(ii = 0;ii < number_of_boxes; ii++){
    printf("number of points that belong in C in box[%d] = %d\n",ii,boxes[ii].number_of_points_in_C); 
  }
}

/*This function prints the coordinates of points that are inside of each box*/ 
void printPointsInsideTheBox(struct box *boxes){
  int i,j;
  for(i = 0;i < number_of_boxes; i++){
    for(j = 1;j <= boxes[i].number_of_points;j++)
      printf("Coordinates of point %d in box[%d] are %lf  %lf  %lf :\n",j,i,boxes[i].points_insideBox[j-1].x, boxes[i].points_insideBox[j-1].y,boxes[i].points_insideBox[j-1].z);
  }
}

/*This function finds the id of the neighbour boxes via their coordinates*/
struct box *findNeighbourId(struct box *boxes,int world_rank){

    double i,j,l,a,b,c;
    double cox,coy,coz;
    double nn = pow(n,-1);
    double mm = pow(m,-1);
    double kk = pow(k,-1);
    int ii = -1;
    int mycounter = -1;
    int counter;
    int flag = 0;
  for(i = boxes[0].box_coordinates[0] ;i <= boxes[number_of_boxes-1].box_coordinates[0]; i+=nn){
    for(j = boxes[0].box_coordinates[1] ;j <= boxes[number_of_boxes-1].box_coordinates[1] ; j+=mm){
      for(l = boxes[0].box_coordinates[2]; l <= boxes[number_of_boxes-1].box_coordinates[2]; l+=kk){
        ii++;

        /*This is a method to determine in which process a neighbour box belongs to via its coordinates*/
        for(a = boxes[ii].box_coordinates[0]-nn; a <= boxes[ii].box_coordinates[0]+nn; a+=nn){
          for(b = boxes[ii].box_coordinates[1]-mm; b <= boxes[ii].box_coordinates[1]+mm; b+=mm){
            for(c = boxes[ii].box_coordinates[2]-kk; c <= boxes[ii].box_coordinates[2]+kk; c+=kk){
              
              if(a == boxes[ii].box_coordinates[0] && b == boxes[ii].box_coordinates[1] && c == boxes[ii].box_coordinates[2])
                continue;
              if(a == -nn || a == 1 || b == -mm || b == 1 || c == -kk || c == 1)
                continue;
              
              boxes[ii].number_of_neighbours++;
              if(boxes[ii].number_of_neighbours == 1)
                boxes[ii].neighbours = (struct neighbour_box *)malloc(sizeof(struct neighbour_box));
              else
                boxes[ii].neighbours = (struct neighbour_box *)realloc(boxes[ii].neighbours,boxes[ii].number_of_neighbours*sizeof(struct neighbour_box)); 
              boxes[ii].neighbours[boxes[ii].number_of_neighbours-1].coor[0] = a;
              boxes[ii].neighbours[boxes[ii].number_of_neighbours-1].coor[1] = b;
              boxes[ii].neighbours[boxes[ii].number_of_neighbours-1].coor[2] = c;
              boxes[ii].neighbours[boxes[ii].number_of_neighbours-1].points_insideBox_in_C = (points_str *)malloc(8500*sizeof(points_str));
              
              if(a >= boxes[0].box_coordinates[0] && a <= boxes[number_of_boxes-1].box_coordinates[0] && b >= boxes[0].box_coordinates[1] && b <= boxes[number_of_boxes-1].box_coordinates[1]
                && c >= boxes[0].box_coordinates[2] && c <= boxes[number_of_boxes-1].box_coordinates[2] ){
                //it belongs in this process
                boxes[ii].neighbours[boxes[ii].number_of_neighbours-1].process = world_rank;
              }else if(processes_per_plain > 1 && number_of_processes_in_a_column != 0){
                //the neighbour belongs to another process
                //now we have to find the process'es rank
                if(a > boxes[0].box_coordinates[0] ){
                  if(b > boxes[number_of_boxes-1].box_coordinates[1] ){
                    if(c > boxes[number_of_boxes-1].box_coordinates[2] )
                      boxes[ii].neighbours[boxes[ii].number_of_neighbours-1].process = number_of_processes_in_a_column + 1 + world_rank + processes_per_plain;
                    else if(c >= boxes[0].box_coordinates[2] && c <= boxes[number_of_boxes-1].box_coordinates[2])
                      boxes[ii].neighbours[boxes[ii].number_of_neighbours-1].process = number_of_processes_in_a_column + world_rank + processes_per_plain;
                    else
                      boxes[ii].neighbours[boxes[ii].number_of_neighbours-1].process = number_of_processes_in_a_column - 1 + world_rank + processes_per_plain;
                  }else if(b >= boxes[0].box_coordinates[1] && b <= boxes[number_of_boxes-1].box_coordinates[1] ){
                    if(c > boxes[number_of_boxes-1].box_coordinates[2] )
                      boxes[ii].neighbours[boxes[ii].number_of_neighbours-1].process = 1 + world_rank + processes_per_plain;
                    else if(c >= boxes[0].box_coordinates[2] && c <= boxes[number_of_boxes-1].box_coordinates[2] )
                      boxes[ii].neighbours[boxes[ii].number_of_neighbours-1].process = world_rank + processes_per_plain;
                    else
                      boxes[ii].neighbours[boxes[ii].number_of_neighbours-1].process = -1 + world_rank + processes_per_plain;
                  }else{
                    if(c > boxes[number_of_boxes-1].box_coordinates[2] )
                      boxes[ii].neighbours[boxes[ii].number_of_neighbours-1].process = world_rank - (number_of_processes_in_a_column - 1) + processes_per_plain;
                    else if(c >= boxes[0].box_coordinates[2] && c <= boxes[number_of_boxes-1].box_coordinates[2])
                      boxes[ii].neighbours[boxes[ii].number_of_neighbours-1].process = world_rank - (number_of_processes_in_a_column ) + processes_per_plain;
                    else
                      boxes[ii].neighbours[boxes[ii].number_of_neighbours-1].process = world_rank - (number_of_processes_in_a_column + 1) + processes_per_plain;
                  }
                }else if(a == boxes[0].box_coordinates[0] ){
                  if(b > boxes[number_of_boxes-1].box_coordinates[1] ){
                    if(c > boxes[number_of_boxes-1].box_coordinates[2] )
                      boxes[ii].neighbours[boxes[ii].number_of_neighbours-1].process = number_of_processes_in_a_column + 1 + world_rank;
                    else if(c >= boxes[0].box_coordinates[2] && c <= boxes[number_of_boxes-1].box_coordinates[2])
                      boxes[ii].neighbours[boxes[ii].number_of_neighbours-1].process = number_of_processes_in_a_column + world_rank;
                    else
                      boxes[ii].neighbours[boxes[ii].number_of_neighbours-1].process = number_of_processes_in_a_column - 1 + world_rank;
                  }else if(b >= boxes[0].box_coordinates[1] && b <= boxes[number_of_boxes-1].box_coordinates[1] ){
                    if(c > boxes[number_of_boxes-1].box_coordinates[2] )
                      boxes[ii].neighbours[boxes[ii].number_of_neighbours-1].process = 1 + world_rank;
                    else if(c >= boxes[0].box_coordinates[2] && c <= boxes[number_of_boxes-1].box_coordinates[2])
                      boxes[ii].neighbours[boxes[ii].number_of_neighbours-1].process = world_rank;
                    else
                      boxes[ii].neighbours[boxes[ii].number_of_neighbours-1].process = -1 + world_rank;
                  }else{
                    if(c > boxes[number_of_boxes-1].box_coordinates[2] )
                      boxes[ii].neighbours[boxes[ii].number_of_neighbours-1].process = world_rank - (number_of_processes_in_a_column - 1);
                    else if(c >= boxes[0].box_coordinates[2] && c <= boxes[number_of_boxes-1].box_coordinates[2])
                      boxes[ii].neighbours[boxes[ii].number_of_neighbours-1].process = world_rank - (number_of_processes_in_a_column );
                    else
                      boxes[ii].neighbours[boxes[ii].number_of_neighbours-1].process = world_rank - (number_of_processes_in_a_column + 1);
                  }
                }else{
                  if(b > boxes[number_of_boxes-1].box_coordinates[1] ){
                    if(c > boxes[number_of_boxes-1].box_coordinates[2] )
                      boxes[ii].neighbours[boxes[ii].number_of_neighbours-1].process = number_of_processes_in_a_column + 1 + world_rank - processes_per_plain;
                    else if(c >= boxes[0].box_coordinates[2] && c <= boxes[number_of_boxes-1].box_coordinates[2])
                      boxes[ii].neighbours[boxes[ii].number_of_neighbours-1].process = number_of_processes_in_a_column + world_rank - processes_per_plain;
                    else
                      boxes[ii].neighbours[boxes[ii].number_of_neighbours-1].process = number_of_processes_in_a_column - 1 + world_rank - processes_per_plain;
                  }else if(b >= boxes[0].box_coordinates[1] && b <= boxes[number_of_boxes-1].box_coordinates[1]){
                    if(c > boxes[number_of_boxes-1].box_coordinates[2] )
                      boxes[ii].neighbours[boxes[ii].number_of_neighbours-1].process = 1 + world_rank - processes_per_plain;
                    else if(c >= boxes[0].box_coordinates[2] && c <= boxes[number_of_boxes-1].box_coordinates[2])
                      boxes[ii].neighbours[boxes[ii].number_of_neighbours-1].process = world_rank - processes_per_plain;
                    else
                      boxes[ii].neighbours[boxes[ii].number_of_neighbours-1].process = -1 + world_rank - processes_per_plain;
                  }else{
                    if(c > boxes[number_of_boxes-1].box_coordinates[2] )
                      boxes[ii].neighbours[boxes[ii].number_of_neighbours-1].process = world_rank - (number_of_processes_in_a_column - 1) - processes_per_plain;
                    else if(c >= boxes[0].box_coordinates[2] && c <= boxes[number_of_boxes-1].box_coordinates[2])
                      boxes[ii].neighbours[boxes[ii].number_of_neighbours-1].process = world_rank - (number_of_processes_in_a_column ) - processes_per_plain;
                    else
                      boxes[ii].neighbours[boxes[ii].number_of_neighbours-1].process = world_rank - (number_of_processes_in_a_column + 1) - processes_per_plain;
                  }
                }
              }else if(number_of_processes_in_a_column == 0 && processes_per_plain > 1){
                //the neighbour belongs to another process
                //now we have to find the process'es rank
                if(a > boxes[ii].box_coordinates[0] ){
                  if(b > boxes[ii].box_coordinates[1] ){
                    if( b >= boxes[0].box_coordinates[1] && b <= boxes[number_of_boxes-1].box_coordinates[1] )
                      boxes[ii].neighbours[boxes[ii].number_of_neighbours-1].process = world_rank + processes_per_plain ;
                    else
                      boxes[ii].neighbours[boxes[ii].number_of_neighbours-1].process = world_rank + processes_per_plain + 1;
                  }else if(b == boxes[ii].box_coordinates[1])
                    boxes[ii].neighbours[boxes[ii].number_of_neighbours-1].process = world_rank  + processes_per_plain;
                  else{
                    if(b >= boxes[0].box_coordinates[1] && b <= boxes[number_of_boxes-1].box_coordinates[1] )
                      boxes[ii].neighbours[boxes[ii].number_of_neighbours-1].process = world_rank + processes_per_plain ;
                    else
                      boxes[ii].neighbours[boxes[ii].number_of_neighbours-1].process = world_rank  + processes_per_plain - 1; 
                  }
                }else if(a == boxes[ii].box_coordinates[0] ){
                  if(b > boxes[ii].box_coordinates[1] )
                    boxes[ii].neighbours[boxes[ii].number_of_neighbours-1].process = world_rank + 1;
                  else if(b == boxes[ii].box_coordinates[1])
                    boxes[ii].neighbours[boxes[ii].number_of_neighbours-1].process = world_rank;
                  else
                    boxes[ii].neighbours[boxes[ii].number_of_neighbours-1].process = world_rank - 1;  
                }else{
                  if(b > boxes[ii].box_coordinates[1] ){
                    if( b >= boxes[0].box_coordinates[1] && b <= boxes[number_of_boxes-1].box_coordinates[1] )
                      boxes[ii].neighbours[boxes[ii].number_of_neighbours-1].process = world_rank - processes_per_plain ;
                    else
                      boxes[ii].neighbours[boxes[ii].number_of_neighbours-1].process = world_rank - processes_per_plain + 1;
                  }else if(b == boxes[ii].box_coordinates[1])
                    boxes[ii].neighbours[boxes[ii].number_of_neighbours-1].process = world_rank  - processes_per_plain;
                  else{
                    if(b >= boxes[0].box_coordinates[1] && b <= boxes[number_of_boxes-1].box_coordinates[1] )
                      boxes[ii].neighbours[boxes[ii].number_of_neighbours-1].process = world_rank - processes_per_plain ;
                    else
                      boxes[ii].neighbours[boxes[ii].number_of_neighbours-1].process = world_rank  - processes_per_plain - 1; 
                  }
                }
              }else if(processes_per_plain == 1 || plains_per_process > 1){
                //the neighbour belongs to another process
                //now we have to find the process'es rank
                if(a < boxes[ii].box_coordinates[0])
                  boxes[ii].neighbours[boxes[ii].number_of_neighbours-1].process = world_rank - 1;
                else if(a == boxes[ii].box_coordinates[0])
                  boxes[ii].neighbours[boxes[ii].number_of_neighbours-1].process = world_rank;
                else
                  boxes[ii].neighbours[boxes[ii].number_of_neighbours-1].process = world_rank + 1;
              }
            }
          }
        }
      }
    }
  }
  return boxes;
}

void printNumberOfNeighbours(struct box *boxes){
  int i;
  for(i = 0;i < number_of_boxes;i++){
    printf("box[%d] number of neibours: %d\n",i,boxes[i].number_of_neighbours );
  }
}

/*This functions finds the nearest point belonging in C for each point belonging in Q*/
struct box *findNearestPoints(struct box *boxes, int world_rank){

  double i,j,l,a,b,c,final_counter;
    double cox,coy,coz,cox2,coy2,coz2,distance,distance1,min,min1;
    double x,y,z;
    double nn = pow(n,-1);
    double mm = pow(m,-1);
    double kk = pow(k,-1);
    struct point *temp,*temp1;
    int ii = -1;
    int jj = -1,flag = 0;
    int ll = -1;
    int counter,counter1,counter2,counter3,counter4;
    MPI_Status status,status1;
    MPI_Request request;
    int flag1 = 0,flag2,flag3,flag4;
  double *p = (double *)malloc(3*sizeof(double)); 
  double pl[3];

    for(i = boxes[0].box_coordinates[0]; i <= boxes[number_of_boxes-1].box_coordinates[0]; i+=nn){
      for(j = boxes[0].box_coordinates[1]; j <= boxes[number_of_boxes-1].box_coordinates[1]; j+=mm){
        for(l = boxes[0].box_coordinates[2]; l <= boxes[number_of_boxes-1].box_coordinates[2] ;l+=kk){
          ii++;

          //first of all we will try to find the nearest points of C for each p that belongs to this box. After that we will search in the neighbouring boxes
          if(boxes[ii].number_of_points != 0){
            for(counter = 0; counter < boxes[ii].number_of_points; counter++){
              cox = boxes[ii].points_insideBox[counter].x;
              coy = boxes[ii].points_insideBox[counter].y;
              coz = boxes[ii].points_insideBox[counter].z;
              min = 10000;
              min1 = 10000;
              distance = 0;
              distance1 = 0;
              if(boxes[ii].number_of_points_in_C != 0 ){
                for(counter1 = 0; counter1 < boxes[ii].number_of_points_in_C; counter1++){
                  cox2 = boxes[ii].points_insideBox_in_C[counter1].x;
                  coy2 = boxes[ii].points_insideBox_in_C[counter1].y;
                  coz2 = boxes[ii].points_insideBox_in_C[counter1].z;
                  x = pow(cox - cox2,2);
                  y = pow(coy - coy2,2);
                  z = pow(coz - coz2,2);
                  distance = sqrt(x+y+z);
                  if(distance < min)
                    min = distance;
                }
              }  

              for(counter1 = 0; counter1 < boxes[ii].number_of_neighbours; counter1++){
                
                //We have to communicate with boxes[ii].neighbours[counter1].process in order to receive the points in C.
                p[0] = boxes[ii].neighbours[counter1].coor[0];
                p[1] = boxes[ii].neighbours[counter1].coor[1];
                p[2] = boxes[ii].neighbours[counter1].coor[2];

                //if the neighbour box belongs in this process then we don't have to communicate with another process
                if(boxes[ii].neighbours[counter1].process == world_rank){
                  jj = -1;
                for(a = boxes[0].box_coordinates[0]; a <= boxes[number_of_boxes-1].box_coordinates[0]; a += nn){
                  for(b = boxes[0].box_coordinates[1]; b <= boxes[number_of_boxes-1].box_coordinates[1]; b += mm){
                    for(c = boxes[0].box_coordinates[2]; c <= boxes[number_of_boxes-1].box_coordinates[2]; c += kk){
                      jj++;
                      if(a == p[0] && b == p[1] && c == p[2] ){
                        flag = 1;
                        break;
                      }
                    }
                    if(flag == 1)
                      break;
                  }
                  if(flag == 1)
                    break;
                }
                boxes[ii].neighbours[counter1].number_of_points_in_C = boxes[jj].number_of_points_in_C;
                boxes[ii].neighbours[counter1].points_insideBox_in_C = boxes[jj].points_insideBox_in_C;

                //listening if another process asks for a box, and if true then send its points and number of points in C to that process
                MPI_Iprobe( MPI_ANY_SOURCE, 1, MPI_COMM_WORLD, &flag1, &status);
                  while(flag1 == 1){
                    MPI_Irecv(p, 3 , MPI_DOUBLE , MPI_ANY_SOURCE, 1, MPI_COMM_WORLD ,&request); //receives a box from another process if there is one
                  MPI_Wait(&request, &status);
                  ll = -1;
                  flag = 0;
                  for(a = boxes[0].box_coordinates[0]; a <= boxes[number_of_boxes-1].box_coordinates[0]; a += nn){
                    for(b = boxes[0].box_coordinates[1]; b <= boxes[number_of_boxes-1].box_coordinates[1]; b += mm){
                      for(c = boxes[0].box_coordinates[2]; c <= boxes[number_of_boxes-1].box_coordinates[2]; c += kk){
                        ll++;
                        if(a == p[0] && b == p[1] && c == p[2] ){
                          flag = 1;
                          break;
                        }
                      }
                      if(flag == 1)
                        break;
                    }
                    if(flag == 1)
                      break;
                  }
                  if(flag == 0){
                    printf("BOX NOT FOUND. PLEASE ABORT THE EXECUTION function -findNearestPoints- in IF world_rank = %d\n",world_rank );
                    exit(1);
                  }
                  MPI_Send(boxes[ll].points_insideBox_in_C , boxes[ll].number_of_points_in_C , mpi_point_type , status.MPI_SOURCE, 0 ,MPI_COMM_WORLD); //send the points in C to another process
                    MPI_Send(&boxes[ll].number_of_points_in_C , 1 , MPI_INT , status.MPI_SOURCE, 2 ,MPI_COMM_WORLD); //send the number of points in C to another process
                  MPI_Iprobe( MPI_ANY_SOURCE, 1, MPI_COMM_WORLD, &flag1, &status );
                  }
                }else{
                  MPI_Send(p, 3 , MPI_DOUBLE , boxes[ii].neighbours[counter1].process, 1 , MPI_COMM_WORLD); //send the neigbour's coordinates to another process

                MPI_Iprobe( boxes[ii].neighbours[counter1].process, 0, MPI_COMM_WORLD, &flag3, &status );
                MPI_Iprobe( boxes[ii].neighbours[counter1].process, 2, MPI_COMM_WORLD, &flag4, &status );
                MPI_Iprobe( MPI_ANY_SOURCE, 1, MPI_COMM_WORLD, &flag1, &status );
                while(1){
              
                  if(flag1 == 0){
                    MPI_Iprobe( MPI_ANY_SOURCE, 1, MPI_COMM_WORLD, &flag1, &status );
                    if(flag3 == 0){
                      if(flag4 == 0){
                        MPI_Iprobe( boxes[ii].neighbours[counter1].process, 0, MPI_COMM_WORLD, &flag3, &status );
                        MPI_Iprobe( boxes[ii].neighbours[counter1].process, 2, MPI_COMM_WORLD, &flag4, &status );
                      }else
                        MPI_Iprobe( boxes[ii].neighbours[counter1].process, 0, MPI_COMM_WORLD, &flag3, &status );
                    }else{
                      if(flag4 == 0)
                        MPI_Iprobe( boxes[ii].neighbours[counter1].process, 2, MPI_COMM_WORLD, &flag4, &status );
                      else
                        break;
                    }
                  }else{
                    while(flag1 == 1){
                        MPI_Irecv(pl, 3 , MPI_DOUBLE , MPI_ANY_SOURCE, 1, MPI_COMM_WORLD ,&request);  //receives a box from another process
                        MPI_Wait(&request, &status);
                        jj = -1;
                        flag = 0;
                      for(a = boxes[0].box_coordinates[0]; a <= boxes[number_of_boxes-1].box_coordinates[0]; a += nn){
                        for(b = boxes[0].box_coordinates[1]; b <= boxes[number_of_boxes-1].box_coordinates[1]; b += mm){
                          for(c = boxes[0].box_coordinates[2]; c <= boxes[number_of_boxes-1].box_coordinates[2]; c += kk){
                            jj++;
                            if(a == pl[0] && b == pl[1] && c == pl[2] ){
                              flag = 1;
                              break;
                            }
                          }
                          if(flag == 1)
                            break;
                        }
                        if(flag == 1)
                          break;
                      }
                      if(flag == 0){
                        printf("BOX NOT FOUND. PLEASE ABORT THE EXECUTION function -findNearestPoints- in second else world_rank = %d, communicated with process = %d\n",world_rank, status.MPI_SOURCE );
                        exit(1);
                      }
                      MPI_Send(boxes[jj].points_insideBox_in_C , boxes[jj].number_of_points_in_C , mpi_point_type , status.MPI_SOURCE, 0 ,MPI_COMM_WORLD); //send the points in C to another process
                      MPI_Send(&boxes[jj].number_of_points_in_C , 1, MPI_INT , status.MPI_SOURCE, 2 ,MPI_COMM_WORLD); // send the number of points in C to another process
                      MPI_Iprobe( MPI_ANY_SOURCE, 1, MPI_COMM_WORLD, &flag1, &status );
                    }
                  }
                }
                MPI_Recv(boxes[ii].neighbours[counter1].points_insideBox_in_C, 8500 , mpi_point_type , boxes[ii].neighbours[counter1].process , 0, MPI_COMM_WORLD ,MPI_STATUS_IGNORE); // receives the points in C from the box we sent in MPI_Send with tag = 0. ATTENTION I use count = 1000 because I don't know the exact number of points.
                MPI_Recv(&boxes[ii].neighbours[counter1].number_of_points_in_C, 1 , MPI_INT , boxes[ii].neighbours[counter1].process , 2, MPI_COMM_WORLD ,MPI_STATUS_IGNORE);
                }

                if(boxes[ii].neighbours[counter1].number_of_points_in_C != 0 ){
                  for(counter2 = 0;counter2 < boxes[ii].neighbours[counter1].number_of_points_in_C; counter2++){  //counter2 counts the points in C belonging to each neighbour
                    cox2 = boxes[ii].neighbours[counter1].points_insideBox_in_C[counter2].x;
                    coy2 = boxes[ii].neighbours[counter1].points_insideBox_in_C[counter2].y;
                    coz2 = boxes[ii].neighbours[counter1].points_insideBox_in_C[counter2].z;
                    x = pow(cox - cox2,2);
                    y = pow(coy - coy2,2);
                    z = pow(coz - coz2,2);
                    distance1 = sqrt(x+y+z);
                    if(distance1 < min1)
                      min1 = distance1;
                  }
                }else
                  min1 = - 1; //no points in C

                if(min1 == -1 && min == 10000){
                  boxes[ii].points_insideBox[counter].min_distance = -1;
                }
                else{
                if(min < min1)
                  boxes[ii].points_insideBox[counter].min_distance = min;
                  else
                  boxes[ii].points_insideBox[counter].min_distance = min1;
              }
            }
            }
          }
        }
      }
    }
  
    return boxes;
}

/*Prints the minimum distance for each point in Q from its neighbour in C. */
void printNearestPoints(struct box *boxes){
  int i,j;
  for(i = 0;i < number_of_boxes; i++){
    for(j = 0; j < boxes[i].number_of_points;j++){
      printf("box[%d] point [%d] minimum distance = %lf\n",i,j,boxes[i].points_insideBox[j].min_distance);
    }
  }
}

/*This function receives requests from other processes, and when the requests are over then it returns to the main function*/
void receiver(struct box *boxes,int world_rank){
  
  int flag,flag1,flag2,flag3;
    double nn = pow(n,-1);
    double mm = pow(m,-1);
    double kk = pow(k,-1);
    double a,b,c;
    int ll = -1;
    int counter = 0;

    int token;
    double *p = (double *)malloc(3*sizeof(double)); 
    MPI_Request request;
    MPI_Status status,status1;
   
    while(1){

    MPI_Iprobe( MPI_ANY_SOURCE, 1, MPI_COMM_WORLD, &flag1, &status );  
    while(flag1 == 1){
       
        MPI_Irecv(p, 3 , MPI_DOUBLE , MPI_ANY_SOURCE, 1, MPI_COMM_WORLD ,&request); //receives a box from another process if there is one
      MPI_Wait(&request, &status);  
      ll = -1;
      flag = 0;
      for(a = boxes[0].box_coordinates[0]; a <= boxes[number_of_boxes-1].box_coordinates[0]; a += nn){
        for(b = boxes[0].box_coordinates[1]; b <= boxes[number_of_boxes-1].box_coordinates[1]; b += mm){
          for(c = boxes[0].box_coordinates[2]; c <= boxes[number_of_boxes-1].box_coordinates[2]; c += kk){
            ll++;
            if(a == p[0] && b == p[1] && c == p[2] ){
              flag = 1;
              break;
            }
          }
          if(flag == 1)
            break;
        }
        if(flag == 1)
          break;
      }
      if(flag == 0){
        printf("POINT NOT FOUND. PLEASE ABORT THE EXECUTION function -receiver- world_rank = %d\n",world_rank );  //impossible to happen
        exit(1);
      }
      MPI_Send(boxes[ll].points_insideBox_in_C , boxes[ll].number_of_points_in_C , mpi_point_type , status.MPI_SOURCE, 0 ,MPI_COMM_WORLD); //send the points in C to another process
      MPI_Send(&boxes[ll].number_of_points_in_C , 1 , MPI_INT , status.MPI_SOURCE, 2 ,MPI_COMM_WORLD); //send the number of points in C to another process
      MPI_Iprobe( MPI_ANY_SOURCE, 1, MPI_COMM_WORLD, &flag1, &status ); 
    }
    
    // Receive from the lower process and send to the higher process. Take care
    // of the special case when you are the first process to prevent deadlock.
    if (world_rank != 0) {
      MPI_Iprobe(world_rank-1, 3, MPI_COMM_WORLD, &flag3, &status);
      if(flag3 == 1){
        MPI_Recv(&token, 1, MPI_INT, world_rank - 1, 3, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Send(&token, 1, MPI_INT, (world_rank + 1) % world_size, 3, MPI_COMM_WORLD);
        if(token == -2)
          break;
      }
    } else {
      if(counter == 0)
        token = -1;  //this will be executed only the first time
    }
    
    // Now process 0 can receive from the last process. This makes sure that at
    // least one MPI_Send is initialized before all MPI_Recvs (again, to prevent
    // deadlock)
    if (world_rank == 0) {
      
      if(counter == 0){
        MPI_Send(&token, 1, MPI_INT, (world_rank + 1) % world_size, 3, MPI_COMM_WORLD);
        counter++;
      }
      
      MPI_Iprobe(world_size-1, 3, MPI_COMM_WORLD, &flag3, &status);
      if(flag3 == 1){
        MPI_Recv(&token, 1, MPI_INT, world_size-1, 3, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        token = -2;
        MPI_Send(&token, 1, MPI_INT, (world_rank + 1) % world_size, 3, MPI_COMM_WORLD);
        break;
      }
    }
  }
}

/*This function finds the id of the neighbour boxes via their coordinates*/
struct box *SerialFindNeighbourId(struct box *boxes){

    double i,j,l,a,b,c;
    double cox,coy,coz;
    double nn = pow(n,-1);
    double mm = pow(m,-1);
    double kk = pow(k,-1);
    int ii = -1,jj = -1;
    int mycounter = -1;
    int counter;
    int flag = 0,flag1;
  for(i = 0;i < 1; i+=nn){
    for(j = 0;j < 1; j+=mm){
      for(l = 0;l < 1; l+=kk){
        ii++;

          /*We find the neighbour's coordinates*/
        for(a = boxes[ii].box_coordinates[0]-nn; a <= boxes[ii].box_coordinates[0]+nn; a+=nn){
          for(b = boxes[ii].box_coordinates[1]-mm; b <= boxes[ii].box_coordinates[1]+mm; b+=mm){
            for(c = boxes[ii].box_coordinates[2]-kk; c <= boxes[ii].box_coordinates[2]+kk; c+=kk){
              if(a == boxes[ii].box_coordinates[0] && b == boxes[ii].box_coordinates[1] && c == boxes[ii].box_coordinates[2])
                continue;
              
              if(a == -nn || a == 1 || b == -mm || b == 1 || c == -kk || c == 1)
                continue;
              
              boxes[ii].number_of_neighbours++;
              if(boxes[ii].number_of_neighbours == 1){
                boxes[ii].neighbours = (struct neighbour_box *)malloc(sizeof(struct neighbour_box));
                boxes[ii].neighbour_boxes_id = (int *)malloc(sizeof(int));
              }
              else{
                boxes[ii].neighbours = (struct neighbour_box *)realloc(boxes[ii].neighbours,boxes[ii].number_of_neighbours*sizeof(struct neighbour_box));
                boxes[ii].neighbour_boxes_id = (int *)realloc(boxes[ii].neighbour_boxes_id,boxes[ii].number_of_neighbours*sizeof(int)); 
              }
              boxes[ii].neighbours[boxes[ii].number_of_neighbours-1].coor[0] = a;
              boxes[ii].neighbours[boxes[ii].number_of_neighbours-1].coor[1] = b;
              boxes[ii].neighbours[boxes[ii].number_of_neighbours-1].coor[2] = c;

              //Time to find the ids of the neighbour boxes
              jj = -1;
              flag1 = 0;
              for(cox = 0;cox < 1; cox+=nn){
                for(coy = 0;coy < 1; coy+=mm){
                  for(coz = 0;coz < 1; coz+=kk){
                    jj++;
                    if(boxes[ii].neighbours[boxes[ii].number_of_neighbours-1].coor[0] == cox && boxes[ii].neighbours[boxes[ii].number_of_neighbours-1].coor[1] == coy 
                      && boxes[ii].neighbours[boxes[ii].number_of_neighbours-1].coor[2] == coz ){

                      boxes[ii].neighbour_boxes_id[boxes[ii].number_of_neighbours-1] = boxes[jj].id;
                      flag1 = 1;
                      break;
                    }
                  }
                  if(flag1 == 1)
                    break;
                }
                if(flag1 == 1)
                  break;
              }
              if(flag1 != 1){
                printf("Neigbour box'es coordinates not found\nABORT EXECUTION\n");
                exit(1);
              }
            }
          }
          }
      }
    }
  }
  return boxes;
}

/*This functions finds the nearest point belonging in C for each point belonging in Q*/
struct box *SerialFindNearestPoints(struct box *boxes){

  double i,j,l;
    double cox,coy,coz,cox2,coy2,coz2,distance,distance1,min,min1;
    double x,y,z;
    double nn = pow(n,-1);
    double mm = pow(m,-1);
    double kk = pow(k,-1);
    struct point *temp,*temp1;
    int ii = -1;
    int counter,counter1,counter2,counter3,counter4;
    for(i = 0;i < 1;i+=nn){
      for(j = 0; j < 1;j+=mm){
        for(l = 0; l < 1;l+=kk){
          ii++;

          //first of all we will try to find the nearest points of C for each p that belong to this box. After that we will search in the neighbouring boxes
          if(boxes[ii].number_of_points != 0){
            for(counter = 0; counter < boxes[ii].number_of_points; counter++){
              cox = boxes[ii].points_insideBox[counter].x;
              coy = boxes[ii].points_insideBox[counter].y;
              coz = boxes[ii].points_insideBox[counter].z;
              min = 10000;
              min1 = 10000;
              distance = 0;
              distance1 = 0;
              if(boxes[ii].number_of_points_in_C != 0 ){
                for(counter1 = 0; counter1 < boxes[ii].number_of_points_in_C; counter1++){
                  cox2 = boxes[ii].points_insideBox_in_C[counter1].x;
                  coy2 = boxes[ii].points_insideBox_in_C[counter1].y;
                  coz2 = boxes[ii].points_insideBox_in_C[counter1].z;
                  x = pow(cox - cox2,2);
                  y = pow(coy - coy2,2);
                  z = pow(coz - coz2,2);
                  distance = sqrt(x+y+z);
                  if(distance < min){
                    min = distance;
                  }
                }
              }  

              for(counter1 = 0; counter1 < boxes[ii].number_of_neighbours; counter1++){
                counter3 = boxes[ii].neighbour_boxes_id[counter1];
                if(boxes[counter3].number_of_points_in_C != 0 ){
                  for(counter2 = 0;counter2 < boxes[counter3].number_of_points_in_C; counter2++){  //counter2 counts the points in C belonging to each neighbour
                    cox2 = boxes[counter3].points_insideBox_in_C[counter2].x;
                    coy2 = boxes[counter3].points_insideBox_in_C[counter2].y;
                    coz2 = boxes[counter3].points_insideBox_in_C[counter2].z;
                    x = pow(cox - cox2,2);
                    y = pow(coy - coy2,2);
                    z = pow(coz - coz2,2);
                    distance1 = sqrt(x+y+z);
                    if(distance1 < min1)
                      min1 = distance1;
                  }
                }else
                  min1 = - 1; //no points in C
              }

              if(min1 == -1 && min == 10000)
                boxes[ii].points_insideBox[counter].min_distance = -1;
              else{
              if(min < min1)
                boxes[ii].points_insideBox[counter].min_distance = min;
              else
                boxes[ii].points_insideBox[counter].min_distance = min1;
              }
            }
          }
        }
      }
    }
    return boxes;
}

/*This functions searches if min_distance == -1. If many distances are equal to -1, then something is wrong*/
void check(struct box *boxes)
{
  int counter = 0;
  int i,j;
  for(i = 0; i < number_of_boxes; i++){
    for(j = 0; j < boxes[i].number_of_points; j++){
      if(boxes[i].points_insideBox[j].min_distance == -1){
        counter++;
      }
    }
  }
  if(counter > 50)
    printf("Too many points have a neighbour point in C far away\n");
  else
    printf("[OK]\n");
  
}
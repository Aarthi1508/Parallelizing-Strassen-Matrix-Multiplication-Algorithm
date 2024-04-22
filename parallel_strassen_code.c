//#!/bin/sh
/*
 * This code parallizes strassen matrix multiplication and records the execution times.
 * Input to the code :
 * K -> where n = 2^k, which is the size of the matrix n x n.
 * K'-> where s = n/2^k', which is the size of terminal matrix s x s.
 * q -> where number of threads = 2^q, which is the total number of threads used for parallization.
 * Output of the code :
 * records the execution time of the parallel strasen matrix multiplication.
 */


#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <omp.h>
#include <stdbool.h>


#define MAX_THREADS 65536
#define DEBUG 0


//declaring global variables
int k, k_terminal, num_of_threads, n;
struct timespec start, stop, start_naive, stop_naive;
double total_time, total_time_naive;


//This function is used to build matrices
int** build_matrix(int n){
    int i,j;
    int **ptr_to_row = (int**)calloc(n, sizeof(int*)); //dynamically creating n blocks of memory to store row pointers. 
    int *row_elm_pointer = (int*)calloc(n*n, sizeof(int));//dynamically creates n*n contiguous blocks of memory each with 4 bytes(sizeof(int)).
    for(i=0; i<n; i++){
        *(ptr_to_row+i) = row_elm_pointer; //dereferencing pointer and storing row pointer.
        row_elm_pointer += n; // shifting the pointer by 16 * 4 = 64 memory locations to get subsequent row pointer.
    }
    if(DEBUG){
        for(i=0;i<n;i++){
            printf("row pointer %d : %d.\n",i,*(ptr_to_row+i));
        }
    }
    if(DEBUG){
        for(i=0;i<n;i++){
            for(j=0;j<n;j++){
                printf("Elements in row %d -> [%d]\n",i,*(row_elm_pointer+j));
            }
        }
    }
    return ptr_to_row; //returing double pointer storing row pointers in it.
}


//initializing matrices A,B,C and P.
//C matrix is used to store result of matrix multiplication using Strassen method
//P matrix is used to store result of matrix mult using navie method
//
//void build_matrices(int n) {
//    int **A = build_matrix(n);
//    int **B = build_matrix(n);
//    int **C = build_matrix(n);
//    int **P = build_matrix(n);
//}


//free memory
void free_pointer_array(int** arr)
{
    free(*arr);
    free(arr);
}


//initialize matrices A & B using rand function of stdlib in c in 0-99 range.
void initialize_matrices(int n, int** A, int** B){
    int i,j;
    for(i=0;i<n;i++){
        for(j=0;j<n;j++){
            A[i][j] = rand()%1000;
            B[i][j] = rand()%1000;
        }
    }
    if(DEBUG) {
    for(i=0;i<n;i++){
      for(j=0;j<n;j++){
           printf("%d ",A[i][j]);
      }
      printf("\n");
   }
    for(i=0;i<n;i++){
      for(j=0;j<n;j++){
           printf("%d ",B[i][j]);
      }
      printf("\n");
   }
}
}


//standard matrix multiplication
void standard_matrix_multiplication_naive(int n, int** A, int** B, int** C){
    int i,j,k;
    for(i=0;i<n;i++){
        for(j=0;j<n;j++){
            C[i][j] = 0;
            for(k=0;k<n;k++){
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }

}


//calculate quarter matrix
void calc_quarter_matrix(int** res_mat, int n, int** org_mat, int x, int y){
	for(int i=0; i<n; i++){
		res_mat[i] = &org_mat[x+i][y];
	}
}


//ADD matrices.
void adding_matrices(int **res_mat, int n, int **A, int **B){
	for (int i = 0; i < n ; i++){
		for (int j = 0; j < n; j++){
			res_mat[i][j] = A[i][j] + B[i][j];
        }
    }
}


//Subtract matrices
void subtract_matrices(int **res_mat, int n, int **A, int **B){
	for (int i = 0; i < n ; i++){
		for (int j = 0; j < n; j++){
			res_mat[i][j] = A[i][j] - B[i][j];
        }
    }	
}


//calculate C matrix from strassen rec compute
/*
void calculate_C_matrix(int size, int** C11, int** C12, int** C21, int** C22, int** M1, int**M2, int** M3, int** M4, int** M5, int**M6, int** M7){
        int i,j;
	for(int i=0; i<size; i++){
		for(int j=0; j<size; j++){
			C11[i][j] = M1[i][j]+M4[i][j]-M5[i][j]+M7[i][j];
			C12[i][j] = M3[i][j]+M5[i][j];
			C21[i][j] = M2[i][j] + M4[i][j];
			C22[i][j] = M1[i][j] - M2[i][j] + M3[i][j] + M6[i][j];		
		}
	}
        free_pointer_array(M1);
        free_pointer_array(M2);
        free_pointer_array(M3);
        free_pointer_array(M4);
        free_pointer_array(M5);
        free_pointer_array(M6);
        free_pointer_array(M7);
        free(A11); 
        free(A12); 
        free(A21); 
        free(A22); 
        free(B11); 
        free(B12); 
        free(B21); 
        free(B22); 
        free(C11); 
        free(C12); 
        free(C21); 
        free(C22);       
}
*/


//compute strasen matrix recursively using multiple threads and accum results
void rec_strassen_matrix_mul(int n, int **A, int **B, int** C){
    //check for terminal matrix size and execute naive function once terminal matrix S of size s X s is reached.
    if(((float)n) <= pow(2,k)/pow(2,k_terminal)){
        standard_matrix_multiplication_naive(n,A,B,C);
    }
    else{
        //decreasing the size by half recursively.
        int new_size = n/2;
        //create matrices with 2X2 size where each matrices has 4 elements.
        //A and B numbered matrices are 2 matrices with size 2X2
        //C numbered matrix is a resultant matrix with size 2X2.
        //A numbered matrices
        int **A11 = (int**)calloc(new_size, sizeof(int*));
        int **A12 = (int**)calloc(new_size, sizeof(int*));
        int **A21 = (int**)calloc(new_size, sizeof(int*));
        int **A22 = (int**)calloc(new_size, sizeof(int*));
        //B numbered matrices
        int **B11 = (int**)calloc(new_size, sizeof(int*));
        int **B12 = (int**)calloc(new_size, sizeof(int*));
        int **B21 = (int**)calloc(new_size, sizeof(int*));
        int **B22 = (int**)calloc(new_size, sizeof(int*));
        //C numbered matrices
        int **C11 = (int**)calloc(new_size, sizeof(int*));
        int **C12 = (int**)calloc(new_size, sizeof(int*));
        int **C21 = (int**)calloc(new_size, sizeof(int*));
        int **C22 = (int**)calloc(new_size, sizeof(int*));
        //Building matrices used for strassen computation
        int **M1= build_matrix(new_size);
        int **M2= build_matrix(new_size);
        int **M3= build_matrix(new_size);
        int **M4= build_matrix(new_size);
        int **M5= build_matrix(new_size);
        int **M6= build_matrix(new_size);
        int **M7= build_matrix(new_size);
        //Building intermediate matrices
        int **aAM1 = build_matrix(new_size);
        int **aBM1 = build_matrix(new_size);
        int **aAM2 = build_matrix(new_size);
        int **sBM3 = build_matrix(new_size);
        int **sBM4 = build_matrix(new_size);
        int **aAM5 = build_matrix(new_size);
        int **aAM6 = build_matrix(new_size);
        int **aBM6 = build_matrix(new_size);
        int **sAM7 = build_matrix(new_size);
        int **aBM7 = build_matrix(new_size);
        //Get elements from the quarter matrix (reducing size)
        calc_quarter_matrix(A11, new_size, A, 0, 0);
        calc_quarter_matrix(A12, new_size, A, 0, new_size);
        calc_quarter_matrix(A21, new_size, A, new_size, 0);
        calc_quarter_matrix(A22, new_size, A, new_size, new_size);	
        calc_quarter_matrix(B11, new_size, B, 0, 0);
        calc_quarter_matrix(B12, new_size, B, 0, new_size);
        calc_quarter_matrix(B21, new_size, B, new_size, 0);
        calc_quarter_matrix(B22, new_size, B, new_size, new_size);
        calc_quarter_matrix(C11, new_size, C, 0, 0);
        calc_quarter_matrix(C12, new_size, C, 0, new_size);
        calc_quarter_matrix(C21, new_size, C, new_size, 0);
        calc_quarter_matrix(C22, new_size, C, new_size, new_size);
        //parallelizing tasks
        //int thread_id = omp_get_thread_num();
        //printf("\nthread id : %d\n", thread_id);
        #pragma omp task
	{
          //int thread_id = omp_get_thread_num();
          //printf("\nthread id : %d\n", thread_id);
          //M1 = (A11 + A22) * (B11 + B22)
	  adding_matrices(aAM1, new_size, A11, A22);
	  adding_matrices(aBM1, new_size, B11, B22);
 	  rec_strassen_matrix_mul(new_size, aAM1, aBM1, M1);
	}
	#pragma omp task
	{
          //int thread_id = omp_get_thread_num();
          //printf("\nthread id : %d\n", thread_id); 
          //M2 = (A21 + A22) * B11
          adding_matrices(aAM2, new_size, A21, A22);
	  rec_strassen_matrix_mul(new_size, aAM2, B11, M2);
	}	
	#pragma omp task
        {
          //int thread_id = omp_get_thread_num();
          //printf("\nthread id : %d\n", thread_id);
          //M3 = A11 * (B12 - B22)
	  subtract_matrices(sBM3, new_size, B12, B22);
	  rec_strassen_matrix_mul(new_size, A11, sBM3, M3);
	}
        #pragma omp task
        {
          //int thread_id = omp_get_thread_num();
          //printf("\nthread id : %d\n", thread_id);
 
          //M4 = A22 * (B21 - B11)
          subtract_matrices(sBM4, new_size, B21, B11);
	  rec_strassen_matrix_mul(new_size, A22, sBM4, M4);
	}
        #pragma omp task
        {
          //int thread_id = omp_get_thread_num();
          //printf("\nthread id : %d\n", thread_id);
          ////M5 = B22 * (A11 + A12)
 
          adding_matrices(aAM5, new_size, A11, A12);
	  rec_strassen_matrix_mul(new_size, aAM5, B22, M5);
	}
	#pragma omp task
        {
          //int thread_id = omp_get_thread_num();
          //printf("\nthread id : %d\n", thread_id);
 
          //M6 = (A21 - A11) * (B11 + B12)
          subtract_matrices(aAM6, new_size, A21, A11);
	  adding_matrices(aBM6, new_size, B11, B12);
	  rec_strassen_matrix_mul(new_size, aAM6, aBM6, M6);
	}
	#pragma omp task
        {
          //int thread_id = omp_get_thread_num();
          //printf("\nthread id : %d\n", thread_id);
 
          //M7 = (A12 - A22) * (B21 + B22)
          subtract_matrices(sAM7, new_size, A12, A22);
	  adding_matrices(aBM7, new_size, B21, B22);
	  rec_strassen_matrix_mul(new_size, sAM7, aBM7, M7);
	}
	#pragma omp taskwait
          {
               //int thread_id = omp_get_thread_num();
               //printf("\nthread id : %d\n", thread_id);
               //calculate_C_matrix(new_size, C11, C12, C21, C22, M1, M2, M3, M4, M5, M6, M7);
	       for(int i=0; i<new_size; i++){
                  for(int j=0; j<new_size; j++){
			C11[i][j] = M1[i][j]+M4[i][j]-M5[i][j]+M7[i][j];
			C12[i][j] = M3[i][j]+M5[i][j];
			C21[i][j] = M2[i][j] + M4[i][j];
			C22[i][j] = M1[i][j] - M2[i][j] + M3[i][j] + M6[i][j];		
		  }
	       }
               free_pointer_array(M1);
               free_pointer_array(M2);
               free_pointer_array(M3);
               free_pointer_array(M4);
               free_pointer_array(M5);
               free_pointer_array(M6);
               free_pointer_array(M7);
               free(A11); 
               free(A12); 
               free(A21); 
               free(A22); 
               free(B11); 
               free(B12); 
               free(B21); 
               free(B22); 
               free(C11); 
               free(C12); 
               free(C21); 
               free(C22);
          }
    }

}


//parallelize strassen matrix mult
void parallel_strassen_multiplication(int n,int **A, int **B, int** C){
    #pragma omp parallel
    {
        //int thread_id = omp_get_thread_num();
        #pragma omp single
        {
            //int thread_id = omp_get_thread_num();
            //printf("\nthread id : %d\n", thread_id);
            rec_strassen_matrix_mul(n, A, B, C);
        }
    }
}


//Check result
void check_result(int** C,int** P, int n, double total_time, double total_time_naive){
    int i,j;
    bool flag = true;
    for(i=0;i<n;i++){
        for(j=0;j<n;j++){
            if(P[i][j] != C[i][j]){
                flag = false;
            }
        }
    }
    if(flag){
        printf("Exec status = Successful,  K = %d, Matrix size = %dX%d, Threshold (k') = %d, No of threads = %d, Exec time (in secs) = %3.4f \n", k, n, n, k_terminal, num_of_threads, total_time);
    }else{
        printf("Wrong Answer!!!\n");
        printf("======================\n");
        printf("Printing Strassen matrix\n");
        for(i=0;i<n;i++){
            for(j=0;j<n;j++){
                printf("%d ", C[i][j]);
            }
            printf("\n");
        }
        printf("========================\n");
        printf("printing naive matrix\n");
        for(i=0;i<n;i++){
            for(j=0;j<n;j++){
                printf("%d ", P[i][j]);
            }
            printf("\n");
        }
        printf("=======================\n");
        printf("K=%d, Matrix size = %dX%d, Threshold(k')= %d, Threads = %d, Exec time = %8.4f sec \n", k, n, n, k_terminal, num_of_threads, total_time);
    }

}


// main function : Program starts execution from here.
int main(int argc, char** argv){
    double total_time, total_time_naive;
    int q, i, j, s;
    //reading input and validating input
    if(argc != 4) {
        printf("Need 3 integer's as input\n");
        printf("Please Use: ./<executable_file_name.exe> <k> <k'> <number_of_threads>\n");
        printf("Where k !> 11 and k' !> k are used to determine matrix & terminal matrix size, i.e., matrix_size(n) = 2^k\n");
        exit(0);
    }
    //get value of k from command-line
    k = atoi(argv[argc-3]); // since k is the first value in command line and it is in argv[1](which is argc-3 => 4-3=1), argv[0] has the executable file name.
    //checking if k value is within range to avoid resource allocation issues.
    if( k >= 100) { //if k > 11, segmentation faults or resource allocation issues are observed.
        printf("Maximum k value allowed: %d.\n", 11);
        exit(0);
    }
    n = 1 << k; // where n is the size of the matrix (n X n)
    k_terminal = atoi(argv[argc-2]);
    //checking if k' value is greater than k
    if(k_terminal > k){
        printf("K' value cannot be more than k value.");
        exit(0);
    }
    s = 1 << k_terminal; //where s is the size of the terminal matrix S X S.
    q = atoi(argv[argc-1]);
    //checking if number of threads is greatert than MAX_THREADS.
    if((num_of_threads = (1 << q)) > MAX_THREADS){
        printf("Maximum number of threads allowed : %d.\n", MAX_THREADS);
        exit(0);
    }
    //build matrices A, B, C and an other matrix for navie matrix result and initialize with 0
    //build_matrices(n);
    //initialize matrices A&B with some random values between 0-99.
    int **A = build_matrix(n);
    int **B = build_matrix(n);
    int **C = build_matrix(n);
    int **P = build_matrix(n);
    initialize_matrices(n, A, B);
    omp_set_dynamic(0);
    //initializing parallel programming functions using open mpi - shared memory.
    omp_set_num_threads(num_of_threads);
    /*
    //starting clock time for naive matrix mult
    clock_gettime(CLOCK_REALTIME, &start_naive);
    //calculate same matrix mult using naive/standard method
    standard_matrix_multiplication_naive(n,A,B,P);
    //stopping clock time for naive matrix mult
    clock_gettime(CLOCK_REALTIME, &stop_naive);
    //computing total time
    total_time_naive = (stop_naive.tv_sec-start_naive.tv_sec)
        +0.000000001*(stop_naive.tv_nsec-start_naive.tv_nsec);
    */
    //starting clock to find execution time
    clock_gettime(CLOCK_REALTIME, &start);
    //This function parallelizes strassen multiplication using threads = num_of_threads.
    //printf("Executing parallel function : \n");
    parallel_strassen_multiplication(n,A,B,C);
    //ending clock to find execution time
    clock_gettime(CLOCK_REALTIME, &stop);
    //compute total time for parallel execution
    total_time = (stop.tv_sec-start.tv_sec)
        +0.000000001*(stop.tv_nsec-start.tv_nsec);
    //printf("%8.4f\n",total_time);
    //starting clock time for naive matrix mult
    clock_gettime(CLOCK_REALTIME, &start_naive);
    //calculate same matrix mult using naive/standard method
    standard_matrix_multiplication_naive(n,A,B,P);
    //stopping clock time for naive matrix mult
    clock_gettime(CLOCK_REALTIME, &stop_naive);
    //computing total time
    total_time_naive = (stop_naive.tv_sec-start.tv_sec)
        +0.000000001*(stop.tv_nsec-start.tv_nsec);
    //printf("%8.4f\n", total_time_naive);
    //This function checks the result and prints the results.
    check_result(C,P,n, total_time, total_time_naive);
}

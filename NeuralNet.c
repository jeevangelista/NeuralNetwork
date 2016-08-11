#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

/*
 * Returns a random number from zero to one
 */
double random_zero_to_one(){
  return rand()/(RAND_MAX + 1.);
}


/*
 * Matrix multiplication
 * INPUTS
 * mat1: 2D array of double values
 * mat1_row: number of rows of mat1
 * mat1_col: number of columns of mat1
 * mat2: 2D array of double values
 * mat2_row: number of rows of mat2
 * mat2_col: number of columns of mat2
 * OUTPUT
 * product: 2D matrix of products
 */
double** matrix_multiplication(double **mat1,
  							               int mat1_row,
                               int mat1_col,
                               double **mat2,
                               int mat2_row,
                               int mat2_col){
  
  // check size of matrices
  if(mat1_col != mat2_row){
    fprintf(stderr, "ERROR: columns of mat1 and rows of mat2 are not matched!");
    exit(EXIT_FAILURE);
  }

  // Matrix allocation
  double** product = (double**) calloc(mat1_row, sizeof(double*)); // allocate matrix pointer
  for(int i=0; i<mat1_row; i++)
    product[i] = (double*) calloc(mat2_col, sizeof(double)); // allocate each element

  // Matrix multiplication
  for(int i=0; i<mat1_row; i++)
    for(int j=0; j<mat2_col; j++)
      for(int k=0; k<mat1_col; k++)
        product[i][j] += mat1[i][k] * mat2[k][j];

  return product;
}


/*
 * Matrix transposition
 * x  x INPUTS
 * mat: 2D array of double values
 * mat_row: number of rows of mat1
 * mat_col: number of columns of mat1
 * OUTPUT
 * transpo: 2D transposed matrix
 */
double** matrix_transposition(double **mat,
                               int mat_row,
                               int mat_col){
  // Matrix allocation
  double** transpo = (double**) calloc(mat_col, sizeof(double*)); // allocate matrix pointer
  for(int i=0; i<mat_col; i++)
    transpo[i] = (double*) calloc(mat_row, sizeof(double)); // allocate each element

  // transpositon
  for(int i=0; i<mat_col; i++)
    for(int j=0; j<mat_row; j++)
      transpo[i][j] = mat[j][i];

  return transpo;
}


/*
 * Matrix addition
 * INPUTS
 * mat1: 2D array of double values
 * mat1_row: number of rows of mat1
 * mat1_col: number of columns of mat1
 * mat2: 2D array of double values
 * mat2_row: number of rows of mat2
 * mat2_col: number of columns of mat2
 * OUTPUT
 * sum: 2D matrix of products
 */
double** matrix_addition(double **mat1,
                         int mat1_row,
                         int mat1_col,
                         double **mat2,
                         int mat2_row,
                         int mat2_col){

  // check size of matrices
  if(mat1_row != mat2_row && mat1_col != mat2_col){
    fprintf(stderr, "ERROR: columns of mat1 and rows of mat2 are not matched!");
    exit(EXIT_FAILURE);
  }

  // Matrix allocation
  double** sum = (double**) calloc(mat1_row, sizeof(double*)); // allocate matrix pointer
  for(int i=0; i<mat1_row; i++)
    sum[i] = (double*) calloc(mat1_col, sizeof(double)); // allocate each element
  printf("%d %d\n", mat1_row, mat1_col);
  // Addition
  for(int i=0; i<mat1_row; i++)
    for(int j=0; j<mat1_col; j++)
      sum[i][j] = mat1[i][j] + mat2[i][j];

  return sum;
}


/*
 * Pointwise multiplication
 * INPUTS
 * mat1: 2D array of double values
 * mat1_row: number of rows of mat1
 * mat1_col: number of columns of mat1
 * mat2: 2D array of double values
 * mat2_row: number of rows of mat2
 * mat2_col: number of columns of mat2
 * OUTPUT
 * product: 2D matrix of products
 */
double** pointwise_multiplication(double **mat1,
                                  int mat1_row,
                                  int mat1_col,
                                  double **mat2,
                                  int mat2_row,
                                  int mat2_col){

  // check size of matrices
  if(mat1_row != mat2_row && mat1_col != mat2_col){
    fprintf(stderr, "ERROR: columns of mat1 and rows of mat2 are not matched!");
    exit(EXIT_FAILURE);
  }

  // Matrix allocation
  double** product = (double**) calloc(mat1_row, sizeof(double*)); // allocate matrix pointer
  for(int i=0; i<mat1_row; i++)
    product[i] = (double*) calloc(mat1_col, sizeof(double)); // allocate each element
  
  // Sigmoid
  for(int i=0; i<mat1_row; i++)
    for(int j=0; j<mat1_col; j++)
      product[i][j] = mat1[i][j] * mat2[i][j];

  return product;
}


/*
 * Sigmoid function
 * INPUTS
 * mat: 2D array of double values
 * mat_row: number of rows of mat1
 * mat_col: number of columns of mat1
 * OUTPUT
 * transpo: 2D transposed matrix
 */
double** matrix_sigmoid(double **mat,
                               int mat_row,
                               int mat_col){
  // Matrix allocation
  double** sigmoid = (double**) calloc(mat_row, sizeof(double*)); // allocate matrix pointer
  for(int i=0; i<mat_row; i++)
    sigmoid[i] = (double*) calloc(mat_col, sizeof(double)); // allocate each element

  // transpositon
  for(int i=0; i<mat_row; i++)
    for(int j=0; j<mat_col; j++)
      sigmoid[i][j] = 1.0/(1.0+exp(-mat[i][j]));

  return sigmoid;
}


/*
 * Returs a permutation of n natural numbers
 * follows the Fisher-Yates algorithm
 */
int* shuffle(int n){
  int* shuffled = (int*) calloc(n, sizeof(int));
  // seed
  time_t t;
  srand((unsigned) time(&t));
  for(int i=0; i<n; i++){
    shuffled[i] = i;
  }
  int temp;
  for(int i=n-1; i>=0; i--){
    int pos = rand() % (i+1);
    temp = shuffled[i];
    shuffled[i] = shuffled[pos];
    shuffled[pos] = temp;
  }
  printf("\n");
  return shuffled;
}


int main(){

  // double** A = (double**) calloc(2, sizeof(double*)); // allocate matrix pointer
  // for(int i=0; i<2; i++)
  //   A[i] = (double*) calloc(3, sizeof(double)); // allocate each element

  // double** B = (double**) calloc(2, sizeof(double*)); // allocate matrix pointer
  // for(int i=0; i<2; i++)
  //   B[i] = (double*) calloc(3, sizeof(double)); // allocate each element
  // A[0][0] = 0;
  // A[0][1] = 1;
  // A[0][2] = 2;
  // A[1][0] = 9;
  // A[1][1] = 8;
  // A[1][2] = 7;

  // B[0][0] = 6;
  // B[0][1] = 5;
  // B[0][2] = 4;
  // B[1][0] = 3;
  // B[1][1] = 4;
  // B[1][2] = 5;
  
  // double **C = matrix_sigmoid(A,2,3);
  // // sigmoid
  // for(int i=0; i<2; i++){
  //   for(int j=0; j<3; j++)
  //     printf("%lf\t",C[i][j]);
  //   printf("\n");
  // }
  // printf("%lf",1.);
  // free(A);
  // free(B);
  // free(C);
  int n = 10;
  int* shuffled = shuffle(n);
  for(int i = 0; i<n; i++)
    printf("%d\t", shuffled[i]);
  printf("\n");
  free(shuffled);
  return 0;
}
/**************************************
 *
 * WRITTEN BY: JOHN EROL EVANGELISTA
 *
 **************************************/

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
  double** product = (double**) malloc(mat1_row * sizeof(double*)); // allocate matrix pointer
  for(int i=0; i<mat1_row; i++)
    product[i] = (double*) calloc(mat2_col, sizeof(double)); // allocate each element, calloc to initialize to zero

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
  double** transpo = (double**) malloc(mat_col * sizeof(double*)); // allocate matrix pointer
  for(int i=0; i<mat_col; i++)
    transpo[i] = (double*) malloc(mat_row * sizeof(double)); // allocate each element

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
 * scal1: scalar value to be added
 * mat1_row: number of rows of mat1
 * mat1_col: number of columns of mat1
 * mat2: 2D array of double values
 * scal2: scalar value
 * mat2_row: number of rows of mat2
 * mat2_col: number of columns of mat2
 * subtract: do subtraction instead
 * OUTPUT
 * sum: 2D matrix of products
 */
double** matrix_addition(double **mat1,
                         double scal1,
                         int mat1_row,
                         int mat1_col,
                         double **mat2,
                         double scal2,
                         int mat2_row,
                         int mat2_col,
                         int subtract){

  // check size of matrices
  if(mat1_row != mat2_row && mat1_col != mat2_col){
    fprintf(stderr, "ERROR: columns of mat1 and rows of mat2 are not matched!");
    exit(EXIT_FAILURE);
  }

  // Matrix allocation
  double** sum = (double**) malloc(mat1_row * sizeof(double*)); // allocate matrix pointer
  for(int i=0; i<mat1_row; i++)
    sum[i] = (double*) malloc(mat1_col * sizeof(double)); // allocate each element
  int subtract_flag = 1;
  if(subtract)
    subtract_flag = -1;
  // Addition
  for(int i=0; i<mat1_row; i++)
    for(int j=0; j<mat1_col; j++)
      if(scal1 && scal2){
        sum[i][j] = scal1 + (subtract * scal2);
      }
      else if(scal1){
        sum[i][j] = scal1 + (subtract_flag * mat2[i][j]);  
      }else if(scal2){
        sum[i][j] = mat1[i][j] + (subtract_flag * scal2);
      }else{
        sum[i][j] = mat1[i][j] + (subtract_flag * mat2[i][j]);
      }

  return sum;
}


/*
 * Pointwise multiplication
 * INPUTS
 * mat1: 2D array of double values
 * scal: scalar to be multiplied, commutative
 * mat1_row: number of rows of mat1
 * mat1_col: number of columns of mat1
 * mat2: 2D array of double values
 * mat2_row: number of rows of mat2
 * mat2_col: number of columns of mat2
 * OUTPUT
 * product: 2D matrix of products
 */
double** pointwise_multiplication(double **mat1,
                                  double scal,
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
  double** product = (double**) malloc(mat1_row * sizeof(double*)); // allocate matrix pointer
  for(int i=0; i<mat1_row; i++)
    product[i] = (double*) malloc(mat1_col * sizeof(double)); // allocate each element
  
  // Sigmoid
  for(int i=0; i<mat1_row; i++)
    for(int j=0; j<mat1_col; j++){
      if(scal && !mat1)
        product[i][j] = scal * mat2[i][j];
      else if(!scal && mat1)
        product[i][j] = mat1[i][j] * mat2[i][j];
      else{
        fprintf(stderr, "ERROR: Scalar or pointwise matrix multiplication? I'm confused!");
        exit(EXIT_FAILURE);
      }
    }

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
  double** sigmoid = (double**) malloc(mat_row * sizeof(double*)); // allocate matrix pointer
  for(int i=0; i<mat_row; i++)
    sigmoid[i] = (double*) malloc(mat_col * sizeof(double)); // allocate each element

  // transpositon
  for(int i=0; i<mat_row; i++)
    for(int j=0; j<mat_col; j++)
      sigmoid[i][j] = 1.0/(1.0+exp(-mat[i][j]));

  return sigmoid;
}


/*
 * Returns a permutation of n natural numbers
 * follows the Fisher-Yates algorithm
 */
int* shuffle(int n){
  int* shuffled = (int*) malloc(n * sizeof(int));
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
  return shuffled;
}


/*
 * Initializes weights of the network
 * structure: Array of network structure (size = hidden_layers + 2)
 * hidden_layers: Number of hidden layer
 * NN[number of hidden layers+1][node to the right of the edge][node to the left of the edge]
 */
double*** initialize_network(int* structure, int hidden_layers){
  double*** NN = (double***) malloc((hidden_layers+1) * sizeof(double**));
  for(int i=0; i<=hidden_layers; i++){
    NN[i] = (double**) malloc(structure[i+1] * sizeof(double*));
    for(int j=0; j<structure[i+1]; j++){
      NN[i][j] = (double*) malloc(structure[i] * sizeof(double));
      for(int k=0; k<structure[i]; k++)
        NN[i][j][k] = random_zero_to_one();
    }
  }
  return NN;
}


/*
 * Initializes biases of the network
 * structure: Array of network structure (size = hidden_layers + 2)
 * hidden_layers: Number of hidden layer
 */
double** initialize_bias(int* structure, int hidden_layers){
  double** bias = (double**) malloc((hidden_layers+1) * sizeof(double*));
  for(int i=0; i<=hidden_layers; i++){
    bias[i] = (double*) malloc(structure[i+1] * sizeof(double));
    for(int j=0; j<structure[i+1]; j++){
      bias[i][j] = random_zero_to_one();
    }
  }
  return bias;
}


/*
 * reserves memory for outputs of the network
 * structure: Array of network structure (size = hidden_layers + 2)
 * hidden_layers: Number of hidden layer
 * out_NN[number of hidden layers+1][node to the right of the edge][node to the left of the edge]
 */
double** initialize_outputs(int* structure, int hidden_layers){
  double** out_NN = (double**) malloc((hidden_layers+1) * sizeof(double*));
  for(int i=0; i<=hidden_layers; i++){
    out_NN[i] = (double*) malloc(structure[i+1] * sizeof(double));
  }
  return out_NN;
}


// reserves a block of memory for a matrix
double** initialize_matrix(int row, int col, int zero_flag){
  double** matrix = (double**) malloc(row * sizeof(double*));
  for(int i=0; i<row; i++)
    if(zero_flag)
      matrix[i] = (double*) calloc(col, sizeof(double));
    else
      matrix[i] = (double*) malloc(col * sizeof(double));
  return matrix;
}


/*
 * Copy pointer
 */
void copy_pointer(double **src,
                  double *src_vector,
                  int src_row,
                  int src_col,
                  double **dest,
                  double *dest_vector,
                  int dest_row,
                  int dest_col){
  if(src_col==1 && dest_col==0 && src && dest_vector && dest == NULL && src_vector == NULL){ // dest is a vector
    for(int i=0; i<dest_row; i++)
      dest_vector[i] = src[i][0];
  }else if(src_col==0 && dest_col==1 && src_vector && dest && dest_vector == NULL && src == NULL){ //src is a vector
    for(int i=0; i<dest_row; i++)
      dest[i][0] = src_vector[i];
  }else if(src_col == dest_col && src_row == dest_row && src_vector == NULL && dest_vector == NULL){
    for(int i=0; i<dest_row; i++)
      for(int j=0; j<dest_col; j++)
        dest[i][j] = src[i][j];
  }else{
    fprintf(stderr, "ERROR: error in matrices!");
    exit(EXIT_FAILURE);
  }
}


/*
 * Frees a matrix pointer
 */
void free_matrix(double** matrix,
                 int row,
                 int col){
  for(int i=0; i<row; i++)
    free(matrix[i]);
  free(matrix);
}


/* 
 * set values of output nodes
 */
void set_output_nodes(int value, int node_num, double** out){
  int pow = 1 << node_num;
  int mod = value % pow;
  if(mod!=value){
    fprintf(stderr, "ERROR: Number of nodes not enough to represent output!\n");
    exit(EXIT_FAILURE);
  }
  
  for(int i=node_num-1; i>=0; i--){
    int pow = 1<< i;
    int quotient = value/pow;
    value = value%pow;
    if(quotient)
      out[i][0]=1;
    else
      out[i][0]=0;
  }
}


/*
 * Train Neural Networks
 * INPUTS
 * max_epoch: maximum number of epochs
 * inputs: number of inputs of the network
 * instances: number of input instances
 * outputs: number of output outcomes
 * input_matrix: input matrix
 * output_vector: list of expected outputs
 * learning_rate: learning rate
 * structure = [input,h1,h2,...,hn,output]
 * NN, out_NN = [h1,h2,...,hn,output]
 */
int train_neural_network(int max_epoch,
                          int instances,
                          int* structure,
                          int hidden_layers,
                          double** input_matrix, 
                          int* output_vector,
                          double learning_rate){

  // initialize network weights
  double*** NN = initialize_network(structure, hidden_layers);
  double** bias = initialize_bias(structure, hidden_layers);
  double** out_NN = initialize_outputs(structure, hidden_layers);

  // initialize other variables
  double* totalerr = (double*) calloc(max_epoch, sizeof(double)); // calloc initializes to zero
  double** desired = initialize_matrix(structure[hidden_layers+1], 1, 1);
  for(int iter=0; iter<max_epoch; iter++){
    int* perm = shuffle(instances);
    for(int n=0; n<instances; n++){
      // Pick an input and out output randomly
      set_output_nodes(output_vector[perm[n]], structure[hidden_layers+1], desired);

      double** in = initialize_matrix(structure[0],1, 0);
      copy_pointer(NULL, input_matrix[perm[n]], structure[0], 0, in, NULL, structure[0], 1);
      // forward pass
      for(int i=0; i<=hidden_layers; i++){
        double** layer_bias = initialize_matrix(structure[i+1], 1, 0);
        copy_pointer(NULL, bias[i], structure[i+1], 0, layer_bias, NULL, structure[i+1], 1);
        double** v_mul = matrix_multiplication(NN[i],structure[i+1],structure[i],in,structure[i],1);
        free_matrix(in,structure[i],1);
        double** v_add = matrix_addition(v_mul,0,structure[i+1],1,layer_bias,0,structure[i+1],1,0);
        free_matrix(v_mul,structure[i+1],1);
        double** out = matrix_sigmoid(v_add, structure[i+1],1);
        free_matrix(v_add,structure[i+1],1);
        free_matrix(layer_bias,structure[i+1],1);
        copy_pointer(out,NULL,structure[i+1],1,NULL,out_NN[i],structure[i+1],0);
        in = out;
      }
      // back propagation
      double** out = in;
      double** err = matrix_addition(desired, 0, structure[hidden_layers+1], 1, out, 0, structure[hidden_layers+1],1,1);
      
      double** delta;

      for(int i=hidden_layers; i>=0; i--){
        // output layer
        if(i==hidden_layers){
          // compute delta
          double** mult_result = pointwise_multiplication(err, 0, structure[i+1], 1, out, structure[i+1], 1);
          double** sub_result = matrix_addition(NULL, 1, structure[i+1], 1, out, 0, structure[i+1], 1, 1);
          delta = pointwise_multiplication(mult_result, 0, structure[i+1], 1, sub_result, structure[i+1], 1);
          // compute new weights
          double** mult_result2 = pointwise_multiplication(NULL, learning_rate, structure[i+1], 1, delta, structure[i+1], 1);
          double** prev_layer_out = initialize_matrix(structure[i], 1, 0);
          copy_pointer(NULL, out_NN[i-1], structure[i], 0, prev_layer_out, NULL, structure[i], 1); // out_NN[i-1] -> kasama output lang no input, so pag i=hidden layer i is pointing sa output i-1 sa hidden layer before it, shift ng 1 sa structure
          double** transpose = matrix_transposition(prev_layer_out, structure[i], 1);
          double** delta_mult = matrix_multiplication(mult_result2, structure[i+1], 1, transpose, 1, structure[i]);
          double** new_weight = matrix_addition(NN[i], 0, structure[i+1], structure[i], delta_mult, 0, structure[i+1], structure[i], 0);
          copy_pointer(new_weight, NULL, structure[i+1], structure[i], NN[i], NULL, structure[i+1], structure[i]);
          
          // compute biases
          double** layer_bias = initialize_matrix(structure[i+1], 1, 0);
          copy_pointer(NULL, bias[i], structure[i+1], 0, layer_bias, NULL, structure[i+1], 1);
          double** new_bias = matrix_addition(layer_bias, 0, structure[i+1], 1, delta_mult, 0, structure[i+1], 1, 0);
          copy_pointer(new_bias, NULL, structure[i+1], 1, NULL, bias[i], structure[i+1], 0);

          // free variables
          free_matrix(mult_result, structure[i+1], 1);
          free_matrix(sub_result, structure[i+1], 1);
          free_matrix(mult_result2, structure[i+1], 1);
          free_matrix(out, structure[i+1], 1);
          free_matrix(transpose, 1, structure[i]);
          free_matrix(delta_mult, structure[i+1], structure[i]);
          free_matrix(new_weight, structure[i+1], structure[i]);
          free_matrix(layer_bias, structure[i+1],1);
          free_matrix(new_bias, structure[i+1],1);

          out = prev_layer_out;
          for(int j=0; j<structure[i+1]; j++)
            printf("%lf ", bias[i][j]);
          printf("\n\n");
        }
      }
    }
  }
  // free variables
  for(int i=0; i<hidden_layers+1; i++){
    for(int j=0; j<structure[i+1]; j++){
      free(NN[i][j]);
    }
      free(NN[i]);
  }
  free(NN);
  for(int i=0; i<hidden_layers+1; i++)
    free(bias[i]);
  free(bias);
  free(totalerr);
  free(desired);

  return 0;
 }


int main(){
  // seed for randomness
  time_t t;
  srand((unsigned) time(&t));

  // double* A = (double*) calloc(3, sizeof(double)); // allocate matrix pointer
  // // for(int i=0; i<3; i++)
  // //   A[i] = (double*) calloc(1, sizeof(double)); // allocate each element

  // double** B = (double**) calloc(3, sizeof(double*)); // allocate matrix pointer
  // for(int i=0; i<3; i++)
  //   B[i] = (double*) calloc(1, sizeof(double)); // allocate each element
  
  // A[0] = 0;
  // A[1] = 1;
  // A[2] = 2;

  // A[1][0] = 9;
  // A[1][1] = 8;
  // A[1][2] = 7;

  // B[0][0] = 6;
  // B[0][1] = 5;
  // B[0][2] = 4;
  // B[1][0] = 3;
  // B[1][1] = 4;
  // B[1][2] = 5;

  // copy_pointer(NULL,A,3,0,B,NULL,3,1);

  // for(int i=0; i<3; i++){
  //   for(int j=0; j<1; j++)
  //     printf("%lf ",B[i][j]);
  //   printf("\n");
  // }
  // free_pointer(NULL,A,3,0);
  // free_pointer(B,NULL,3,1);
  
  // double **C = matrix_transposition(A,2,3);
  // // sigmoid
  // for(int i=0; i<3; i++){
  //   for(int j=0; j<2; j++)
  //     printf("%lf\t",C[i][j]);
  //   printf("\n");
  // }
  // free(A);
  // free(B);
  // free(C);
  // int n = 10;
  // int* shuffled = shuffle(n);
  // for(int i = 0; i<n; i++)
  //   printf("%d\t", shuffled[i]);
  // printf("\n");
  // free(shuffled);

  int* structure = (int*) calloc(4, sizeof(int));
  structure[0] = 3;
  structure[1] = 5;
  structure[2] = 4;
  structure[3] = 3;
  //structure[4] = 2;

  int hidden_layers = 2;
  int max_epoch = 10000;
  int instances = 8;
  double learning_rate = 0.1;
  double** input_matrix = initialize_matrix(instances, 3, 0);
  input_matrix[0][0] = 0;
  input_matrix[0][1] = 0;
  input_matrix[0][2] = 0;
  input_matrix[1][0] = 1;
  input_matrix[1][1] = 0;
  input_matrix[1][2] = 0;
  input_matrix[2][0] = 0;
  input_matrix[2][1] = 1;
  input_matrix[2][2] = 0;
  input_matrix[3][0] = 1;
  input_matrix[3][1] = 1;
  input_matrix[3][2] = 0;
  input_matrix[4][0] = 0;
  input_matrix[4][1] = 0;
  input_matrix[4][2] = 1;
  input_matrix[5][0] = 1;
  input_matrix[5][1] = 0;
  input_matrix[5][2] = 1;
  input_matrix[6][0] = 0;
  input_matrix[6][1] = 1;
  input_matrix[6][2] = 1;
  input_matrix[7][0] = 1;
  input_matrix[7][1] = 1;
  input_matrix[7][2] = 1;
  int* output_vector = (int*) malloc((1 << structure[hidden_layers+1])*sizeof(int));
  output_vector[0] = 0;
  output_vector[1] = 6;
  output_vector[2] = 5;
  output_vector[3] = 3;
  output_vector[4] = 3;
  output_vector[5] = 4;
  output_vector[6] = 6;
  output_vector[7] = 0;

  train_neural_network(max_epoch, instances, structure, hidden_layers, input_matrix, output_vector, learning_rate);
  free_matrix(input_matrix,8,3);
  free(output_vector);
  printf("%d ",1<<3);
  // double*** NN = initialize_network(structure, 3);

  // for(int i=0; i<4; i++){
  //   for(int j=0; j<structure[i+1]; j++){
  //     for(int k=0; k<structure[i]; k++){
  //       printf("%lf ", NN[i][j][k]);
  //     }
  //     printf("\n");
  //   }
  //     printf("\n");
  // }
  // for(int i=0; i<4; i++){
  //   for(int j=0; j<structure[i+1]; j++){
  //     free(NN[i][j]);
  //   }
  //     free(NN[i]);
  // }
  // free(NN);
  // double ** bias = initialize_bias(structure, 3);
  // for(int i=0; i<4; i++){
  //   for(int j=0; j< structure[i+1]; j++){
  //     printf("%lf ", bias[i][j]);
  //   }
  //   printf("\n\n");
  // }
  // for(int i=0; i<4; i++)
  //   free(bias[i]);
  // free(bias);
  return 0;
}
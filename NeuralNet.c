/**************************************
 *
 * WRITTEN BY: JOHN EROL EVANGELISTA
 *
 **************************************/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include <limits.h>
#define cell_length 15


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
  double** sum = (double**) calloc(mat1_row, sizeof(double*)); // allocate matrix pointer
  for(int i=0; i<mat1_row; i++)
    sum[i] = (double*) calloc(mat1_col, sizeof(double)); // allocate each element
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
  double** product = (double**) calloc(mat1_row, sizeof(double*)); // allocate matrix pointer
  for(int i=0; i<mat1_row; i++)
    product[i] = (double*) calloc(mat1_col, sizeof(double)); // allocate each element
  
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
 * Returns a permutation of n natural numbers
 * follows the Fisher-Yates algorithm
 */
int* shuffle(int n){
  int* shuffled = (int*) calloc(n, sizeof(int));
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
  double*** NN = (double***) calloc((hidden_layers+1), sizeof(double**));
  for(int i=0; i<=hidden_layers; i++){
    NN[i] = (double**) calloc(structure[i+1], sizeof(double*));
    for(int j=0; j<structure[i+1]; j++){
      NN[i][j] = (double*) calloc(structure[i], sizeof(double));
      for(int k=0; k<structure[i]; k++){
        NN[i][j][k] = random_zero_to_one();
      }
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
  double** bias = (double**) calloc((hidden_layers+1), sizeof(double*));
  for(int i=0; i<=hidden_layers; i++){
    bias[i] = (double*) calloc(structure[i+1], sizeof(double));
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
  double** out_NN = (double**) calloc((hidden_layers+1), sizeof(double*));
  for(int i=0; i<=hidden_layers; i++){
    out_NN[i] = (double*) calloc(structure[i+1], sizeof(double));
  }
  return out_NN;
}


// reserves a block of memory for a matrix
double** initialize_matrix(int row, int col, int zero_flag){
  double** matrix = (double**) calloc(row, sizeof(double*));
  for(int i=0; i<row; i++)
    if(zero_flag)
      matrix[i] = (double*) calloc(col, sizeof(double));
    else
      matrix[i] = (double*) calloc(col, sizeof(double));
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
void free_matrix(double*** matrix,
                 int row,
                 int col){
  for(int i=0; i<row; i++){
    free((*matrix)[i]);
  }
  free(*matrix);
}


/* 
 * set values of output nodes
 */
void set_output_nodes(int value, int node_num, double*** out){
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
      (*out)[node_num-1-i][0]=1;
    else
      (*out)[node_num-1-i][0]=0;
  }
}


/* 
 * set values of output nodes
 */
int get_output_in_decimal(int node_num, double** out){
  int val = 0;
  for(int i=node_num-1; i>=0; i--){
    int pow = 1<< i;
    if(out[node_num-1-i][0]>0.5)
      val = val+pow;
  }
  return val;
}


/* 
 *determines if string contains prefix
 */
int check_prefix(const char *prefix, const char *str)
{
    return strncmp(prefix, str, strlen(prefix)) == 0;
}


/*
 * Reads a csv file and store inputs to a 2x2 matrix
 */
void read_dataset(double*** matrix, int row, int col, char* filename){
  FILE* dataset=fopen(filename, "r");
  int bufflen = row * cell_length; // buffer length is number of features times max cell len
  char* line = (char*) calloc(bufflen, sizeof(char));
  // Separate to tokens with comma delimiter
  int i = 0;
  while(fgets(line, bufflen, dataset)){
    char* dup = strdup(line);
    char* ptr = dup;
    char* token;
    int j = 0;
    while ((token = strsep(&dup, ","))){
      (*matrix)[i][j] = atof(token);
      j++;
      if(j>col){
        fprintf(stderr, "ERROR: Too few allocated columns!\n");
        exit(EXIT_FAILURE);   
      }
    }
    free(ptr);
    dup = NULL;
    i++;
    if(i>row){
      fprintf(stderr, "ERROR: Too few allocated rows!\n");
      exit(EXIT_FAILURE); 
    }
  }
  free(line);
  fclose(dataset);
}


int read_labels(int** vector, int out_instances, int out_nodes, char* filename){
  FILE* labelset = fopen(filename, "r");
  int bufflen = 20;
  char* line = (char*) calloc(bufflen, sizeof(char));
  int i=0;
  int min=INT_MAX;
  int max=INT_MIN;
  while(fgets(line, bufflen, labelset)){
    (*vector)[i] = atoi(line);
    if((*vector)[i]<min)
      min = (*vector)[i];
    if((*vector)[i]>max)
      max = (*vector)[i];
    i++;
    if(i>out_instances){
      fprintf(stderr, "ERROR: Too few allocated out rows!\n");
      exit(EXIT_FAILURE);
    }
  }
  if(pow(2.0,out_nodes) < (max-min)){
    fprintf(stderr, "ERROR: Too few nodes to represent the outputs!\n");
    exit(EXIT_FAILURE);
  }

  // Adjust the outputs so that they start with zero
  if(min!=0){
    for(int i=0; i<out_instances; i++)
      (*vector)[i]-=min;
  }
  free(line);
  fclose(labelset);
  return min;
}


/*
 * Train Neural Networks
 * INPUTS
 * max_epoch: maximum number of epochs
 * inputs: number of inputs of the network
 * train_instances: number of input train_instances
 * outputs: number of output outcomes
 * train_input_matrix: input matrix
 * train_output_vector: list of expected outputs
 * validation_input_matrix: input matrix
 * validation_output_vector: list of expected outputs
 * learning_rate: learning rate
 * structure = [input,h1,h2,...,hn,output]
 * NN, out_NN = [h1,h2,...,hn,output]
 */
void train_neural_network(int max_epoch,
                          int* structure,
                          int hidden_layers,
                          int train_instances,
                          double** train_input_matrix, 
                          int* train_output_vector,
                          int validation_instances,
                          double** validation_input_matrix, 
                          int* validation_output_vector,
                          double learning_rate,
                          double*** NN,
                          double** bias,
                          char* err_filename){

  double** out_NN = initialize_outputs(structure, hidden_layers);
  FILE* err_file = fopen(err_filename, "w");
  // initialize other variables
  double* totalerr = (double*) calloc(max_epoch, sizeof(double)); // calloc initializes to zero
  double** desired = initialize_matrix(structure[hidden_layers+1], 1, 1);
  for(int iter=0; iter<max_epoch; iter++){
    printf("Iteration %d\n",iter);
    int* perm = shuffle(train_instances);
    double** err;
    for(int n=0; n<train_instances; n++){
      // Pick an input and out output randomly
      set_output_nodes(train_output_vector[perm[n]], structure[hidden_layers+1], &desired);

      double** in = initialize_matrix(structure[0],1, 0);
      copy_pointer(NULL, train_input_matrix[perm[n]], structure[0], 0, in, NULL, structure[0], 1);
      // for(int i=0; i<structure[0]; i++){
      //   printf("%d %d\n", train_input_matrix[perm[n]][i], in[i]);
      // }
      // forward pass
      double** out;
      for(int i=0; i<=hidden_layers; i++){
        double** layer_bias = initialize_matrix(structure[i+1], 1, 0);
        copy_pointer(NULL, bias[i], structure[i+1], 0, layer_bias, NULL, structure[i+1], 1);
        double** v_mul = matrix_multiplication(NN[i],structure[i+1],structure[i],in,structure[i],1);
        free_matrix(&in,structure[i],1);
        double** v_add = matrix_addition(v_mul,0,structure[i+1],1,layer_bias,0,structure[i+1],1,0);
        free_matrix(&v_mul,structure[i+1],1);
        out = matrix_sigmoid(v_add, structure[i+1],1);
        free_matrix(&v_add,structure[i+1],1);
        free_matrix(&layer_bias,structure[i+1],1);
        copy_pointer(out,NULL,structure[i+1],1,NULL,out_NN[i],structure[i+1],0);
        in = out;
      }
      // back propagation
      err = matrix_addition(desired, 0, structure[hidden_layers+1], 1, out, 0, structure[hidden_layers+1],1,1);
      
      double** delta;

      for(int i=hidden_layers; i>=0; i--){
        // output layer
        if(i==hidden_layers){
          // compute delta
          double** mult_result = pointwise_multiplication(err, 0, structure[i+1], 1, out, structure[i+1], 1);
          double** sub_result = matrix_addition(NULL, 1, structure[i+1], 1, out, 0, structure[i+1], 1, 1);
          delta = pointwise_multiplication(mult_result, 0, structure[i+1], 1, sub_result, structure[i+1], 1);
          
          // free variables
          free_matrix(&mult_result, structure[i+1], 1);
          free_matrix(&sub_result, structure[i+1], 1);
          free_matrix(&err, structure[i+1], 1);

        }else{
          double** sub_result = matrix_addition(NULL, 1, structure[i+1], 1, out, 0, structure[i+1], 1, 1);
          double** weight_next_transpose = matrix_transposition(NN[i+1], structure[i+2], structure[i+1]);
          double** weight_prevdelta_mult = matrix_multiplication(weight_next_transpose, structure[i+1], structure[i+2], delta, structure[i+2], 1);
          //free unused memory
          free_matrix(&delta, structure[i+2], 1);
          free_matrix(&weight_next_transpose, structure[i+1], structure[i+2]);
          
          double** mult_out = pointwise_multiplication(out, 0, structure[i+1], 1, sub_result, structure[i+1], 1);
          delta = pointwise_multiplication(mult_out, 0, structure[i+1], 1, weight_prevdelta_mult, structure[i+1], 1);
          free_matrix(&mult_out, structure[i+1], 1);
          free_matrix(&sub_result, structure[i+1], 1);
          free_matrix(&weight_prevdelta_mult, structure[i+1], structure[i+2]);
        }
        // compute new weights
        double** mult_result2 = pointwise_multiplication(NULL, learning_rate, structure[i+1], 1, delta, structure[i+1], 1);
        double** prev_layer_out = initialize_matrix(structure[i], 1, 0);
        if(i==0)
          copy_pointer(NULL, train_input_matrix[perm[n]], structure[i], 0, prev_layer_out, NULL, structure[i], 1);
        else
          copy_pointer(NULL, out_NN[i-1], structure[i], 0, prev_layer_out, NULL, structure[i], 1); // out_NN[i-1] -> kasama output lang no input, so pag i=hidden layer i is pointing sa output i-1 sa hidden layer before it, shift ng 1 sa structure
        double** transpose = matrix_transposition(prev_layer_out, structure[i], 1);
        double** delta_mult = matrix_multiplication(mult_result2, structure[i+1], 1, transpose, 1, structure[i]);
        double** new_weight = matrix_addition(NN[i], 0, structure[i+1], structure[i], delta_mult, 0, structure[i+1], structure[i], 0);
        copy_pointer(new_weight, NULL, structure[i+1], structure[i], NN[i], NULL, structure[i+1], structure[i]);
        
        // compute biases
        double** layer_bias = initialize_matrix(structure[i+1], 1, 0);
        copy_pointer(NULL, bias[i], structure[i+1], 0, layer_bias, NULL, structure[i+1], 1);
        double** new_bias = matrix_addition(layer_bias, 0, structure[i+1], 1, mult_result2, 0, structure[i+1], 1, 0);
        copy_pointer(new_bias, NULL, structure[i+1], 1, NULL, bias[i], structure[i+1], 0);        
        // free variables
        free_matrix(&mult_result2, structure[i+1], 1);
        free_matrix(&out, structure[i+1], 1);
        free_matrix(&transpose, 1, structure[i]);
        free_matrix(&delta_mult, structure[i+1], structure[i]);
        free_matrix(&new_weight, structure[i+1], structure[i]);
        free_matrix(&layer_bias, structure[i+1],1);
        free_matrix(&new_bias, structure[i+1],1);
        out = prev_layer_out;
      }
      free_matrix(&delta, structure[1], 1); // free deltas of first hidden layer
      free_matrix(&out, structure[0], 1);
      // double** err_mult = pointwise_multiplication(err,0,structure[hidden_layers+1], 1, err, structure[hidden_layers+1], 1);
      // // sum errors for mean square
      // for(int i=0; i<structure[hidden_layers+1];i++){
      //   totalerr[iter] += err_mult[i][0];
      // }
    }

    // validation stage
    for(int n=0; n<validation_instances; n++){
      // Pick an input and out output
      set_output_nodes(validation_output_vector[n], structure[hidden_layers+1], &desired);

      double** in = initialize_matrix(structure[0],1, 0);
      copy_pointer(NULL, validation_input_matrix[n], structure[0], 0, in, NULL, structure[0], 1);
      
      // forward pass
      double** out;
      for(int i=0; i<=hidden_layers; i++){
        double** layer_bias = initialize_matrix(structure[i+1], 1, 0);
        copy_pointer(NULL, bias[i], structure[i+1], 0, layer_bias, NULL, structure[i+1], 1);
        double** v_mul = matrix_multiplication(NN[i],structure[i+1],structure[i],in,structure[i],1);
        free_matrix(&in,structure[i],1);
        double** v_add = matrix_addition(v_mul,0,structure[i+1],1,layer_bias,0,structure[i+1],1,0);
        free_matrix(&v_mul,structure[i+1],1);
        out = matrix_sigmoid(v_add, structure[i+1],1);
        free_matrix(&v_add,structure[i+1],1);
        free_matrix(&layer_bias,structure[i+1],1);
        in = out;
      }
      err = matrix_addition(desired, 0, structure[hidden_layers+1], 1, out, 0, structure[hidden_layers+1],1,1);

      double** err_mult = pointwise_multiplication(err,0,structure[hidden_layers+1], 1, err, structure[hidden_layers+1], 1);
      // sum errors
      for(int i=0; i<structure[hidden_layers+1];i++){
        totalerr[iter] += err_mult[i][0];
      }
      free_matrix(&err, structure[hidden_layers+1], 1);
      free_matrix(&out, structure[hidden_layers+1], 1);
      free_matrix(&err_mult, structure[hidden_layers+1], 1);
    }
    // // last lang
    // double** err_mult = pointwise_multiplication(err,0,structure[hidden_layers+1], 1, err, structure[hidden_layers+1], 1);
    // // sum errors
    // for(int i=0; i<structure[hidden_layers+1];i++){
    //   totalerr[iter] += err_mult[i][0];
    // }

    // Mean square
    totalerr[iter] = totalerr[iter]/validation_instances;
    char err_str[20];
    sprintf(err_str, "%lf\n", totalerr[iter]);
    printf("%s",err_str);
    fputs(err_str, err_file);
    // Print update
    if(iter%500 == 0)
      printf("Iteration: %d Error: %lf\n", iter, totalerr[iter]);
    // stopping condition
    if(totalerr[iter]<0.001){
      printf("Completed after %d iterations. Error is %lf\n", iter, totalerr[iter]);
      break;
    }
    free(perm);
  }
  // for(int i=0; i<hidden_layers+1; i++){
  //   for(int j=0; j<structure[i+1]; j++){
  //     for(int k=0; k<structure[i]; k++){
  //       printf("%lf", NN[i][j][k]);
  //       if(k<structure[i]-1)
  //         printf(",");
  //     }
  //     printf("\n");
  //   }
  // }
  // free variables
  free(totalerr);
  free_matrix(&desired, structure[hidden_layers+1], 1);
  free_matrix(&out_NN, hidden_layers+1, structure[hidden_layers+1]);
  fclose(err_file);
 }


void test_neural_net(int* structure,
                     int hidden_layers,
                     int test_instances,
                     double** test_input_matrix,
                     double*** NN,
                     double** bias,
                     int orig_min,
                     char* out_filename){
  FILE* out_file = fopen(out_filename, "w");
  for(int n=0; n<test_instances; n++){
    double** out;
    double** in = initialize_matrix(structure[0],1, 0);
    copy_pointer(NULL, test_input_matrix[n], structure[0], 0, in, NULL, structure[0], 1);  
    for(int i=0; i<=hidden_layers; i++){
      double** layer_bias = initialize_matrix(structure[i+1], 1, 0);
      copy_pointer(NULL, bias[i], structure[i+1], 0, layer_bias, NULL, structure[i+1], 1);
      double** v_mul = matrix_multiplication(NN[i],structure[i+1],structure[i],in,structure[i],1);
      free_matrix(&in,structure[i],1);
      double** v_add = matrix_addition(v_mul,0,structure[i+1],1,layer_bias,0,structure[i+1],1,0);
      free_matrix(&v_mul,structure[i+1],1);
      out = matrix_sigmoid(v_add, structure[i+1],1);
      free_matrix(&v_add,structure[i+1],1);
      free_matrix(&layer_bias,structure[i+1],1);
      in = out;
    }
    int output = get_output_in_decimal(structure[hidden_layers+1],out);
    free_matrix(&out, structure[hidden_layers+1], 1);
    char out_str[20];
    sprintf(out_str, "%d\n", output + orig_min);
    fputs(out_str, out_file);
  }
  fclose(out_file);
}

/*
 * PARAMS
 * --train-file train_file
 * --train-output desired output of train data
 * --train-instances number of training instances
 * --validation-file validation_file
 * --validation-instances number of validation instances
 * --validation-output desired output of validation file
 * --test-file test_file
 * --test-instances number of testing instances
 * --output output name for test file
 * --structure comma separated no space sequence of numbers
 * --learning-rate learning rate
 * --max-epoch max_epoch
 * --help help file
 */
int main(int argc, char *argv[]){
  // seed for randomness
  time_t t;
  srand((unsigned) time(&t));

  // Declare Variables
  char train_file[255];
  char desired_train_out[255];
  int train_instances;
  char validation_file[255];
  char desired_validation_out[255];
  int validation_instances;
  char test_file[255];
  char desired_test_out[255];
  int test_instances;
  char test_out[255];
  char total_err_file[255];
  int hidden_layers;
  char struct_string[255];
  int* structure;
  double learning_rate;
  int max_epoch;
  char weights_file[255];
  char bias_file[255];
  int orig_min = 0;

  int test_only = 0;
  FILE *bf, *nf;

  if(argc<2){
    fprintf(stderr, "ERROR: Missing config file!\n");
    exit(EXIT_FAILURE); 
  }
  if(argc>2){
    fprintf(stderr, "ERROR: Too many arguments!\n");
    exit(EXIT_FAILURE);
  }

  // Open Config File
  char* config_file = argv[1];
  printf("Opening %s\n", config_file);

  FILE* config=fopen(config_file, "r");
  char line[5000];
  int count=0;
  while(fgets(line, 5000, config)){
    if(line[0]=='#')
      continue;
    count++;
    line[strcspn(line, "#")] = 0; //remove comment
    line[strcspn(line, " ")] = 0; //remove spaces
    line[strcspn(line, "\n")] = 0; // remove newline
    char*l = line+strcspn(line, "=")+1;
    if(strlen(l)==0){
      fprintf(stderr, "ERROR: Missing params: %s? !\n", line);
      exit(EXIT_FAILURE);
    }
    if(check_prefix("TrainFile=", line))
      strcpy(train_file, l);
    else if(check_prefix("DesiredTrainOutFile=", line))
      strcpy(desired_train_out, l);
    else if(check_prefix("TrainInstances=", line))
      train_instances = atoi(l);
    else if(check_prefix("ValidationFile=", line))
      strcpy(validation_file, l);
    else if(check_prefix("DesiredValidationOutFile=", line))
      strcpy(desired_validation_out, l);
    else if(check_prefix("ValidationInstances=", line))
      validation_instances = atoi(l);
    else if(check_prefix("TestFile=", line))
      strcpy(test_file, l);
    else if(check_prefix("TestOutFile=", line))
      strcpy(test_out, l);
    else if(check_prefix("TestInstances=", line))
      test_instances = atoi(l);
    else if(check_prefix("TotalErrFile=", line))
      strcpy(total_err_file, l);
    else if(check_prefix("HiddenLayers=", line))
      hidden_layers = atoi(l);
    else if(check_prefix("Structure=", line))
      strcpy(struct_string, l);
    else if(check_prefix("LearningRate=", line))
      learning_rate = atof(l);
    else if(check_prefix("MaxEpoch=", line))
      max_epoch = atoi(l);
    else if(check_prefix("WeightsFile=", line))
      strcpy(weights_file, l);
    else if(check_prefix("BiasFile=", line))
      strcpy(bias_file, l);
    else if(check_prefix("OriginalMinimum=", line))
      orig_min = atoi(l);
  }

  // Close file
  fclose(config);
  
  // Check params
  if((nf = fopen(weights_file,"r")) && (bf = fopen(bias_file,"r"))){ // needs only test file, test out, instances, struct, hl, weights, and bias
    if(count<7){
      fprintf(stderr, "ERROR: Missing params in config file!\n");
      exit(EXIT_FAILURE);  
    }
    test_only = 1;
  }
  if(!test_only && count<16){
    fprintf(stderr, "ERROR: Missing params in config file!\n");
    exit(EXIT_FAILURE);
  }

  // Initialize structure
  structure = (int*) calloc(hidden_layers + 2, sizeof(int));
  char* struct_dup = strdup(struct_string);
  char* struct_ptr = struct_dup;
  count = 0;
  char* token;
  while((token=strsep(&struct_dup, ","))){
    structure[count++] = atoi(token);
  }
  free(struct_ptr);

  double*** NN;
  double** bias;
  // You train
  if(!test_only){
    // Copy train and validation dataset and labels
    printf("Copying training dataset...\n");
    double** train_matrix = initialize_matrix(train_instances, structure[0], 0);
    double** validation_matrix = initialize_matrix(validation_instances, structure[0], 0);
    int* train_labels = (int*) calloc(train_instances, sizeof(int));
    int* validation_labels = (int*) calloc(validation_instances, sizeof(int));
    read_dataset(&train_matrix, train_instances, structure[0], train_file);
    read_dataset(&validation_matrix, validation_instances, structure[0], validation_file);

    int orig_min = read_labels(&train_labels, train_instances, structure[hidden_layers+1], desired_train_out);

    // initialize network weights
    NN = initialize_network(structure, hidden_layers);

    bias = initialize_bias(structure, hidden_layers);

    train_neural_network(max_epoch, structure, hidden_layers, train_instances, train_matrix, train_labels, validation_instances, validation_matrix, validation_labels, learning_rate, NN, bias, total_err_file);
    
    nf = fopen(weights_file,"w");
    bf = fopen(bias_file,"w");
    for(int i=0; i<hidden_layers+1; i++){
      for(int j=0; j<structure[i+1]; j++){
        char str[1000] = "";
        for(int k=0; k<structure[i]; k++){
          sprintf(str, "%lf", NN[i][j][k]);
          fputs(str, nf);
          if(k<structure[i]-1)
            fputs(",", nf);
        }
        fputs("\n", nf);
      }
    }
    fclose(nf);
    for(int i=0; i<hidden_layers+1; i++){
      char str[1000] = "";
      for(int j=0; j<structure[i+1]; j++){\
        sprintf(str, "%lf", bias[i][j]);
        fputs(str, bf);
        if(j<structure[i+1]-1)
          fputs(",", bf);
      }
      fputs("\n", bf);
    }
    fclose(bf);
    free_matrix(&train_matrix,train_instances, structure[0]);
    free_matrix(&validation_matrix,validation_instances, structure[0]);
    free(train_labels);
    free(validation_labels);
  }else{ // Get existing NN and bias
    NN = (double***) calloc((hidden_layers+1), sizeof(double**));
    for(int i=0; i<=hidden_layers; i++){
      NN[i] = (double**) calloc(structure[i+1], sizeof(double*));
      for(int j=0; j<structure[i+1]; j++){
        int bufflen = structure[i+1] * cell_length;
        char line[bufflen];
        fgets(line, bufflen, nf);
        char* dup = strdup(line);
        char* ptr = dup;
        char* token;
        NN[i][j] = (double*) calloc(structure[i], sizeof(double));
        for(int k=0; k<structure[i]; k++){
          token = strsep(&dup, ",");
          NN[i][j][k] = atof(token);
        }
        free(ptr);  
      }
    }

    bias = (double**) calloc((hidden_layers+1), sizeof(double*));
    for(int i=0; i<=hidden_layers; i++){
      bias[i] = (double*) calloc(structure[i+1], sizeof(double));
      int bufflen = structure[i+1] * cell_length;
      char line[bufflen];
      fgets(line, bufflen, bf);
      char* dup = strdup(line);
      char* ptr = dup;
      char* token;
      for(int j=0; j<structure[i+1]; j++){
        token = strsep(&dup, ",");
        bias[i][j] = atof(token);
      }
      free(ptr);
    }

  }
  


  // copy test dataset
  printf("Copying test dataset...\n");
  double** test_matrix = initialize_matrix(test_instances, structure[0], 0);
  read_dataset(&test_matrix, test_instances, structure[0], test_file);
  test_neural_net(structure, hidden_layers, test_instances, test_matrix, NN, bias, orig_min, test_out);
  
  // free variables
  free_matrix(&test_matrix,test_instances, structure[0]);
  

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
  free(structure);
  return 0;
}
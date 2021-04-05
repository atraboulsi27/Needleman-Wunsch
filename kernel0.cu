
#include "common.h"
#include "timer.h"

#define BLOCK_SIZE 1024

__global__ void nw_kernel(unsigned char* reference, unsigned char* query, int* output_matrix, unsigned int N, unsigned int iteration_number) {


    // Get the position of the block inside the matrix.
    unsigned int block_row = blockIdx.x;
    unsigned int block_col = iteration_number - block_row; 
    

    // Transformation of the 1D thread vector into a 2D diagonal thread vector.
    unsigned int thread_diag_pos_x = (BLOCK_SIZE - 1) - threadIdx.x;
    unsigned int thread_diag_pos_y = threadIdx.x;

    for( unsigned int diagonal = 0; diagonal < BLOCK_SIZE; diagonal++ ) {

        // Get the position of the thread inside the block
        int pos_in_block_x = (BLOCK_SIZE - 1) - thread_diag_pos_x;
        int pos_in_block_y = diagonal - pos_in_block_x;

        // Calculate the position of the thread inside the matrix.
        int pos_in_matrix =  block_row * N + block_col; 

        // Calculate left, top, and top-left.
        int top = (thread_diag_pos_x == (BLOCK_SIZE - 1) && blockIdx.x == 0) ? (iteration_number + 1) : matrix[ pos_in_matrix - N ];
        int left = (thread_diag_pos_y == (BLOCK_SIZE - 1) && blockIdx.x == gridDim.x - 1) ? () : matrix[ pos_in_matrix - 1];
        int topleft = () ? () : matrix[ pos_in_matrix - N - 1];

    }

}

void nw_run(unsigned char* reference_d, unsigned char* query_d, int* matrix_d, unsigned int N, unsigned int iteration_number) {

    // Configure next run
    unsigned int numThreadsPerBlock = BLOCK_SIZE;
    unsigned int numBlocks = (iteration_number < (N + BLOCK_SIZE - 1) / BLOCK_SIZE) ? (iteration_number + 1) : (2 * (N + BLOCK_SIZE - 1) / BLOCK SIZE - iteration_number - 1);
    
    nw_kernel<<<numBlocks, numThreadsPerBlock>>>(reference_d, query_d, matrix_d, N, iteration_number);

}

void nw_gpu0(unsigned char* reference_d, unsigned char* query_d, int* matrix_d, unsigned int N) {



}


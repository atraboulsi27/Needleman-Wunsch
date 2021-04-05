
#include "common.h"
#include "timer.h"

#define BLOCK_SIZE 1024

__global__ void nw_kernel(unsigned char* reference, unsigned char* query, int* output_matrix, unsigned int N, unsigned int iteration_number) {


    for( unsigned int diagonal = 0; diagonal < BLOCK_SIZE; diagonal++ ) {

        // Verify that the diagonal thread index does not exceed the maximum number of elements allowed by the diagonal at this iteration.
        if( threadIdx.x <= diagonal  ) {

            // Get the position of the thread inside the block.
            int pos_in_block_x = threadIdx.x;
            int pos_in_block_y = diagonal - pos_in_block_x;

            // Calculate the positions of the thread inside the matrix.
            int mat_row = iteration_number * blockDim.x + pos_in_block_y;
            int mat_col = iteration_number * blockDim.x + pos_in_block_x;
            
            if( mat_row < N && mat_col < N ) {

                // Calculate value left, top, and top-left neighbors.
                int top = 
                    (mat_row == 0) ? 
                        (iteration_number + 1)*DELETION : matrix[ (mat_row - 1)*N + mat_col ];
                
                int left = 
                    (mat_col == 0) ? 
                        (iteration_number + 1)*INSERTION : matrix[ mat_row*N + (mat_col - 1) ];
                
                int topleft = 
                    (mat_row == 0) ? 
                        (iteration_number + 1)*DELETION : (mat_col == 0) ? 
                            (iteration_number + 1)*INSERTION : matrix[ (mat_row - 1)*N + (mat_col - 1) ];

                // Determine scores of the three possible outcomes: insertion, deletion, and match.
                int insertion = top  + INSERTION;
                int deletion  = left + DELETION;

                // Get the characters to verify if there is a match.
                char ref_char   = reference[];
                char query_char = query[];

                int match = topleft + (ref_char == query_char) ? MATCH : MISMATCH;
                
                // Select the maximum between the three.
                int max = (insertion > deletion) ? insertion : deletion;
                max = (match > max) ? match : max; 

                // Update the matrix at the correct position
                matrix[  mat_row*N + mat_col ] = max;
                
            }
        }

        __syncthreads();

    }


}

void nw_run(unsigned char* reference_d, unsigned char* query_d, int* matrix_d, unsigned int N, unsigned int iteration_number) {

    // Configure next run
    unsigned int numThreadsPerBlock = BLOCK_SIZE;
    unsigned int numBlocks = (iteration_number < (N + BLOCK_SIZE - 1) / BLOCK_SIZE) ? (iteration_number + 1) : (2 * (N + BLOCK_SIZE - 1) / BLOCK SIZE - iteration_number - 1);
    
    nw_kernel<<<numBlocks, numThreadsPerBlock>>>(reference_d, query_d, matrix_d, N, iteration_number);

}

void nw_gpu0(unsigned char* reference_d, unsigned char* query_d, int* matrix_d, unsigned int N) {

    for(int iter=0; iter < 2*N; iter++) {

        nw_run(reference_d, query_d, matrix_d, N, iter);

        cudaDeviceSynchronize();

    }

}


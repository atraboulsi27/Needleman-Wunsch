
#include "common.h"
#include "timer.h"

#define BLOCK_SIZE 1024

__global__ void nw_kernel(unsigned char* reference, unsigned char* query, int* matrix, unsigned int N, int iteration_number) {

    // Transform 1D Grid Coordinates into 2D Diagonal Coordinates.
    int diagonal_block_row = blockIdx.x;
    int diagonal_block_col = iteration_number - diagonal_block_row;

    // Get the effective coordinates of the block int the matrix.
    int block_row = diagonal_block_row * blockDim.x;
    int block_col = diagonal_block_col * blockDim.x;

    for( int diagonal = 0; diagonal < BLOCK_SIZE; diagonal++ ) {

        // Verify that the diagonal thread index does not exceed the maximum number of elements allowed by the diagonal at this iteration.
        if( threadIdx.x <= diagonal  ) {

            // Get the position of the thread inside the block.
            int pos_in_block_x = threadIdx.x;
            int pos_in_block_y = diagonal - pos_in_block_x;

            // Calculate the positions of the thread inside the matrix.
            int mat_row = block_row + pos_in_block_y;
            int mat_col = block_col + pos_in_block_x;
            
            if( mat_row < N && mat_col < N ) {

                // Calculate value left, top, and top-left neighbors.
                int top = 
                    (mat_row == 0) ? 
                        ((mat_col + 1)*DELETION) : matrix[ (mat_row - 1)*N + mat_col ];
                
                int left = 
                    (mat_col == 0) ? 
                        ((mat_row + 1)*INSERTION) : matrix[ mat_row*N + (mat_col - 1) ];
                
                int topleft = 
                    (mat_row == 0) ? 
                        (mat_col*DELETION) : (mat_col == 0) ? 
                            (mat_row*INSERTION) : matrix[ (mat_row - 1)*N + (mat_col - 1) ];

                // Determine scores of the three possible outcomes: insertion, deletion, and match.
                int insertion = top  + INSERTION;
                int deletion  = left + DELETION;

                // Get the characters to verify if there is a match.
                char ref_char   = reference[mat_col];
                char query_char = query[mat_row];

                int match = topleft + ( (ref_char == query_char) ? MATCH : MISMATCH );
                
                // Select the maximum between the three.
                int max = (insertion > deletion) ? insertion : deletion;
                max = (match > max) ? match : max; 

                // Update the matrix at the correct position
                matrix[  mat_row*N + mat_col ] = max;
                
            }
        }

        __syncthreads();

    }

    for( int diagonal = BLOCK_SIZE; diagonal < 2*BLOCK_SIZE; diagonal++ ) {

        if( threadIdx.x < 2*BLOCK_SIZE - diagonal ) {

            int pos_in_block_x = BLOCK_SIZE - threadIdx.x - 1;
            int pos_in_block_y = diagonal - pos_in_block_x - 1;

            int mat_row = block_row + pos_in_block_y;
            int mat_col = block_col + pos_in_block_x;

            if( mat_row < N && mat_col < N ) {

                int top = 
                    (mat_row == 0) ? 
                        ((mat_col + 1)*DELETION) : matrix[ (mat_row - 1)*N + mat_col ];
                
                int left = 
                    (mat_col == 0) ? 
                        ((mat_row + 1)*INSERTION) : matrix[ mat_row*N + (mat_col - 1) ];
                
                int topleft = 
                    (mat_row == 0) ? 
                        (mat_col*DELETION) : (mat_col == 0) ? 
                            (mat_row*INSERTION) : matrix[ (mat_row - 1)*N + (mat_col - 1) ];

                int insertion = top  + INSERTION;
                int deletion  = left + DELETION;

                char ref_char   = reference[mat_col];
                char query_char = query[mat_row];

                int match = topleft + ( (ref_char == query_char) ? MATCH : MISMATCH );
                
                int max = (insertion > deletion) ? insertion : deletion;
                max = (match > max) ? match : max; 

                if (mat_row == 0 && mat_col == 1023)
                    printf("row: %d, col: %d, max: %d, del: %d, ins: %d, match: %d, %c, %c\n", mat_row, mat_col, max, deletion, insertion, match, ref_char, query_char);

                matrix[  mat_row*N + mat_col ] = max;
                
            }
        }

        __syncthreads();
    }
}


void nw_gpu0(unsigned char* reference_d, unsigned char* query_d, int* matrix_d, unsigned int N) { 
    
    unsigned int numThreadsPerBlock = BLOCK_SIZE;

    for(int iter=0; iter < (N + BLOCK_SIZE - 1) / BLOCK_SIZE; iter++) {

        // Configure next run
        unsigned int numBlocks = (iter < (N + BLOCK_SIZE - 1) / BLOCK_SIZE) ? (iter + 1) : (2 * (N + BLOCK_SIZE - 1) / BLOCK_SIZE - iter - 1);
      
        printf("%d\n", numBlocks);
        // Launch kernel
        nw_kernel<<<numBlocks, numThreadsPerBlock>>>(reference_d, query_d, matrix_d, N, iter);
        
        cudaDeviceSynchronize();

    }

}


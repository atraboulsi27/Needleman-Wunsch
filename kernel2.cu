

#include "common.h"
#include "timer.h"

#define BLOCK_SIZE 32
#define COVERAGE 2


__device__ void private_nw_function(unsigned char* reference, unsigned char* query, int* matrix, unsigned int N, int matrix_s[BLOCK_SIZE][BLOCK_SIZE], int pos_x, int pos_y, int mat_row, int mat_col) {


    // Calculate value left, top, and top-left neighbors.
    int top = 
    (mat_row == 0) ? 
        ((mat_col + 1)*DELETION) : (pos_y == 0) ?
            matrix[ (mat_row - 1)*N + mat_col ] : matrix_s[pos_y - 1][pos_x   ];

    int left = 
        (mat_col == 0) ? 
            ((mat_row + 1)*INSERTION) : (pos_x == 0) ?
                matrix[ mat_row*N + (mat_col - 1) ] : matrix_s[pos_y   ][pos_x - 1];

    int topleft = 
        (mat_row == 0) ? 
            (mat_col*DELETION) : (mat_col == 0) ? 
                (mat_row*INSERTION) : (pos_y == 0 || pos_x == 0) ? 
                    matrix[ (mat_row - 1)*N + mat_col - 1 ] : matrix_s[pos_y - 1][pos_x - 1]; 

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
    matrix_s[ pos_y ][ pos_x ] = max;

}

__global__ void nw_kernel2(unsigned char* reference, unsigned char* query, int* matrix, unsigned int N, int iteration_number) {


    // Transform 1D Grid Coordinates into 2D Diagonal Coordinates.
    int diagonal_block_row = blockIdx.x;
    int diagonal_block_col = iteration_number - diagonal_block_row;

    if( iteration_number > gridDim.x) {
        diagonal_block_row = ( (N + BLOCK_SIZE*COVERAGE - 1)/(BLOCK_SIZE*COVERAGE) ) - blockIdx.x - 1;
        diagonal_block_col = iteration_number - diagonal_block_row;
    } 

    int block_row = diagonal_block_row * BLOCK_SIZE * COVERAGE;
    int block_col = diagonal_block_col * BLOCK_SIZE * COVERAGE;
    
    __shared__ int matrix_s[BLOCK_SIZE][BLOCK_SIZE];

    if( threadIdx.x < BLOCK_SIZE ) {

        for( int diagonal = 0; diagonal < BLOCK_SIZE; diagonal++ ) {

            // Get the position of the thread inside the block.
            int pos_x = threadIdx.x;
            int pos_y = diagonal - pos_x;

            // Calculate the positions of the thread inside the matrix.
            int mat_row = block_row + pos_y;
            int mat_col = block_col + pos_x;
            
            if( mat_row < N && mat_col < N && pos_x < BLOCK_SIZE && pos_y < BLOCK_SIZE && pos_x >= 0 && pos_y >= 0) { 

                private_nw_function(reference, query, matrix, N, matrix_s, pos_x, pos_y, mat_row, mat_col); 
              
            }

        }

    } 

    __syncthreads(); 

    for(int i=0; i < COVERAGE * COVERAGE; i++) {

        int output_row = 0;

        for( int diagonal = 0; diagonal < BLOCK_SIZE; diagonal++ ) { 

            // Get the position of the thread inside the block.
            int pos_x = threadIdx.x;
            int pos_y = diagonal - pos_x;
            
            // Calculate the positions of the thread inside the matrix.
            int mat_row = block_row + ( (i+1) / COVERAGE ) * BLOCK_SIZE + pos_y;
            int mat_col = block_col + ( (i+1) % COVERAGE ) * BLOCK_SIZE + pos_x;
            
            if( threadIdx.x >= BLOCK_SIZE ) {

                pos_x = 2 * (BLOCK_SIZE) - threadIdx.x - 1; 
                pos_y = BLOCK_SIZE - pos_x + diagonal;
               
               // Calculate the positions of the thread inside the matrix.
                mat_row = block_row + ( i / COVERAGE ) * BLOCK_SIZE + pos_y;
                mat_col = block_col + ( i % COVERAGE ) * BLOCK_SIZE + pos_x;
           
            }
     
            if( threadIdx.x < BLOCK_SIZE ) {

                    int row = block_row + ( i / COVERAGE ) * BLOCK_SIZE + output_row;
                    int col = block_col + ( i % COVERAGE ) * BLOCK_SIZE + threadIdx.x;

                    if( row < N && col < N ) {

                        matrix[row * N + col] = matrix_s[output_row][threadIdx.x];

                    }

            }
            
            if( mat_row < N && mat_col < N && pos_x < BLOCK_SIZE && pos_y < BLOCK_SIZE && pos_x >= 0 && pos_y >= 0) {
        
                private_nw_function(reference, query, matrix, N, matrix_s, pos_x, pos_y, mat_row, mat_col);

            }
            
            __syncthreads();
    
            output_row++;

        }

        __syncthreads();

        /*if(iteration_number == 1 && threadIdx.x == 0 && blockIdx.x == 0) {

            printf("\n\n");
        
            for( int i=0; i<32; i++) {
                for(int j=0; j<32; j++) {
                    printf("%d, ", matrix_s[i][j]);
                }
                printf("\n");
            }

            printf("\n\n");

            for( int i=0; i<32; i++) {
                for(int j=64; j<(64+32); j++) {
                    printf("%d, ", matrix[i*N + j]);
                }
                printf("\n");
            }

            printf("\n");

        }*/


    }

    

}


void nw_gpu2(unsigned char* reference_d, unsigned char* query_d, int* matrix_d, unsigned int N) { 
    
    unsigned int numThreadsPerBlock = COVERAGE * BLOCK_SIZE;

    for(int iter=0; iter < 2* ( (N + BLOCK_SIZE * COVERAGE - 1) / (BLOCK_SIZE * COVERAGE)) - 1; iter++) {

        // Configure next run
        unsigned int numBlocks = (iter < (N + BLOCK_SIZE * COVERAGE - 1) / (BLOCK_SIZE * COVERAGE)) ? (iter + 1) : (2*((N + BLOCK_SIZE * COVERAGE - 1) / (BLOCK_SIZE * COVERAGE)) - iter - 1);

        //printf("iteration: %d, blocks: %d, threads: %d\n", iter, numBlocks, numThreadsPerBlock);
        // Launch kernel
        nw_kernel2<<<numBlocks, numThreadsPerBlock>>>(reference_d, query_d, matrix_d, N, iter);
        
        cudaDeviceSynchronize();

    }

}

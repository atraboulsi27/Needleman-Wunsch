

#include "common.h"
#include "timer.h"

#define BLOCK_SIZE 32
#define COVERAGE 4


__device__ void private_nw_function_3(unsigned char* reference, unsigned char* query, int* matrix, unsigned int N, int matrix_s[BLOCK_SIZE][BLOCK_SIZE], int pos_x, int pos_y, int mat_row, int mat_col) {


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

    matrix[ mat_row * N + mat_col ] = max;

}

__global__ void nw_kernel3(unsigned char* reference, unsigned char* query, int* matrix, unsigned int N, int iteration_number) {

    // Transform 1D Grid Coordinates into 2D Diagonal Coordinates.
    int diagonal_block_row = blockIdx.x;
    int diagonal_block_col = iteration_number - diagonal_block_row;

    if( iteration_number > gridDim.x) {
        diagonal_block_row = ( (N + BLOCK_SIZE*COVERAGE - 1)/(BLOCK_SIZE*COVERAGE) ) - blockIdx.x - 1;
        diagonal_block_col = iteration_number - diagonal_block_row;
    }

    int block_row = diagonal_block_row * BLOCK_SIZE * COVERAGE;
    int block_col = diagonal_block_col * BLOCK_SIZE * COVERAGE;

    // Initialise Shared Memory
    __shared__ int matrix_s[BLOCK_SIZE][BLOCK_SIZE];

    // Setup thread activation
    if(  )
    for( int i=0; i < BLOCK_SIZE; i++ ) {



    }

    bool isThreadActive = (threadIdx.x + threadIdx.y) % 2 == 0;
    bool shouldThreadLoadNewValue = false; 
 

    if( threadIdx.x < BLOCK_SIZE ) {

        for( int diagonal = 0; diagonal < BLOCK_SIZE; diagonal++ ) {

            // Get the position of the thread inside the block.
            int pos_x = threadIdx.x;
            int pos_y = diagonal - pos_x;

            // Calculate the positions of the thread inside the matrix.
            int mat_row = block_row + pos_y;
            int mat_col = block_col + pos_x;
            
            if( mat_row < N && mat_col < N && pos_x < BLOCK_SIZE && pos_y < BLOCK_SIZE && pos_x >= 0 && pos_y >= 0) { 

                private_nw_function_3(reference, query, matrix, N, matrix_s, pos_x, pos_y, mat_row, mat_col); 
              
            }

        }

    }

    __syncthreads(); 

    int diagFirstPos = (threadIdx.x / BLOCK_SIZE) * 2;
    bool isSecondDiag = false;

    int diagPos = diagFirstPos + isSecondDiag;

    pos_x = threadIdx.x;
    pos_y = diagonalPos - pos_x;

    if( pos_x < BLOCK_SIZE && pos_y < BLOCK_SIZE && pos_x >= 0 && pos_y >= 0  ) {



    }

    isSecondDiag = !(isSecondDiag)

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

    }

    

}


void nw_gpu3(unsigned char* reference_d, unsigned char* query_d, int* matrix_d, unsigned int N) { 
    
    dim3 numThreadsPerBlock( BLOCK_SIZE, BLOCK_SIZE );

    mm_tiled_kernel<<< numBlocks, numThreadsPerBlock>>>( A_d, B_d, C_d, M, N, K);

    for(int iter=0; iter < 2* ( (N + BLOCK_SIZE * COVERAGE - 1) / (BLOCK_SIZE * COVERAGE)) - 1; iter++) {

        // Configure next run
        unsigned int numBlocks = (iter < (N + BLOCK_SIZE * COVERAGE - 1) / (BLOCK_SIZE * COVERAGE)) ? (iter + 1) : (2*((N + BLOCK_SIZE * COVERAGE - 1) / (BLOCK_SIZE * COVERAGE)) - iter - 1);

        //printf("iteration: %d, blocks: %d, threads: %d\n", iter, numBlocks, numThreadsPerBlock);
        // Launch kernel
        nw_kernel3<<<numBlocks, numThreadsPerBlock>>>(reference_d, query_d, matrix_d, N, iter);
        
        cudaDeviceSynchronize();

    }

}

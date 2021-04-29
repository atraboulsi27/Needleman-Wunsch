

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

    if( threadIdx.y == 0 ) {

        for( int diagonal=0; diagonal < BLOCK_SIZE; diagonal++ ) {

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

    bool isThreadActive = (threadIdx.x + threadIdx.y) % 2 == 0;

    int row = 0;
    int col = 0;
    int diag = 0;

    for( int subBlock = 1; subBlock < COVERAGE * COVERAGE; subBlock++ ) {

        row++;
        col--;

        // AFTER GREATEST DIAGONAL
        if( diag >= COVERAGE ) {
            diag++;
            row = diag - COVERAGE + 1; 
            col = COVERAGE;
        }

        // RESET

        if(col < 0) {
            diag++;
            row = 0;
            col = diag;
        }

        if ( diag >= COVERAGE && col < diag - COVERAGE + 1 ) {
            diag++;
            row = diag - COVERAGE + 1; 
            col = COVERAGE; 
        }

        __syncthreads();

        // LOAD NEXT SUB-BLOCK
        if( threadIdx.x == 0 && threadIdx.y == 0 ) {

            // Calculate the positions of the thread inside the matrix.
            int mat_row = block_row + row * BLOCK_SIZE;
            int mat_col = block_col + col * BLOCK_SIZE;

            if ( mat_row < N && mat_col < N ) {
                private_nw_function_3(reference, query, matrix, N, matrix_s, 0, 0, mat_row, mat_col);
            }

        }

        __syncthreads();

        int pos_x = threadIdx.x;
        int pos_y = threadIdx.y;

        int current_diagonal = (pos_x + pos_y) / 2;

        int subBlock_row = row - current_diagonal;
        int subBlock_col = diag - subBlock_row;

        if( subBlock_row < 0 && diag < COVERAGE ) {

            int num_jumps = (-1 * subBlock_row) - 1;
            int diag_jumps = 1;

            subBlock_row = diag - diag_jumps;
            subBlock_col = 0;

            while( num_jumps > 0 ) {
            
                subBlock_row--;
                subBlock_col++;
            
                if( subBlock_row < 0 ) {

                    diag_jumps++;

                    subBlock_row = diag - diag_jumps;
                    subBlock_col = 0;

                }

                num_jumps--;

            }

            if( subBlock == 9 ) {

                printf("num_jumps: %d, x: %d, y: %d, row: %d, col: %d\n", num_jumps, threadIdx.x, threadIdx.y, subBlock_row, subBlock_col);
    
            }

            if ( subBlock_row < 0 ) {
                subBlock_row = 0;
                subBlock_col = 0;
            }

        }

        if( subBlock_row < diag - COVERAGE + 1 && diag >= COVERAGE ) {

            int num_jumps = (diag - COVERAGE + 1) - subBlock_row - 1;
            int diag_jumps = 1;

            subBlock_row = COVERAGE;
            subBlock_col = (diag - diag_jumps) - COVERAGE + 1;

            while( num_jumps > 0 ) {
            
                subBlock_row--;
                subBlock_col++;
            
                if( subBlock_row < diag - COVERAGE + 1 ) {

                    diag_jumps++;

                    subBlock_row = COVERAGE;
                    subBlock_col = (diag - diag_jumps) - COVERAGE + 1;

                }

                if( diag_jumps <= 0 ) {
                    subBlock_row = 0;
                    subBlock_col = 0;
                    break;
                }

                num_jumps--;

            }

            if ( subBlock_row < 0 ) {
                subBlock_row = 0;
                subBlock_col = 0;
            }

        }

        __syncthreads();

        int mat_row = block_row + subBlock_row * BLOCK_SIZE + pos_y;
        int mat_col = block_col + subBlock_col * BLOCK_SIZE + pos_x;

        // First Activation
        if( isThreadActive && threadIdx.x != 0 && threadIdx.x != 0 ) {

            // if( subBlock == 2 ) {
            //     printf( "mat_col: %d\n", mat_col);
            // }

            if( mat_row < N && mat_col < N ) {

                private_nw_function_3(reference, query, matrix, N, matrix_s, pos_x, pos_y, mat_row, mat_col);

            }
    
        }

        __syncthreads();

        isThreadActive = !(isThreadActive);


        // SECOND ACTIVATION
        if( isThreadActive ) {

            if( mat_row < N && mat_col < N ) {
        
                private_nw_function_3(reference, query, matrix, N, matrix_s, pos_x, pos_y, mat_row, mat_col);

            }

        }

        isThreadActive = !(isThreadActive);

        __syncthreads();

        // if( subBlock == 11 && threadIdx.x == 0 && threadIdx.y == 0 ) {

        //     printf("%d, %d", mat_row, mat_col);

            // printf("\n");
            // for( int i=32*0; i<32*1; i++ ) {
            //     for( int j=32*1; j<32*2; j++ ) {
            //         printf( "%d, ", matrix[i*N + j]);
            //     }
            //     printf("\n");
            // }

            // printf("\n");
            // for( int i=0; i<32; i++ ) {
            //     for( int j=0; j<32; j++ ) {
            //         printf( "%d, ", matrix_s[i][j]);
            //     }
            //     printf("\n");
            // }
        // }


        __syncthreads();

    }

    if( threadIdx.x == 0 && threadIdx.y == 0 ) {

        printf("\nResult:\n\n");
        for( int i=32*0; i<32*1; i++ ) {
            for( int j=32*0; j<32*1; j++ ) {
                printf( "%d, ", matrix[i*N + j]);
            }
            printf("\n");
        }

    }
    

}


void nw_gpu3(unsigned char* reference_d, unsigned char* query_d, int* matrix_d, unsigned int N) { 
    
    dim3 numThreadsPerBlock( BLOCK_SIZE, BLOCK_SIZE );

    for(int iter=0; iter < 2* ( (N + BLOCK_SIZE * COVERAGE - 1) / (BLOCK_SIZE * COVERAGE)) - 1; iter++) {

        // Configure next run
        unsigned int numBlocks = (iter < (N + BLOCK_SIZE * COVERAGE - 1) / (BLOCK_SIZE * COVERAGE)) ? (iter + 1) : (2*((N + BLOCK_SIZE * COVERAGE - 1) / (BLOCK_SIZE * COVERAGE)) - iter - 1);

        printf("iteration: %d, blocks: %d\n", iter, numBlocks);
        // Launch kernel
        nw_kernel3<<<numBlocks, numThreadsPerBlock>>>(reference_d, query_d, matrix_d, N, iter);
        
        cudaDeviceSynchronize();

    }

}


#include "common.h"
#include "timer.h"

// BLOCK SIZE is now limited to the size of shared memory
// 31 is chosen instead of 32 as 1 row and 1 col are assigned to elements from a previously computed block.
#define BLOCK_SIZE 32

#define IN_TILE_DIM 32
#define OUT_TILE_DIM ( (IN_TILE_DIM) - 1 )

__global__ void nw_kernel1(unsigned char* reference, unsigned char* query, int* matrix, unsigned int N, int iteration_number) {


    // Transform 1D Grid Coordinates into 2D Diagonal Coordinates.
    int diagonal_block_row = blockIdx.x;
    int diagonal_block_col = iteration_number - diagonal_block_row;

    if( iteration_number >= ( (N + OUT_TILE_DIM - 1)/OUT_TILE_DIM )) {
        diagonal_block_row = ( (N + OUT_TILE_DIM - 1)/OUT_TILE_DIM ) - blockIdx.x - 1;
        diagonal_block_col = iteration_number - diagonal_block_row;
    }

    __shared__ int matrix_s[IN_TILE_DIM][IN_TILE_DIM];

    // Load elements from the previous column (Non-Coalessed)
    if(diagonal_block_col != 0 && threadIdx.x < OUT_TILE_DIM && diagonal_block_row * OUT_TILE_DIM + threadIdx.x < N) {
        matrix_s[threadIdx.x+1][0] = matrix[(diagonal_block_row*OUT_TILE_DIM + threadIdx.x)*N + (diagonal_block_col * OUT_TILE_DIM-1)];
    }

    // Load elements from the previous row (Coalessed)
    if(diagonal_block_row != 0 && threadIdx.x < OUT_TILE_DIM && diagonal_block_col * OUT_TILE_DIM + threadIdx.x < N) {
        matrix_s[0][threadIdx.x+1] = matrix[(diagonal_block_row*OUT_TILE_DIM - 1)*N + (diagonal_block_col * OUT_TILE_DIM + threadIdx.x)];
    }

    if( threadIdx.x == 0 && diagonal_block_col > 0 && diagonal_block_row > 0) {
        matrix_s[0][0] = matrix[(diagonal_block_row * OUT_TILE_DIM - 1)*N + (diagonal_block_col * OUT_TILE_DIM - 1)];
    }


    __syncthreads();

    for( int diagonal = 0; diagonal < 2*OUT_TILE_DIM; diagonal++ ) {

        int thread_limit = (diagonal < OUT_TILE_DIM) ? (diagonal) : (2 * OUT_TILE_DIM - diagonal);

        // Verify that the diagonal thread index does not exceed the maximum number of elements allowed by the diagonal at this iteration.
        if( threadIdx.x <= thread_limit ) {

            // Get the position of the thread inside the block.
            int pos_x = threadIdx.x;
            int pos_y = diagonal - pos_x;

            if( diagonal > OUT_TILE_DIM ) {
                pos_x = OUT_TILE_DIM - threadIdx.x - 1;
                pos_y = diagonal - pos_x - 1;
            }

            // Find the positions of the thread in the output matrix.
            int mat_row = diagonal_block_row * OUT_TILE_DIM + pos_y;
            int mat_col = diagonal_block_col * OUT_TILE_DIM + pos_x;

            if( mat_row < N && mat_col < N && pos_x + 1 < IN_TILE_DIM && pos_y + 1 < IN_TILE_DIM) {

                // Calculate value left, top, and top-left neighbors.
                // FIND A WAY TO MOVE THOSE CONDITIONAL STATEMENTS OUT OF HERE.
                int top     = (mat_row == 0) ? ( (mat_col + 1) * DELETION ) : matrix_s[pos_y    ][pos_x + 1];
                int left    = (mat_col == 0) ? ( (mat_row + 1) * INSERTION) : matrix_s[pos_y + 1][pos_x    ];

                int topleft = 
                    (mat_row == 0) ? 
                        ( mat_col * DELETION ) : (mat_col == 0) ?
                            ( mat_row * INSERTION) : matrix_s[pos_y][pos_x]; 

                // Determine scores of the three possible outcomes: insertion, deletion, and match.
                int insertion = top  + INSERTION;
                int deletion  = left + DELETION;

                // Get the characters to verify if there is a match.
                char ref_char   = reference[ mat_col ];
                char query_char = query[ mat_row ];

                int match = topleft + ( (ref_char == query_char) ? MATCH : MISMATCH );
                

                // Select the maximum between the three.
                int max = (insertion > deletion) ? insertion : deletion;
                max = (match > max) ? match : max;

                // Update the matrix at the correct position
                matrix_s[pos_y + 1][pos_x + 1] = max;
            
            }
        }

        __syncthreads();

    }

    // Update the output matrix at the correct positions (Writes are coalsced).
    for(int i=0; i<OUT_TILE_DIM; i++) {
        if( diagonal_block_row * OUT_TILE_DIM + i < N && diagonal_block_col * OUT_TILE_DIM + threadIdx.x < N && threadIdx.x < OUT_TILE_DIM ) {
            matrix[ (diagonal_block_row*OUT_TILE_DIM + i)*N + (diagonal_block_col * OUT_TILE_DIM) + threadIdx.x ] = matrix_s[i+1][threadIdx.x+1];
        }
    }

}


void nw_gpu1(unsigned char* reference_d, unsigned char* query_d, int* matrix_d, unsigned int N) {
    
    unsigned int numThreadsPerBlock = IN_TILE_DIM;

    for(int iter=0; iter < 2* ( (N + OUT_TILE_DIM - 1) / OUT_TILE_DIM) - 1; iter++) {

        // Configure next run
        unsigned int numBlocks = (iter < (N + OUT_TILE_DIM - 1) / OUT_TILE_DIM) ? (iter + 1) : (2*((N + OUT_TILE_DIM - 1) / OUT_TILE_DIM) - iter - 1);

        // printf("%d, %d\n", iter, numBlocks);
        // Launch kernel
        nw_kernel1<<<numBlocks, numThreadsPerBlock>>>(reference_d, query_d, matrix_d, N, iter);
        
        cudaDeviceSynchronize();

        // if ( cudaGetLastError() != 0 ) {
        //     printf( "iteration: %d, blocks: %d, error: %d\n", iter, numBlocks, cudaGetLastError(), cudaGetErrorString);
        // }

    }

}


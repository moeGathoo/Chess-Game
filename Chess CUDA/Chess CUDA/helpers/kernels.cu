#include "../headers.cuh"

__global__ void kernel(cell* board, char* pieceTypes, int* pieceValues, int* scores) {
    int tRow = threadIdx.y; int tCol = threadIdx.x;
    int thread = tRow * BOARD_WIDTH + tCol;

    __shared__ cell lBoard[(BOARD_WIDTH)][(BOARD_WIDTH)];
    __shared__ char lPieceTypes[(PIECES)];
    __shared__ int lPieceValues[(PIECES)];

    lBoard[tRow][tCol] = board[tRow * BOARD_WIDTH + tCol];
    if (thread < PIECES) {
        lPieceTypes[thread] = pieceTypes[thread];
        lPieceValues[thread] = pieceValues[thread];
    }
    __syncthreads();

    char myPiece = lBoard[tRow][tCol].piece;
    if (myPiece != '-') {
        bool found = false; int idx = -1, pos = -1;
        for (int i = 0; i < PIECES; i++)
            if (myPiece == lPieceTypes[i]) {
                idx = i; pos = i;
                found = true; break;
            }
        if (!found) {
            myPiece = myPiece + 32;
            for (int i = 0; i < PIECES; i++)
                if (myPiece == lPieceTypes[i]) {
                    idx = i + 6; pos = i;
                    found = true; break;
                }
        }
        atomicAdd(&scores[idx], lPieceValues[pos]);
    }
    __syncthreads();
}
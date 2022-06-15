#include "headers.cuh"

//initializes values of all global variables
char pieceTypes[6] = { 'r', 'n', 'b', 'q', 'k', 'p' };    //types of pieces on board
int pieceValues[6] = { 50, 30, 30, 90, 900, 10 };         //piece weightings for search and evaluation algorithms
vector black[PIECES];
vector white[PIECES];

/**
 * @brief Main function to play game.
 *
 * @param argc Number of arguments.
 * @param argv Each argument pertains to a particular part of fen notation.
 * ? argv[0]: fen string representation of board
 * ? argv[1]: side to play
 * ? argv[2]: castling rights
 * ? argv[3]: en passant move
 * ? argv[4]: half moves (total moves since pawn move or piece capture)
 * ? argv[5]: full moves (increments by 1 after black moves)
 *
 * @return int
 */
int main(int argc, char* argv[]) {

    for (int i = 0; i < PIECES; i++) {
        initVector(&black[i]);
        initVector(&white[i]);
    }

    state game = initState(argv[1], *argv[2], argv[3], argv[4], atoi(argv[5]), atoi(argv[6]));    
    cell board[RANK][RANK];
    initBoard(board, game.fen);

    //for (int i = 0; i < 10; i++) {
    //    char move[5] = "";
    //    alphaBeta(board, game, 4, -10001, 10001, &move[0], true);
    //    printf("%s\n", move);
    //    checkMove(board, &game, move);
    //    if (game.gameOver) break; //end game of game is over
    //}

    /*/size_t numBytes = sizeof(int) * RANK * RANK;
    int hA[RANK][RANK], * dA = 0;
    int* hOutput = (int*)calloc(RANK * RANK, sizeof(int));
    for (int i = 0; i < RANK; i++)
        for (int j = 0; j < RANK; j++)
            hA[i][j] = 0;*/

    size_t numBytes = sizeof(cell) * BOARD_WIDTH * BOARD_WIDTH;
    int* hScores = (int*)malloc(PIECES * 2 * sizeof(int));

    cell* dBoard = 0;
    int* dScore = 0; int* dPieceValues = 0;
    char* dPieceTypes = 0; int* dScores = 0;

    cudaMalloc((void**)&dBoard, numBytes);
    cudaMalloc((void**)&dPieceValues, PIECES * sizeof(int));
    cudaMalloc((void**)&dPieceTypes, PIECES * sizeof(char));
    cudaMalloc((void**)&dScores, PIECES * 2 * sizeof(int));

    cudaMemcpy(dBoard, board, numBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(dPieceTypes, &pieceTypes, 6 * sizeof(char), cudaMemcpyHostToDevice);
    cudaMemcpy(dPieceValues, pieceValues, 6 * sizeof(int), cudaMemcpyHostToDevice);

    dim3 gridSize(1, 1); dim3 blockSize(RANK, RANK);
    kernel << <gridSize, blockSize >> > (dBoard, dPieceTypes, dPieceValues, dScores);
    cudaMemcpy(hScores, dScores, PIECES * 2 * sizeof(int), cudaMemcpyDeviceToHost);
    
    for (int i = 0; i < PIECES * 2; i++) {
        printf("%d ", hScores[i]);
    }

    cudaFree(dBoard);
    cudaFree(dPieceTypes);
    cudaFree(dPieceValues);
    cudaFree(dScores);
    free(hScores);

    /*for (int i = 0; i < RANK; i++) {
        for (int j = 0; j < RANK; j++) {
            printf("%c\t", hoBoard[i*BOARD_WIDTH + j].piece);
        }
        printf("\n");
    }*/

    for (int i = 0; i < PIECES; i++) {
        vectorFree(&black[i]);
        vectorFree(&white[i]);
    }

    return 0;
}
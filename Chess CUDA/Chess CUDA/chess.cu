#include "headers.cuh"

//initializes values of all global variables
char pieceTypes[6] = { 'r', 'n', 'b', 'q', 'k', 'p' };    //types of pieces on board
int pieceValues[6] = { 50, 30, 30, 90, 900, 10 };         //piece weightings for search and evaluation algorithms
cell board[RANK][RANK];
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
    initBoard(game.fen);
    
    for (int i = 0; i < 10; i++) {
        char move[5] = "";
        alphaBeta(game, 4, -10001, 10001, &move[0], true);
        printf("%s\n", move);
        checkMove(&game, move);
        if (game.gameOver) break; //end game of game is over
    }

    for (int i = 0; i < PIECES; i++) {
        vectorFree(&black[i]);
        vectorFree(&white[i]);
    }

    return 0;
}
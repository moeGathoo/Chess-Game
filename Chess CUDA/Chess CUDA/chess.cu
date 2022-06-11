#include "headers.cuh"

//initializes values of all global variables
char pieceTypes[6] = { 'r', 'n', 'b', 'q', 'k', 'p' };    //types of pieces on board
int pieceValues[6] = { 50, 30, 30, 90, 900, 10 };         //piece weightings for search and evaluation algorithms
cell board[RANK][RANK];
vector black[PIECES];
vector white[PIECES];

int main(int argc, char* argv[]) {

    for (int i = 0; i < PIECES; i++) {
        initVector(&black[i]);
        initVector(&white[i]);
    }

    state game = initState(argv[1], *argv[2], argv[3], argv[4], atoi(argv[5]), atoi(argv[6]));
    initBoard(game.fen);
    printBoard(); printf("\n");
    vector* moves = black;

    for (int i = 0; i < PIECES; i++) {
        vectorFree(&black[i]);
        vectorFree(&white[i]);
    }

    return 0;
}
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

    //char* posPtr = strchr(pieceTypes, 'q');
    //int idx = posPtr - pieceTypes;
    //printf("%d\n", idx);
    vector moves; initVector(&moves);
    kingMoves(&game, &board[7][4], &moves);
    if (moves.size != 0)
        for (int i = 0; i < moves.size; i++) {
            char* move = (char*)vectorGet(&moves, i);
            printf("%s\n", move);
        }

    for (int i = 0; i < PIECES; i++) {
        vectorFree(&black[i]);
        vectorFree(&white[i]);
    }

    return 0;
}
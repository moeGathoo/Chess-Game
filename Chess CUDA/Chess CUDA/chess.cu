#include "headers.cuh"

//initializes values of all global variables
char pieceTypes[6] = { 'r', 'n', 'b', 'q', 'k', 'p' };    //types of pieces on board
int pieceValues[6] = { 50, 30, 30, 90, 900, 10 };         //piece weightings for search and evaluation algorithms
cell board[RANK][RANK];
vector black[PIECES];
vector white[PIECES];

int main(void) {
    char* fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR";

    for (int i = 0; i < PIECES; i++) {
        initVector(&black[i]);
        initVector(&white[i]);
    }

    initBoard(fen);
    
    FILE* f = fopen("textFiles/fen.txt", "w");
    char newFen[128];
    for (int i = 0; i < RANK; i++) {
        int empty = 0;
        for (int j = 0; j < RANK; j++) {
            cell *space = &board[i][j];
            if (space->piece != '-') {
                if (empty != 0) fprintf(f, "%d", empty);
                fprintf(f, "%c", space->piece);
                empty = 0;
            }
            else empty++;
        }
        if (empty != 0) fprintf(f, "%d", empty);
        if (i != RANK - 1) fprintf(f, "%c", '/');
    }
    fclose(f);

    f = fopen("textFiles/fen.txt", "r");
    fgets(newFen, 128, f);
    fclose(f);

    for (int i = 0; i < PIECES; i++) {
        vectorFree(&black[i]);
        vectorFree(&white[i]);
    }
}
#include "../headers.cuh"

/**
 * @brief Initializes empty board, sets 'position' attribute for all spaces.
 *
 * @param fen Required to add pieces to empty board.
 *
 * @return Nothing. Empty board is initialized and pieces are added
 * to relevant spaces based on fen string.
 */
void initBoard(cell board[][RANK], char* fen) {
    for (int i = RANK - 1; i >= 0; i--) {
        for (int j = FILEA; j < FILEA + RANK; j++) {
            //convert rank and file to chars
            char file[2]; sprintf(file, "%c", j);
            char rank[2]; sprintf(rank, "%d", RANK - i);
            //get position string
            char* pos = (char*)malloc(2); if (pos == nullptr) { printf("Allocation error."); return; }
            strcpy(pos, file); strcat(pos, rank);
            board[i][j - FILEA].position = pos; //modify board position attribute
            board[i][j - FILEA].colour = '-';
            board[i][j - FILEA].hasPiece = false;
            board[i][j - FILEA].piece = '-';
        }
    }
    //addPieces(board, fen); //add pieces to board based on string
}

/**
 * @brief Prints the pieces of the board in relevant positions.
 *
 * @param None
 *
 * @return Nothing. Current board is printed to terminal.
 */
void printBoard(cell board[][RANK]) {
    for (int i = 0; i < RANK; i++) {
        for (int j = 0; j < RANK; j++) {
            printf("%c\t", board[i][j].piece);
        }
        printf("\n");
    }
}
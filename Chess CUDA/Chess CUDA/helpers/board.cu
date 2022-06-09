#include "../headers.cuh"

/**
 * @brief Fdds pieces to board based on fen string representation.
 *
 * @param fen: Fen string representation of current states board.
 *
 * @return Nothing. Pieces are added to the board.
 */
void addPieces(char* fen) {
    int row = 0, col = 0;
    for (int i = 0; i < strlen(fen); i++) {
        char c = fen[i];
        //end of rank (row), move to next rank and reset file counter
        if (c == '/') { row++; col = 0; continue; }
        //add piece to space on board, change relevant attributes of space
        if (!isdigit(c)) {
            board[row][col].hasPiece = true;
            board[row][col].piece = c;
            if (islower(c)) board[row][col].colour = 'b';
            else board[row][col].colour = 'w';
            col++;
        }
        else {      //skip spaces with no pieces
            int skip = atoi(&c);
            col += skip;
        }
    }
    getPositions();
}

/**
 * @brief Initializes empty board, sets 'position' attribute for all spaces.
 *
 * @param fen Required to add pieces to empty board.
 *
 * @return Nothing. Empty board is initialized and pieces are added
 * to relevant spaces based on fen string.
 */
void initBoard(char* fen) {
    for (int i = RANK - 1; i >= 0; i--) {
        for (int j = FILE; j < FILE + RANK; j++) {
            //convert rank and file to chars
            char file[2]; sprintf(file, "%c", j);
            char rank[2]; sprintf(rank, "%d", RANK - i);
            //get position string
            char* pos = (char*)malloc(2);
            strcpy(pos, file); strcat(pos, rank);
            board[i][j - FILE].position = pos; //modify board position attribute
        }
    }
    addPieces(fen); //add pieces to board based on string
}

/**
 * @brief Prints the pieces of the board in relevant positions.
 *
 * @param None
 *
 * @return Nothing. Current board is printed to terminal.
 */
void printBoard() {
    for (int i = 0; i < RANK; i++) {
        for (int j = 0; j < RANK; j++) {
            printf("%c\t", board[i][j].piece);
        }
        printf("\n");
    }
}
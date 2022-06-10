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
        for (int j = FILEA; j < FILEA + RANK; j++) {
            //convert rank and file to chars
            char file[2]; sprintf(file, "%c", j);
            char rank[2]; sprintf(rank, "%d", RANK - i);
            //get position string
            char* pos = (char*)malloc(2);
            strcpy(pos, file); strcat(pos, rank);
            board[i][j - FILEA].position = pos; //modify board position attribute
        }
    }
    addPieces(fen); //add pieces to board based on string
}

/**
 * @brief Clears all pieces off the board.
 *
 * @param None
 *
 * @return Nothing. All pieces are removed from the board.
 */
void resetBoard() {
    for (int i = 0; i < RANK; i++) {
        for (int j = 0; j < RANK; j++) {
            cell* space = &board[i][j]; //point to space on board
            //set space to have no piece and no colour
            space->hasPiece = false;
            space->piece = '-';
            space->colour = '-';
        }
    }
}

/**
 * @brief Checks if a space is a viable move option for a piece.
 *
 * @param piece Pointer to piece to move.
 * @param row Row of space in array not board.
 * @param col Column of space in array not board.
 * @param moves Pointer to a vector containing all the moves for piece
 *
 * @return Nothing. If the space is a viable option, it is added to the
 * vector the pointer 'moves' points to. Adds piece position plus space position
 * e.g. a8a8
 *
 * TODO: add flag for pawn movements to reduce to one function
 */
void checkSpace(cell* piece, int row, int col, vector* moves) {
    if (row >= 0 && row < RANK && col >= 0 && col < RANK) { //row and col must be within bounds of spaces on board
        cell space = board[row][col];
        char* move = "";
        strcpy(move, piece->position); strcat(move, space.position);
        if (!space.hasPiece) //if the space is empty, add it moves
            vectorPushBack(moves, move);
        else {
            if (piece->colour != space.colour) //if the piece is of the enmy colour, add to moves
                vectorPushBack(moves, move);
        }
    }
}

/**
 * @brief Since pawns cannot capture forward, check if space ahead of pawn is empty
 *
 * @param piece Pointer to piece to move.
 * @param row Row of space in array not board.
 * @param col Column of space in array not board.
 * @param moves Pointer to a vector containing all the moves for piece.
 *
 * @return Nothing. If the space is a viable option, it is added to the
 * vector the pointer 'moves' points to. Adds piece position plus space position
 * e.g. d2d3
 */
void checkSpaceP(cell* piece, int row, int col, vector* moves) {
    if (row >= 0 && row < RANK && col >= 0 && col < RANK) {
        cell space = board[row][col];
        char* move = "";
        strcpy(move, piece->position); strcat(move, space.position);
        if (piece->colour != space.colour)
            vectorPushBack(moves, move);
    }
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
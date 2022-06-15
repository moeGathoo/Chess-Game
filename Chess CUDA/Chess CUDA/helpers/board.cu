#include "../headers.cuh"

/**
 * @brief Fdds pieces to board based on fen string representation.
 *
 * @param fen: Fen string representation of current states board.
 *
 * @return Nothing. Pieces are added to the board.
 */
void addPieces(cell board[][RANK], char* fen) {
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
    getPositions(board);
}

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
    addPieces(board, fen); //add pieces to board based on string
}

/**
 * @brief Clears all pieces off the board.
 *
 * @param None
 *
 * @return Nothing. All pieces are removed from the board.
 */
void resetBoard(cell board[][RANK]) {
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
void checkSpace(cell board[][RANK], cell* piece, int row, int col, vector* moves) {
    if (row >= 0 && row < RANK && col >= 0 && col < RANK) { //row and col must be within bounds of spaces on board
        cell space = board[row][col];
        char *move = (char*)malloc(sizeof(piece->position) + sizeof(space.position));
        if (move != NULL) { strcpy(move, piece->position); strcat(move, space.position); }
        else { printf("Allocation error."); return; }
        if (!space.hasPiece) { //if the space is empty, add it moves
            vectorPushBack(moves, move);
        }
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
void checkSpaceP(cell board[][RANK], cell* piece, int row, int col, vector* moves) {
    if (row >= 0 && row < RANK && col >= 0 && col < RANK) {
        cell space = board[row][col];
        char* move = (char*)malloc(sizeof(piece->position) + sizeof(space.position));
        if (move == nullptr) { printf("Allocation error."); return; }
        strcpy(move, piece->position); strcat(move, space.position);
        if (piece->colour != space.colour)
            vectorPushBack(moves, move);
    }
}

/**
 * @brief Checks the viability of all spaces a king can move to.
 *
 * @param king Pointer to king to be moved.
 * @param moves Pointer to vector storing all moves available.
 *
* @return Nothing. If the space is a viable option, it is added to the
 * vector the pointer 'moves' points to. Adds piece position plus space position
 * e.g. d2d3
 */
void checkSpacesK(cell board[][RANK], state* game, cell* king, vector* moves) {
    //convert kings position to row and column for array position
    int coords[2] = { 0, 0 };
    if (king == NULL) {
        game->gameOver = true;
        return;
    }
    toCoords(king->position, coords);

    //checks all spaces in 3x3 neighbourhood of king
    checkSpace(board, king, coords[0] + 1, coords[1], moves);
    checkSpace(board, king, coords[0] - 1, coords[1], moves);
    checkSpace(board, king, coords[0], coords[1] + 1, moves);
    checkSpace(board, king, coords[0], coords[1] - 1, moves);
    checkSpace(board, king, coords[0] + 1, coords[1] + 1, moves);
    checkSpace(board, king, coords[0] - 1, coords[1] - 1, moves);
    checkSpace(board, king, coords[0] + 1, coords[1] - 1, moves);
    checkSpace(board, king, coords[0] - 1, coords[1] + 1, moves);

    //sort list of moves alphabetically
    //sort(moves->begin(), moves->end());
}

/**
 * @brief Checks for possible castling moves available to the king playing.
 *
 * @param state Curenst state of the board.
 * @param king King to be moves.
 * @param moves Pointer to vector storing all moves available.
 *
 * @return Nothing. If a castle move is available based on state, it will
 * be added to the list of moves.
 */
void castle(cell board[][RANK], state* state, cell* king, vector* moves) {
    if (state->castle == "-") return; //no castling moves available
    bool kSide = false, qSide = false; //initialize queen and king side castling flags to false
    
    char k = 'k', q = 'q';
    if (king->colour == 'w') { k = toupper(k); q = toupper(q); }
    
    char* posPtr = strchr(state->castle, k);
    if (posPtr != NULL) kSide = true;
    posPtr = strchr(state->castle, q);
    if (posPtr != NULL) qSide = true;

    //get king row and column
    int* coords = (int*)calloc(2, sizeof(int));
    if (coords == nullptr) { printf("Allocation error."); return; }
    toCoords(king->position, coords);
    if (kSide) {
        //if all spaces between king and rook is empty on kings side, add move
        if (!board[coords[0]][coords[1] + 1].hasPiece && !board[coords[0]][coords[1] + 2].hasPiece) {
            char* move = (char*)malloc(sizeof(king->position) + sizeof(board[coords[0]][coords[1] + 2].position));
            if (move == nullptr) { printf("Allocation error."); return; }
            strcpy(move, king->position); strcat(move, board[coords[0]][coords[1] + 2].position);
            vectorPushBack(moves, move);
        }
    }
    if (qSide) {
        //if all spaces between king and rook is empty on queens side, add move
        if (!board[coords[0]][coords[1] - 1].hasPiece && !board[coords[0]][coords[1] - 2].hasPiece && !board[coords[0]][coords[1] - 3].hasPiece) {
            char* move = (char*)malloc(sizeof(king->position) + sizeof(board[coords[0]][coords[1] - 2].position));
            if (move == nullptr) { printf("Allocation error."); return; }
            strcpy(move, king->position); strcat(move, board[coords[0]][coords[1] - 2].position);
            vectorPushBack(moves, move);
        }
    }
    free(coords);
}

/**
 * @brief Moves a piece from one space on the board to another.
 *
 * @param startSpace Pointer to space containing the piece to move.
 * @param goalSpace Pointer to space the piece is moving to.
 *
 * @return Nothing. The attributes of startSpace and goalSpace are updated
 * to reflect moving the piece.
 */
void movePiece(cell board[][RANK], cell* startSpace, cell* goalSpace) {
    goalSpace->piece = startSpace->piece; startSpace->piece = '-';
    goalSpace->colour = startSpace->colour; startSpace->colour = '-';
    goalSpace->hasPiece = true; startSpace->hasPiece = false;
    getPositions(board);
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
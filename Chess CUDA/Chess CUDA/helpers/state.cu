#include "../headers.cuh"

/**
 * @brief Initializes a state based on current baord based on fen information.
 *
 * @param fen Fen string representation of board.
 * @param side Side to play.
 * @param castle Castling rights for each side.
 * @param enPassant En passant move available.
 * @param halfMove Total moves since pawn move or piece capture.
 * @param fullMove Total number of moves incremented after black moves.
 *
 * @return state State object with current board information.
 */
state initState(char* fen, char side, char *castle, char* enPassant, int halfMove, int fullMove) {
    struct state curr;
    strcpy(curr.fen, fen);
    curr.side = side;
    strcpy(curr.castle, castle);
    curr.enPassant = enPassant;
    curr.halfMove = halfMove;
    curr.fullMove = fullMove;
    return curr;
}

/**
 * @brief Converts position of space to corresponding row and column in 2D array.
 *
 * @param pos String position to convert e.g. e5
 * @param coords Pointer to integer array created to store row and column.
 *
 * @return Nothing. The converted values are stored in the array,
 * e.g. e5 corresponds to row 3, column 4.
 */
void toCoords(char* pos, int* coords) {
    int row = RANK - (pos[1] - '0');    //convert rank number to int   
    int col = pos[0] - FILEA;            //convert file letter to int
    coords[0] = row; coords[1] = col;
}

/**
 * @brief Get the positions of all pieces on the board.
 *
 * @param None
 *
 * @return Nothing. The respective vectors storing the different piece
 * type's positions are updated with piece positions
 */
void getPositions() {
    vectorClear();

    for (int i = 0; i < RANK; i++) {
        for (int j = 0; j < RANK; j++) {
            if (board[i][j].piece != '-') { //if board space does not have piece
                //find index of piece in pieceTypes array to determine which vector to update
                bool found = false; int idx = -1;
                for (int p = 0; p < PIECES; p++) {
                    if (board[i][j].piece == pieceTypes[p]) {
                        found = true;
                        idx = p;
                        break;
                    }
                }
                if (found)
                    //lowercase piece not found, therefore white piece type
                    vectorPushBack(&black[idx], &board[i][j]);
                else {
                    for (int p = 0; p < PIECES; p++) {
                        if (board[i][j].piece == toupper(pieceTypes[p])) {
                            found = true;
                            idx = p;
                            break;
                        }
                    }
                    vectorPushBack(&white[idx], &board[i][j]);
                }
            }
        }
    }
}

/**
 * @brief Update the current state of the board, usually after moving a piece.
 *
 * @param state Pointer to state object to be updated
 * @param piece Pointer to space of piece that was moved.
 * @param flag Boolean indicating whether a pawn was moved or piece was captured.
 *
 * @return Nothing. State object is updated.
 */
void updateState(state* state, cell* piece, bool flag) {
    //create new fen string representing board by traversing in row-major order
    FILE* f = fopen("textFiles/fen.txt", "w");
    char newFen[STR_BUFFER];
    for (int i = 0; i < RANK; i++) {
        int empty = 0;
        for (int j = 0; j < RANK; j++) {
            cell* space = &board[i][j];
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
    strncpy((char*)state->fen, newFen, STR_BUFFER); //assign new fen string generated

    //castling piece types
    char k = 'k', q = 'q', r = 'r';
    if (piece->colour == 'w') { k = toupper(k); q = toupper(q); r = toupper(r); }
    //printf("%c %c %c\n", k, q, r);

    if (piece->piece == k) { //if king moves, remove all castling rights for that side
        char* posPtr = strchr(state->castle, k);
        if (posPtr != NULL) {
            int idxToDel = posPtr - state->castle;
            memmove(&state->castle[idxToDel], &state->castle[idxToDel+1], strlen(state->castle) - idxToDel);
        }
        posPtr = strchr(state->castle, q);
        if (posPtr != NULL) {
            int idxToDel = posPtr - state->castle;
            memmove(&state->castle[idxToDel], &state->castle[idxToDel + 1], strlen(state->castle) - idxToDel);
        }
    }
    else if (piece->piece == r) { //rook moves
        //find which side rook is and remove relevant castling rights flag
        int* coords = (int*)calloc(2,sizeof(int));
        toCoords(piece->position, coords);
        if (coords[1] > 5) {
            char* posPtr = strchr(state->castle, k);
            if (posPtr != NULL) {
                int idxToDel = posPtr - state->castle;
                memmove(&state->castle[idxToDel], &state->castle[idxToDel + 1], strlen(state->castle) - idxToDel);
            }
        }
        else {
            char* posPtr = strchr(state->castle, q);
            if (posPtr != NULL) {
                int idxToDel = posPtr - state->castle;
                memmove(&state->castle[idxToDel], &state->castle[idxToDel + 1], strlen(state->castle) - idxToDel);
            }
        }
        free(coords);
    }
    if (strlen(state->castle) == 0) strcpy(state->castle, "-");
    //printf("castle updated");

    if (state->side == 'w') state->side = 'b'; //switch side to play
    else {
        state->side = 'w';
        state->fullMove++; //if black played, increment full moves
    }

    if (flag) state->halfMove = 0; //if pawn moved or piece captured, reset half moves
    else state->halfMove++; //else increment

    //if at least one king is missing, game is over
    if (white[4].size == 0 || black[4].size == 0) state->gameOver = true;

}

/**
 * @brief Prints out attributes of current state of board.
 *
 * @param currState Pointer to state object with current state information.
 *
 * @return Nothing. Current state information is printed to terminal.
 */
void printState(state* currState) {
    printf("%s ", currState->fen);
    printf("%c ", currState->side);
    printf("%s ", currState->castle);
    printf("%s ", currState->enPassant);
    printf("%d ", currState->halfMove);
    printf("%d\n", currState->fullMove);
}

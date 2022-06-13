#include "../headers.cuh"

/**
 * @brief Generates all possible moves to play for given rook.
 *
 * @param rook Pointer to space on board containing rook to move.
 * @param moves Pointer to vector of strings stroing all moves for said piece.
 *
 * @return Nothing. All moves deemed playable for rook is sent to 'moves' vector
 */
void rookMoves(cell* rook, vector* moves) {
    //determine row and column of rook appearing in 2D array
    int coords[2] = { 0, 0 };
    toCoords(rook->position, coords);

    //since the rook is a sliding piece, find how many spaces piece can slide to before running into another piece, check all those spaces
    //rook moveset is all spaces vertical and horizontal
    //determine number of spaces to slide to vertically down
    int i = 1;
    while (coords[0] + i < RANK && !board[coords[0] + i][coords[1]].hasPiece) i++;
    for (int j = 1; j <= i; j++) checkSpace(rook, coords[0] + j, coords[1], moves);
    //determine number of spaces to slide to vertically up
    i = 1;
    while (coords[0] - i >= 0 && !board[coords[0] - i][coords[1]].hasPiece) i++;
    for (int j = 1; j <= i; j++) checkSpace(rook, coords[0] - j, coords[1], moves);
    //determine number of spaces to slide to horizontally right
    i = 1;
    while (coords[1] + i < RANK && !board[coords[0]][coords[1] + i].hasPiece) i++;
    for (int j = 1; j <= i; j++) checkSpace(rook, coords[0], coords[1] + j, moves);
    //determine number of spaces to slide to horizontally left
    i = 1;
    while (coords[1] - i >= 0 && !board[coords[0]][coords[1] - i].hasPiece) i++;
    for (int j = 1; j <= i; j++) checkSpace(rook, coords[0], coords[1] - j, moves);

    //sort moves alphabetically
    vectorSort(moves);
}

/**
 * @brief Generates all possible moves to play for given knight.
 *
 * @param knight Pointer to space on board containing knight to move.
 * @param moves Pointer to vector of strings stroing all moves for said piece.
 *
 * @return Nothing. All moves deemed playable for knight is sent to 'moves' vector
 */
void knightMoves(cell* knight, vector* moves) {
    //determine row and column of rook appearing in 2D array
    int coords[2] = { 0, 0 };
    toCoords(knight->position, coords);

    //since knights aren't sliding pieces and can hop over other pieces,
    //individual spaces can be checked
    //check all spaces a knight can move to
    checkSpace(knight, coords[0] - 1, coords[1] - 2, moves);
    checkSpace(knight, coords[0] - 2, coords[1] - 1, moves);
    checkSpace(knight, coords[0] - 2, coords[1] + 1, moves);
    checkSpace(knight, coords[0] - 1, coords[1] + 2, moves);
    checkSpace(knight, coords[0] + 1, coords[1] + 2, moves);
    checkSpace(knight, coords[0] + 2, coords[1] + 1, moves);
    checkSpace(knight, coords[0] + 1, coords[1] - 2, moves);
    checkSpace(knight, coords[0] + 2, coords[1] - 1, moves);

    //sort moves alphabetically
    vectorSort(moves);
}

/**
 * @brief Generates all possible moves to play for given bishop.
 *
 * @param bishop Pointer to space on board containing bishop to move.
 * @param moves Pointer to vector of strings stroing all moves for said piece.
 *
 * @return Nothing. All moves deemed playable for bishop is sent to 'moves' vector
 */
void bishopMoves(cell* bishop, vector* moves) {
    //determine row and column of rook appearing in 2D array
    int coords[2] = { 0, 0 };
    toCoords(bishop->position, coords);

    //since the bishop is a sliding piece, find how many spaces piece can slide to before running into another piece, check all those spaces
    //bishop moveset is all diagonal spaces
    //determine number of spaces to slide to down and right
    int i = 1;
    while (coords[0] + i < RANK && coords[1] + i < RANK && !board[coords[0] + i][coords[1] + i].hasPiece) i++;
    for (int j = 1; j <= i; j++) checkSpace(bishop, coords[0] + j, coords[1] + j, moves);
    //determine number of spaces to slide to down and left
    i = 1;
    while (coords[0] + i<RANK && coords[1] - i>-1 && !board[coords[0] + i][coords[1] - i].hasPiece) i++;
    for (int j = 1; j <= i; j++) checkSpace(bishop, coords[0] + j, coords[1] - j, moves);
    //determine number of spaces to slide to up and right
    i = 1;
    while (coords[0] - i > -1 && coords[1] + i < RANK && !board[coords[0] - i][coords[1] + i].hasPiece) i++;
    for (int j = 1; j <= i; j++) checkSpace(bishop, coords[0] - j, coords[1] + j, moves);
    //determine number of spaces to slide to up and left
    i = 1;
    while (coords[0] - i > -1 && coords[1] - i > -1 && !board[coords[0] - i][coords[1] - i].hasPiece) i++;
    for (int j = 1; j <= i; j++) checkSpace(bishop, coords[0] - j, coords[1] - j, moves);

    //sort moves alphabetically
    vectorSort(moves);
}

/**
 * @brief Generates all possible moves to play for given queen.
 *
 * @param queen Pointer to space on board containing queen to move.
 * @param moves Pointer to vector of strings stroing all moves for said piece.
 *
 * @return Nothing. All moves deemed playable for queen is sent to 'moves' vector
 */
void queenMoves(cell* queen, vector* moves) {
    //since queen moveset is combined bishop and rook moveset and a sliding piece, call those functions
    bishopMoves(queen, moves);
    rookMoves(queen, moves);
    //sort moves alphabetically
    vectorSort(moves);
}

/**
 * @brief Generates all possible moves to play for given king.
 *
 * @param king Pointer to space on board containing king to move.
 * @param moves Pointer to vector of strings stroing all moves for said piece.
 *
 * @return Nothing. All moves deemed playable for king is sent to 'moves' vector
 */
void kingMoves(state* state, cell* king, vector* moves) {
    //check all spaces for king moveset (all spaces in 3x3 neighbourhood)
    vector kingMoves; initVector(&kingMoves);
    checkSpacesK(king, &kingMoves);
    //check for possible castling moves
    castle(state, king, &kingMoves);

    vector* enemy; vector enemyMoves; initVector(&enemyMoves);
    if (king->colour == 'b') enemy = white;
    else enemy = black;
    //since a king cannot move itself into check, make sure none of the spaces a king can move to will do that
    //generate list of all spaces enemy side can move to
    for (int i = 0; i < PIECES; i++) {
        if (i == 4) checkSpacesK((cell*)vectorGet(&enemy[4],0), &enemyMoves);
        else if (enemy[i].size != 0) {
            for (int j = 0; j < enemy[i].size; j++) {
                cell* piece = (cell*)vectorGet(&enemy[i], j);
                pieceMoves(state, piece, i, &enemyMoves);
            }
        }
    }
    vectorSort(&enemyMoves);

    //remove all the moves a king can move to that an enemy piece can move to as well
    for (int i = 0; i < kingMoves.size; i++) {
        char* move = (char*)vectorGet(&kingMoves, i);
        char goal[3] = ""; strncpy(goal, &move[2], 2); //extract space to move to
        for (int j = 0; j < enemyMoves.size; j++) { //search for space amongst enemy moves
            char* enemyMove = (char*)vectorGet(&enemyMoves, j);
            char enemy[3] = ""; strncpy(enemy, &enemyMove[2], 2);
            if (!strcmp(goal, enemy)) { //if space is found
                vectorRemove(&kingMoves, i); //remove it from 'moves' vector
                i--;
                break; //break out of enemy search since space has been found, no need to search further
            }
        }
    }

    //add remaining legal moves
    if (kingMoves.size != 0) {
        for (int i = 0; i < kingMoves.size; i++)
            vectorPushBack(moves, (char*)vectorGet(&kingMoves, i));
    }

    //sort moves alphabetically
    vectorSort(moves);
}

/**
 * @brief Generates all possible moves to play for given pawn.
 *
 * @param pawn Pointer to space on board containing pawn to move.
 * @param moves Pointer to vector of strings stroing all moves for said piece.
 *
 * @return Nothing. All moves deemed playable for pawn is sent to 'moves' vector
 */
void pawnMoves(cell* pawn, vector* moves) {
    //determine row and column of pawn appearing in 2D array
    int coords[2] = { 0, 0 };
    toCoords(pawn->position, coords);

    if (pawn->colour == 'b') { //search row in front of black pawn
        if (coords[0] + 1 < RANK)
            if (!board[coords[0] + 1][coords[1]].hasPiece) {
                char* move = (char*)malloc(sizeof(pawn->position) + sizeof(board[coords[0] + 1][coords[1]].position));
                strcpy(move, pawn->position); strcat(move, board[coords[0] + 1][coords[1]].position);
                vectorPushBack(moves, move);
            }
        if (coords[0] == 1 && !board[coords[0] + 1][coords[1]].hasPiece) //check for possible 2 space move if pawn is in starting rank
            if (!board[coords[0] + 2][coords[1]].hasPiece) {
                char* move = (char*)malloc(sizeof(pawn->position) + sizeof(board[coords[0] + 2][coords[1]].position));
                strcpy(move, pawn->position); strcat(move, board[coords[0] + 2][coords[1]].position);
                vectorPushBack(moves, move);
            }
        //check for diagonal captures for pawn
        if (board[coords[0] + 1][coords[1] - 1].hasPiece) checkSpaceP(pawn, coords[0] + 1, coords[1] - 1, moves);
        if (board[coords[0] + 1][coords[1] + 1].hasPiece) checkSpaceP(pawn, coords[0] + 1, coords[1] + 1, moves);
    }
    else { //search row in front of white pawn
        if (coords[0] + 1 >= 0)
            if (!board[coords[0] - 1][coords[1]].hasPiece) {
                char* move = (char*)malloc(sizeof(pawn->position) + sizeof(board[coords[0] - 1][coords[1]].position));
                strcpy(move, pawn->position); strcat(move, board[coords[0] - 1][coords[1]].position);
                vectorPushBack(moves, move);
            }
        if (coords[0] == 6 && !board[coords[0] - 1][coords[1]].hasPiece) //check for possible 2 space move if pawn is in starting rank
            if (!board[coords[0] - 2][coords[1]].hasPiece) {
                char* move = (char*)malloc(sizeof(pawn->position) + sizeof(board[coords[0] - 2][coords[1]].position));
                strcpy(move, pawn->position); strcat(move, board[coords[0] - 2][coords[1]].position);
                vectorPushBack(moves, move);
            }
        //check for diagonal captures for pawn
        if (board[coords[0] - 1][coords[1] - 1].hasPiece) checkSpaceP(pawn, coords[0] - 1, coords[1] - 1, moves);
        if (board[coords[0] - 1][coords[1] + 1].hasPiece) checkSpaceP(pawn, coords[0] - 1, coords[1] + 1, moves);
    }

    //sort moves alphabetically
    vectorSort(moves);
}

/**
 * @brief Generate moves for a piece given index.
 *
 * @param state Pointer to state object containing current board's information.
 * @param piece Pointer to cell containing piece to move.
 * @param index Index of piece type to determine which piece function to call.
 * @param moves Pointer to vector containing moves.
 *
 * @return Nothing. The moves for the piece specified that are generated are store in 'moves' vector.
 */
void pieceMoves(state* state, cell* piece, int index, vector* moves) {
    //swicth case assignment based on piece index and respective function for said piece's move generation
    switch (index) {
    case 0:
        rookMoves(piece, moves);
        break;
    case 1:
        knightMoves(piece, moves);
        break;
    case 2:
        bishopMoves(piece, moves);
        break;
    case 3:
        queenMoves(piece, moves);
        break;
    case 4:
        kingMoves(state, piece, moves);
        break;
    case 5:
        pawnMoves(piece, moves);
        break;
    }
}

/**
 * @brief Determines if the move specified is a legal move to make.
 *
 * @param state Pointer to state object containing current board's information.
 * @param move Pointer to string containing move to check legality.
 *
 * @return Boolean indicating legailty of move:
 * true if move is legal (move will be made),
 * false if move is illegal (error message will be printed)
 */
bool checkMove(state* state, char* move) {
    //get coordinates of start space and goal space in 2D array
    char start[3] = ""; strncpy(start, &move[0], 2);
    int* startCoords = (int*)calloc(2,sizeof(int)); toCoords(start, startCoords);
    char goal[3] = ""; strncpy(goal, &move[2], 2);
    int* goalCoords = (int*)calloc(2,sizeof(int)); toCoords(goal, goalCoords);

    //get pointers to said spaces on board
    cell* startSpace = &board[startCoords[0]][startCoords[1]];
    cell* goalSpace = &board[goalCoords[0]][goalCoords[1]];

    //flag to indicate if pawn was moved or piece was captured
    //used for half move update
    bool flag = false;
    if (startSpace->piece == 'p' || startSpace->piece == 'P' || goalSpace->hasPiece) flag = true;

    int idx = 0;
    vector moves; initVector(&moves);
    if (startSpace->colour == 'b') { //piece is black
        char* posPtr = strchr(pieceTypes, startSpace->piece);
        if (posPtr != NULL) idx = posPtr - pieceTypes;
        else return false;
        pieceMoves(state, startSpace, idx, &moves); //generate all moves for respective piece
    }
    else { //piece is white, convert to lowercase for search
        char* posPtr = strchr(pieceTypes, tolower(startSpace->piece));
        if (posPtr != NULL) idx = posPtr - pieceTypes;
        else return false;
        pieceMoves(state, startSpace, idx, &moves);
    }

    //search for move in generated moves list
    if (vectorSearch(&moves, move)) { //move was found
        vectorFree(&moves);
        movePiece(startSpace, goalSpace); //move piece
        if (idx == 4) { //if king was moved
            //check if castkling moved was made on queen side
            if (startCoords[1] - goalCoords[1] == 2) {
                //move rook accordingly
                cell* oldRook = &board[goalCoords[0]][0];
                cell* newRook = &board[goalCoords[0]][goalCoords[1] + 1];
                movePiece(oldRook, newRook);
            } //check if castkling moved was made on king side
            else if (startCoords[1] - goalCoords[1] == -2) {
                //move rook accordingly
                cell* oldRook = &board[goalCoords[0]][RANK - 1];
                cell* newRook = &board[goalCoords[0]][goalCoords[1] - 1];
                movePiece(oldRook, newRook);
            }
        }
        updateState(state, goalSpace, flag); //update state to indicate move played
        free(startCoords); free(goalCoords);
        return true;
    }
    //move was not found (illegal move), return false
    free(startCoords); free(goalCoords); vectorFree(&moves);
    return false;
}

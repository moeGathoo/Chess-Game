#include "headers.cuh"

__device__ __constant__ char pieceType[PIECES];
__device__ __constant__ int pieceValue[PIECES];

//initializes values of all global variables
char pieceTypes[6] = { 'r', 'n', 'b', 'q', 'k', 'p' };    //types of pieces on board
int pieceValues[6] = { 50, 30, 30, 90, 900, 10 };         //piece weightings for search and evaluation algorithms

__device__ __host__ unsigned int strLen(char* p) {
    unsigned int count = 0;
    while (*p != '\0') {
        count++;
        p++;
    }
    return count;
}

__device__ __host__ char* strCpy(char* destination, const char* source) {
    // return if no memory is allocated to the destination
    if (destination == NULL) {
        return NULL;
    }

    // take a pointer pointing to the beginning of the destination string
    char* ptr = destination;

    // copy the C-string pointed by source into the array
    // pointed by destination
    while (*source != '\0')
    {
        *destination = *source;
        destination++;
        source++;
    }

    // include the terminating null character
    *destination = '\0';

    // the destination is returned by standard `strcpy()`
    return ptr;
}

__device__ __host__ char* strCat(char* dest, const char* src) {
    char* ptr = dest + strLen(dest);
    while (*src != '\0')
        *ptr++ = *src++;
    *ptr = '\0';
    return dest;
}

__device__ __host__ char* strChr(char* str, char c) {
    while (*str != c && *str != '\0')
        str++;
    if (*str == c) return str;
    else return NULL;
}

__device__ __host__ int strCmp(const char* str1, const char* str2) {
    while (*str1) {
        if (*str1 != *str2) break;
        str1++; str2++;
    }
    return *(const unsigned char*)str1 - *(const unsigned char*)str2;
}

__device__ __host__ char* strNcpy(char* dest, char* src, size_t num) {
    if (dest == NULL) return NULL;
    char* ptr = dest;
    while (*src && num--) {
        *dest = *src;
        dest++; src++;
    }
    *dest = '\0';
    return ptr;
}

__device__ __host__ int charToInt(char* p) {
    int k = 0;
    while (*p) {
        k = (k << 3) + (k << 1) + (*p) - '0';
        p++;
    }
    return k;
    /*int res = 0;
    for (int i = 0; p[i] != '\0'; ++i) {
        res = res * 10 + p[i] - '0';
    }
    return res;*/
}

__device__ __host__ int isInt(char c) {
    if (c >= '0' && c <= '9') return 1;
    return 0;
}

__device__ __host__ int isLower(char c) {
    if (c >= 97 && c <= 122) return 1;
    return 0;
}

__device__ __host__ int isUpper(char c) {
    if (c >= 65 && c <= 90) return 1;
    return 0;
}

__device__ __host__ int toUpper(char c) {
    if (c >= 'a' && c <= 'z') return c - 32;
    return c;
}

__device__ __host__ int toLower(char c) {
    if (c >= 'A' && c <= 'Z') return c + 32;
    return c;
}

__device__ __host__ void memCpy(void* dest, void* src, size_t n)
{
    // Typecast src and dest addresses to (char *)
    char* csrc = (char*)src;
    char* cdest = (char*)dest;

    // Copy contents of src[] to dest[]
    for (int i = 0; i < n; i++)
        cdest[i] = csrc[i];
}

__device__ __host__ void* reAlloc(void* ptr, size_t currSize, size_t newSize) {
    if (newSize == 0) {
        free(ptr);
        return NULL;
    }
    else if (!ptr) {
        return malloc(newSize);
    }
    else if (newSize <= currSize) {
        return ptr;
    }
    else {
        //assert((ptr) && (newLength > originalLength));
        void* ptrNew = malloc(newSize);
        if (ptrNew) {
            memCpy(ptrNew, ptr, currSize);
            free(ptr);
        }
        return ptrNew;
    }
}

__device__ __host__ void toCoords(char* pos, int* coords) {
    int row = RANK - (pos[1] - '0');    //convert rank number to int   
    int col = pos[0] - FILEA;            //convert file letter to int
    coords[0] = row; coords[1] = col;
}

/*==================================================================================================================================================*/

__device__ __host__ void initVector(vector* v) {
    v->capacity = VECTOR_INIT_CAPACITY;
    v->size = 0;
    v->items = (void**)malloc(sizeof(void*) * v->capacity);
}

__device__ __host__ static void vectorResize(vector* v, int capacity) {
    void** items = (void**)reAlloc(v->items, sizeof(void**) * v->capacity, sizeof(void**) * capacity);
    if (items) {
        v->items = items;
        v->capacity = capacity;
    }
}

__device__ __host__ void vectorPushBack(vector* v, void* item) {
    if (v->capacity == v->size)
        vectorResize(v, v->capacity * 2);
    v->items[v->size++] = item;
}

__device__ __host__ void* vectorGet(vector* v, int index) {
    if (index >= 0 && index < v->size)
        return v->items[index];
    return NULL;
}

__device__ __host__ void vectorRemove(vector* v, int index) {
    if (index < 0 || index >= v->size)
        return;

    v->items[index] = NULL;
    for (int i = index; i < v->size; i++) {
        v->items[i] = v->items[i + 1];
    }
    v->items[v->size] = NULL;
    v->size--;
    if (v->size > 0 && v->size == v->capacity / 4)
        vectorResize(v, v->capacity / 2);
}

__device__ __host__ int vectorSearch(vector* v, char* str) {
    int len = (sizeof(void*) * v->size) / sizeof(v->items[0]);
    for (int i = 0; i < len; i++)
        if (!strCmp((const char*)v->items[i], str))
            return 1;
    return 0;
}

__device__ __host__ void vectorFree(vector* v, int malloced) {
    if (malloced && v->size != 0)
        for (int i = 0; i < v->size; i++) {
            free(v->items[i]);
        }
    free(v->items);
}

/*==================================================================================================================================================*/

/**
 * @brief Fdds pieces to board based on fen string representation.
 *
 * @param fen: Fen string representation of current states board.
 *
 * @return Nothing. Pieces are added to the board.
 */
__device__ __host__ void addPieces(cell board[][BOARD_WIDTH], char* fen) {
    int row = 0, col = 0;
    for (int i = 0; i < (int)strLen(fen); i++) {
        char c = fen[i];
        //end of rank (row), move to next rank and reset file counter
        if (c == '/') { row++; col = 0; continue; }
        //add piece to space on board, change relevant attributes of space
        if (!isInt(c)) {
            board[row][col].hasPiece = true;
            board[row][col].piece = c;
            if (isLower(c)) board[row][col].colour = 'b';
            else board[row][col].colour = 'w';
            col++;
        }
        else {      //skip spaces with no pieces
            int skip = charToInt(&c);
            col += skip;
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
__device__ __host__ void checkSpace(cell board[][RANK], cell* piece, int row, int col, vector* moves) {
    if (row >= 0 && row < RANK && col >= 0 && col < RANK) { //row and col must be within bounds of spaces on board
        cell space = board[row][col];
        char* move = (char*)malloc(sizeof(piece->position) + sizeof(space.position));
        if (move != NULL) { strCpy(move, piece->position); strCat(move, space.position); }
        else { return; }
        if (!space.hasPiece) { //if the space is empty, add it moves
            //printf("%s\n", move);
            vectorPushBack(moves, move);
        }
        else {
            if (piece->colour != space.colour) { //if the piece is of the enmy colour, add to moves
                //printf("%s\n", move);
                vectorPushBack(moves, move);
            }
        }
        //free(move);
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
__device__ __host__ void checkSpaceP(cell board[][RANK], cell* piece, int row, int col, vector* moves) {
    if (row >= 0 && row < RANK && col >= 0 && col < RANK) {
        cell space = board[row][col];
        char* move = (char*)malloc(sizeof(piece->position) + sizeof(space.position));
        if (move == nullptr) { return; }
        strCpy(move, piece->position); strCat(move, space.position);
        if (piece->colour != space.colour)
            vectorPushBack(moves, move);
        //free(move);
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
__device__ __host__ void checkSpacesK(cell board[][RANK], state* game, cell* king, vector* moves) {
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
__device__ __host__ void castle(cell board[][RANK], state* state, cell* king, vector* moves) {
    if (state->castle == "-") return; //no castling moves available
    bool kSide = false, qSide = false; //initialize queen and king side castling flags to false

    char k = 'k', q = 'q';
    if (king->colour == 'w') { k = toUpper(k); q = toUpper(q); }

    char* posPtr = strChr(state->castle, k);
    if (posPtr != NULL) kSide = true;
    posPtr = strChr(state->castle, q);
    if (posPtr != NULL) qSide = true;

    //get king row and column
    int coords[2] = {0, 0};
    toCoords(king->position, coords);
    if (kSide) {
        //if all spaces between king and rook is empty on kings side, add move
        if (!board[coords[0]][coords[1] + 1].hasPiece && !board[coords[0]][coords[1] + 2].hasPiece) {
            char* move = (char*)malloc(sizeof(king->position) + sizeof(board[coords[0]][coords[1] + 2].position));
            if (move == nullptr) { return; }
            strCpy(move, king->position); strCat(move, board[coords[0]][coords[1] + 2].position);
            vectorPushBack(moves, move);
            //free(move);
        }
    }
    if (qSide) {
        //if all spaces between king and rook is empty on queens side, add move
        if (!board[coords[0]][coords[1] - 1].hasPiece && !board[coords[0]][coords[1] - 2].hasPiece && !board[coords[0]][coords[1] - 3].hasPiece) {
            char* move = (char*)malloc(sizeof(king->position) + sizeof(board[coords[0]][coords[1] - 2].position));
            if (move == nullptr) { return; }
            strCpy(move, king->position); strCat(move, board[coords[0]][coords[1] - 2].position);
            vectorPushBack(moves, move);
            //free(move);
        }
    }
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
__device__ __host__ void movePiece(cell board[][RANK], cell* startSpace, cell* goalSpace) {
    goalSpace->piece = startSpace->piece; startSpace->piece = '-';
    goalSpace->colour = startSpace->colour; startSpace->colour = '-';
    goalSpace->hasPiece = true; startSpace->hasPiece = false;
}

/*==================================================================================================================================================*/

/**
 * @brief Generates all possible moves to play for given rook.
 *
 * @param rook Pointer to space on board containing rook to move.
 * @param moves Pointer to vector of strings stroing all moves for said piece.
 *
 * @return Nothing. All moves deemed playable for rook is sent to 'moves' vector
 */
__device__ __host__ void rookMoves(cell board[][RANK], cell* rook, vector* moves) {
    //determine row and column of rook appearing in 2D array
    int coords[2] = { 0, 0 };
    toCoords(rook->position, coords);

    //since the rook is a sliding piece, find how many spaces piece can slide to before running into another piece, check all those spaces
    //rook moveset is all spaces vertical and horizontal
    //determine number of spaces to slide to vertically down
    int i = 1;
    while (coords[0] + i < RANK && !board[coords[0] + i][coords[1]].hasPiece)
        i++;
    for (int j = 1; j <= i; j++)
        checkSpace(board, rook, coords[0] + j, coords[1], moves);
    //determine number of spaces to slide to vertically up
    i = 1;
    while (coords[0] - i >= 0 && !board[coords[0] - i][coords[1]].hasPiece)
        i++;
    for (int j = 1; j <= i; j++)
        checkSpace(board, rook, coords[0] - j, coords[1], moves);
    //determine number of spaces to slide to horizontally right
    i = 1;
    while (coords[1] + i < RANK && !board[coords[0]][coords[1] + i].hasPiece)
        i++;
    for (int j = 1; j <= i; j++)
        checkSpace(board, rook, coords[0], coords[1] + j, moves);
    //determine number of spaces to slide to horizontally left
    i = 1;
    while (coords[1] - i >= 0 && !board[coords[0]][coords[1] - i].hasPiece)
        i++;
    for (int j = 1; j <= i; j++)
        checkSpace(board, rook, coords[0], coords[1] - j, moves);

    //sort moves alphabetically
    //vectorSort(moves);
}

/**
 * @brief Generates all possible moves to play for given knight.
 *
 * @param knight Pointer to space on board containing knight to move.
 * @param moves Pointer to vector of strings stroing all moves for said piece.
 *
 * @return Nothing. All moves deemed playable for knight is sent to 'moves' vector
 */
__device__ __host__ void knightMoves(cell board[][RANK], cell* knight, vector* moves) {
    //determine row and column of rook appearing in 2D array
    int coords[2] = { 0, 0 };
    toCoords(knight->position, coords);

    //since knights aren't sliding pieces and can hop over other pieces,
    //individual spaces can be checked
    //check all spaces a knight can move to
    checkSpace(board, knight, coords[0] - 1, coords[1] - 2, moves);
    checkSpace(board, knight, coords[0] - 2, coords[1] - 1, moves);
    checkSpace(board, knight, coords[0] - 2, coords[1] + 1, moves);
    checkSpace(board, knight, coords[0] - 1, coords[1] + 2, moves);
    checkSpace(board, knight, coords[0] + 1, coords[1] + 2, moves);
    checkSpace(board, knight, coords[0] + 2, coords[1] + 1, moves);
    checkSpace(board, knight, coords[0] + 1, coords[1] - 2, moves);
    checkSpace(board, knight, coords[0] + 2, coords[1] - 1, moves);

    //sort moves alphabetically
}

/**
 * @brief Generates all possible moves to play for given bishop.
 *
 * @param bishop Pointer to space on board containing bishop to move.
 * @param moves Pointer to vector of strings stroing all moves for said piece.
 *
 * @return Nothing. All moves deemed playable for bishop is sent to 'moves' vector
 */
__device__ __host__ void bishopMoves(cell board[][RANK], cell* bishop, vector* moves) {
    //determine row and column of rook appearing in 2D array
    int coords[2] = { 0, 0 };
    toCoords(bishop->position, coords);

    //since the bishop is a sliding piece, find how many spaces piece can slide to before running into another piece, check all those spaces
    //bishop moveset is all diagonal spaces
    //determine number of spaces to slide to down and right
    int i = 1;
    while (coords[0] + i < RANK && coords[1] + i < RANK && !board[coords[0] + i][coords[1] + i].hasPiece) i++;
    for (int j = 1; j <= i; j++) checkSpace(board, bishop, coords[0] + j, coords[1] + j, moves);
    //determine number of spaces to slide to down and left
    i = 1;
    while (coords[0] + i<RANK && coords[1] - i>-1 && !board[coords[0] + i][coords[1] - i].hasPiece) i++;
    for (int j = 1; j <= i; j++) checkSpace(board, bishop, coords[0] + j, coords[1] - j, moves);
    //determine number of spaces to slide to up and right
    i = 1;
    while (coords[0] - i > -1 && coords[1] + i < RANK && !board[coords[0] - i][coords[1] + i].hasPiece) i++;
    for (int j = 1; j <= i; j++) checkSpace(board, bishop, coords[0] - j, coords[1] + j, moves);
    //determine number of spaces to slide to up and left
    i = 1;
    while (coords[0] - i > -1 && coords[1] - i > -1 && !board[coords[0] - i][coords[1] - i].hasPiece) i++;
    for (int j = 1; j <= i; j++) checkSpace(board, bishop, coords[0] - j, coords[1] - j, moves);

    //sort moves alphabetically
}

/**
 * @brief Generates all possible moves to play for given queen.
 *
 * @param queen Pointer to space on board containing queen to move.
 * @param moves Pointer to vector of strings stroing all moves for said piece.
 *
 * @return Nothing. All moves deemed playable for queen is sent to 'moves' vector
 */
__device__ __host__ void queenMoves(cell board[][RANK], cell* queen, vector* moves) {
    //since queen moveset is combined bishop and rook moveset and a sliding piece, call those functions
    bishopMoves(board, queen, moves);
    rookMoves(board, queen, moves);
    //sort moves alphabetically
}

/**
 * @brief Generates all possible moves to play for given king.
 *
 * @param king Pointer to space on board containing king to move.
 * @param moves Pointer to vector of strings stroing all moves for said piece.
 *
 * @return Nothing. All moves deemed playable for king is sent to 'moves' vector
 */
__device__ __host__ void kingMoves(cell board[][RANK], state* state, cell* king, vector* moves) {
    //check all spaces for king moveset (all spaces in 3x3 neighbourhood)
    vector kingMoves; initVector(&kingMoves);
    checkSpacesK(board, state, king, &kingMoves);
    //check for possible castling moves
    castle(board, state, king, &kingMoves);

    vector enemyMoves; initVector(&enemyMoves);
    for (int i = 0; i < BOARD_WIDTH; i++) {
        for (int j = 0; j < BOARD_WIDTH; j++) {
            cell piece = board[i][j];
            if (piece.hasPiece && piece.colour != state->side) {
                char c = piece.piece; if (state->side == 'w') c = toLower(c);
                char* pos = strChr(pieceType, c);
                int idx = -1;
                if (pos != NULL) idx = pos - pieceType;
                if (idx == 4) checkSpacesK(board, state, &piece, &enemyMoves);
                pieceMoves(board, state, &piece, idx, &enemyMoves);
            }
        }
    }

    //remove all the moves a king can move to that an enemy piece can move to as well
    for (int i = 0; i < kingMoves.size; i++) {
        char* move = (char*)vectorGet(&kingMoves, i);
        char goal[3] = ""; strNcpy(goal, &move[2], 2); //extract space to move to
        for (int j = 0; j < enemyMoves.size; j++) { //search for space amongst enemy moves
            char* enemyMove = (char*)vectorGet(&enemyMoves, j);
            char enemy[3] = ""; strNcpy(enemy, &enemyMove[2], 2);
            if (!strCmp(goal, enemy)) { //if space is found
                free(kingMoves.items[i]);
                vectorRemove(&kingMoves, i); //remove it from 'moves' vector
                i--;
                break; //break out of enemy search since space has been found, no need to search further
            }
        }
    }
    vectorFree(&enemyMoves, 1);

    //add remaining legal moves
    if (kingMoves.size != 0) {
        for (int i = 0; i < kingMoves.size; i++)
            vectorPushBack(moves, (char*)vectorGet(&kingMoves, i));
    }
    //sort moves alphabetically
}

/**
 * @brief Generates all possible moves to play for given pawn.
 *
 * @param pawn Pointer to space on board containing pawn to move.
 * @param moves Pointer to vector of strings stroing all moves for said piece.
 *
 * @return Nothing. All moves deemed playable for pawn is sent to 'moves' vector
 */
__device__ __host__ void pawnMoves(cell board[][RANK], cell* pawn, vector* moves) {
    //determine row and column of pawn appearing in 2D array
    int coords[2] = { 0, 0 };
    toCoords(pawn->position, coords);

    if (pawn->colour == 'b') { //search row in front of black pawn
        if (coords[0] + 1 < RANK)
            if (!board[coords[0] + 1][coords[1]].hasPiece) {
                char* move = (char*)malloc(sizeof(pawn->position) + sizeof(board[coords[0] + 1][coords[1]].position));
                if (move == nullptr) { return; }
                strCpy(move, pawn->position); strCat(move, board[coords[0] + 1][coords[1]].position);
                vectorPushBack(moves, move);
                //free(move);
            }
        if (coords[0] == 1 && !board[coords[0] + 1][coords[1]].hasPiece) //check for possible 2 space move if pawn is in starting rank
            if (!board[coords[0] + 2][coords[1]].hasPiece) {
                char* move = (char*)malloc(sizeof(pawn->position) + sizeof(board[coords[0] + 2][coords[1]].position));
                if (move == nullptr) { return; }
                strCpy(move, pawn->position); strCat(move, board[coords[0] + 2][coords[1]].position);
                vectorPushBack(moves, move);
                //free(move);
            }
        //check for diagonal captures for pawn
        if (board[coords[0] + 1][coords[1] - 1].hasPiece) checkSpaceP(board, pawn, coords[0] + 1, coords[1] - 1, moves);
        if (board[coords[0] + 1][coords[1] + 1].hasPiece) checkSpaceP(board, pawn, coords[0] + 1, coords[1] + 1, moves);
    }
    else { //search row in front of white pawn
        if (coords[0] + 1 >= 0)
            if (!board[coords[0] - 1][coords[1]].hasPiece) {
                char* move = (char*)malloc(sizeof(pawn->position) + sizeof(board[coords[0] - 1][coords[1]].position));
                if (move == nullptr) { return; }
                strCpy(move, pawn->position); strCat(move, board[coords[0] - 1][coords[1]].position);
                vectorPushBack(moves, move);
                //free(move);
            }
        if (coords[0] == 6 && !board[coords[0] - 1][coords[1]].hasPiece) //check for possible 2 space move if pawn is in starting rank
            if (!board[coords[0] - 2][coords[1]].hasPiece) {
                char* move = (char*)malloc(sizeof(pawn->position) + sizeof(board[coords[0] - 2][coords[1]].position));
                if (move == nullptr) { return; }
                strCpy(move, pawn->position); strCat(move, board[coords[0] - 2][coords[1]].position);
                vectorPushBack(moves, move);
                //free(move);
            }
        //check for diagonal captures for pawn
        if (board[coords[0] - 1][coords[1] - 1].hasPiece) checkSpaceP(board, pawn, coords[0] - 1, coords[1] - 1, moves);
        if (board[coords[0] - 1][coords[1] + 1].hasPiece) checkSpaceP(board, pawn, coords[0] - 1, coords[1] + 1, moves);
    }

    //sort moves alphabetically
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
__device__ __host__ void pieceMoves(cell board[][RANK], state* state, cell* piece, int index, vector* moves) {
    //swicth case assignment based on piece index and respective function for said piece's move generation
    switch (index) {
    case 0:
        rookMoves(board, piece, moves);
        break;
    case 1:
        knightMoves(board, piece, moves);
        break;
    case 2:
        bishopMoves(board, piece, moves);
        break;
    case 3:
        queenMoves(board, piece, moves);
        break;
    case 4:
        kingMoves(board, state, piece, moves);
        break;
    case 5:
        pawnMoves(board, piece, moves);
        break;
    }
}

/*==================================================================================================================================================*/

__global__ void kernel(cell* board, state* game, int* threads) {
    int tRow = threadIdx.y; int tCol = threadIdx.x;
    int thread = tRow * BOARD_WIDTH + tCol;

    __shared__ cell lBoard[(BOARD_WIDTH)][(BOARD_WIDTH)];
    __shared__ char lPieceTypes[(PIECES)];
    __shared__ int lPieceValues[(PIECES)];

    lBoard[tRow][tCol] = board[thread];
    if (thread == 0) {
        addPieces(lBoard, game->fen);
    }
    if (thread < PIECES) {
        lPieceTypes[thread] = pieceType[thread];
        lPieceValues[thread] = pieceValue[thread];
    }
    __syncthreads();
    board[thread] = lBoard[tRow][tCol];
    char c = lBoard[tRow][tCol].piece;
    if (lBoard[tRow][tCol].colour == 'w') c = toLower(c);
    char* pos = strChr(lPieceTypes, c);
    if (pos != NULL) {
        int idx = pos - lPieceTypes;
        threads[thread] = idx;
    }
    else threads[thread] = -1;
}

int main(int argc, char* argv[]) {

    state game = initState(argv[1], *argv[2], argv[3], argv[4], atoi(argv[5]), atoi(argv[6]));
    cell board[BOARD_WIDTH][BOARD_WIDTH];
    initBoard(board, game.fen);
    //addPieces(board, game.fen);
    //printBoard(board);

    size_t numBytes = sizeof(cell) * BOARD_WIDTH * BOARD_WIDTH;

    cell* dBoard = 0;
    state* dState = 0;
    int* dThreads = 0;

    cell* hBoard = (cell*)malloc(BOARD_WIDTH * BOARD_WIDTH * sizeof(cell));
    int* hThreads = (int*)calloc(BOARD_WIDTH * BOARD_WIDTH, sizeof(int));

    cudaMemcpyToSymbol(pieceType, pieceTypes, PIECES * sizeof(char));
    cudaMemcpyToSymbol(pieceValue, pieceValues, PIECES * sizeof(int));

    cudaMalloc((void**)&dBoard, numBytes);
    cudaMalloc((void**)&dState, sizeof(state));
    cudaMalloc((void**)&dThreads, BOARD_WIDTH * BOARD_WIDTH * sizeof(int));

    cudaMemcpy(dBoard, board, numBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(dState, &game, sizeof(state), cudaMemcpyHostToDevice);

    dim3 gridSize(1, 1); dim3 blockSize(BOARD_WIDTH, BOARD_WIDTH);
    kernel << <gridSize, blockSize >> > (dBoard, dState, dThreads);

    cudaMemcpy(hBoard, dBoard, numBytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(hThreads, dThreads, BOARD_WIDTH * BOARD_WIDTH * sizeof(int), cudaMemcpyDeviceToHost);

    for (int i = 0; i < BOARD_WIDTH; i++) {
        for (int j = 0; j < BOARD_WIDTH; j++) {
            board[i][j] = hBoard[i * BOARD_WIDTH + j];
        }
    }

    cudaFree(dBoard);
    cudaFree(dState);
    cudaFree(dThreads);
    free(hBoard);
    free(hThreads);

    printBoard(board); printf("\n");
    vector moves; initVector(&moves);
    cell* king = &board[4][4]; state* state = &game;

    vector kingMoves; initVector(&kingMoves);
    checkSpacesK(board, state, king, &kingMoves);
    //check for possible castling moves
    castle(board, state, king, &kingMoves);

    vector enemyMoves; initVector(&enemyMoves);
    for (int i = 0; i < BOARD_WIDTH; i++) {
        for (int j = 0; j < BOARD_WIDTH; j++) {
            cell piece = board[i][j];
            if (piece.hasPiece && piece.colour != state->side) {
                char c = piece.piece; if (state->side == 'w') c = toLower(c);
                char* pos = strChr(pieceTypes, c);
                int idx = -1;
                if (pos != NULL) idx = pos - pieceTypes;
                if (idx == 4) checkSpacesK(board, state, &piece, &enemyMoves);
                pieceMoves(board, state, &piece, idx, &enemyMoves);
            }
        }
    }

    //remove all the moves a king can move to that an enemy piece can move to as well
    for (int i = 0; i < kingMoves.size; i++) {
        char* move = (char*)vectorGet(&kingMoves, i);
        char goal[3] = ""; strNcpy(goal, &move[2], 2); //extract space to move to
        for (int j = 0; j < enemyMoves.size; j++) { //search for space amongst enemy moves
            char* enemyMove = (char*)vectorGet(&enemyMoves, j);
            char enemy[3] = ""; strNcpy(enemy, &enemyMove[2], 2);
            if (!strCmp(goal, enemy)) { //if space is found
                free(kingMoves.items[i]);
                vectorRemove(&kingMoves, i); //remove it from 'moves' vector
                i--;
                break; //break out of enemy search since space has been found, no need to search further
            }
        }
    }
    vectorFree(&enemyMoves, 1);

    //add remaining legal moves
    if (kingMoves.size != 0) {
        for (int i = 0; i < kingMoves.size; i++)
            vectorPushBack(&moves, (char*)vectorGet(&kingMoves, i));
    }

    for (int i = 0; i < moves.size; i++) {
        char* move = (char*)vectorGet(&moves, i);
        printf("%s\n", move);
    }
    vectorFree(&moves, 1);

    return 0;
}

//char source[] = "Hello world";
//int* result = (int*)malloc(sizeof(int));
//
//char* src = 0; char* dest = 0; int* len = 0;
//cudaMalloc((void**)&src, (25 * sizeof(char)));
//cudaMalloc((void**)&dest, (25 * sizeof(char)));
//cudaMalloc((void**)&len, sizeof(int));
//cudaMemcpy(src, &source, 25 * sizeof(char), cudaMemcpyHostToDevice);
//kernel << <1, 1 >> > (dest, src, len);
//cudaMemcpy(result, len, sizeof(int), cudaMemcpyDeviceToHost);
//
//printf("%d", *result);
//
//cudaFree(src);
//cudaFree(dest);
//free(result);
//
//
//
//if (newSize == 0) {
//    free(ptr);
//    return NULL;
//}
//else if (!ptr) {
//    return malloc(newSize);
//}
//else if (newSize <= currSize) {
//    return ptr;
//}
//else {
//    //assert((ptr) && (newLength > originalLength));
//    void* ptrNew = malloc(newSize);
//    if (ptrNew) {
//        memCpy(ptrNew, ptr, currSize);
//        free(ptr);
//    }
//    return ptrNew;
//}
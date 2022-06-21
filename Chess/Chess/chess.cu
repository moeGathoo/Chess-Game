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

__device__ __host__ char* strNcat(char* destination, const char* source, size_t num) {
    // make `ptr` point to the end of the destination string
    char* ptr = destination + strLen(destination);

    // Appends characters of the source to the destination string
    while (*source != '\0' && num--)
        *ptr++ = *source++;

    // null terminate destination string
    *ptr = '\0';

    // destination string is returned by standard `strncat()`
    return destination;
}

__device__ __host__ int charToInt(char* p) {
    int k = 0;
    while (*p) {
        k = (k << 3) + (k << 1) + (*p) - '0';
        p++;
    }
    return k;
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

__device__ __host__ void* memMove(void* dest, const void* src, unsigned int n) {
    char* pDest = (char*)dest;
    const char* pSrc = (const char*)src;
    //allocate memory for tmp array
    char* tmp = (char*)malloc(sizeof(char) * n);
    if (NULL == tmp) return NULL;
    else {
        unsigned int i = 0;
        // copy src to tmp array
        for (i = 0; i < n; ++i)
            *(tmp + i) = *(pSrc + i);
        //copy tmp to dest
        for (i = 0; i < n; ++i)
            *(pDest + i) = *(tmp + i);
        free(tmp); //free allocated memory
    }
    return dest;
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
    v->items = 0; cudaMalloc((void**)&v->items, sizeof(void*) * v->capacity);
    //v->items = (void**)malloc(sizeof(void*) * v->capacity);
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
            cudaFree(v->items[i]);
        }
    cudaFree(v->items);
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
 * @brief Initializes empty board, sets 'position' attribute for all spaces.
 *
 * @param fen Required to add pieces to empty board.
 *
 * @return Nothing. Empty board is initialized and pieces are added
 * to relevant spaces based on fen string.
 */
__device__ __host__ void initBoard(cell board[][RANK]) {
    board[0][0].position = "a8"; board[0][1].position = "b8"; board[0][2].position = "c8"; board[0][3].position = "d8"; board[0][4].position = "e8"; board[0][5].position = "f8"; board[0][6].position = "g8"; board[0][7].position = "h8";
    board[1][0].position = "a7"; board[1][1].position = "b7"; board[1][2].position = "c7"; board[1][3].position = "d7"; board[1][4].position = "e7"; board[1][5].position = "f7"; board[1][6].position = "g7"; board[1][7].position = "h7";
    board[2][0].position = "a6"; board[2][1].position = "b6"; board[2][2].position = "c6"; board[2][3].position = "d6"; board[2][4].position = "e6"; board[2][5].position = "f6"; board[2][6].position = "g6"; board[2][7].position = "h6";
    board[3][0].position = "a5"; board[3][1].position = "b5"; board[3][2].position = "c5"; board[3][3].position = "d5"; board[3][4].position = "e5"; board[3][5].position = "f5"; board[3][6].position = "g5"; board[3][7].position = "h5";
    board[4][0].position = "a4"; board[4][1].position = "b4"; board[4][2].position = "c4"; board[4][3].position = "d4"; board[4][4].position = "e4"; board[4][5].position = "f4"; board[4][6].position = "g4"; board[4][7].position = "h4";
    board[5][0].position = "a3"; board[5][1].position = "b3"; board[5][2].position = "c3"; board[5][3].position = "d3"; board[5][4].position = "e3"; board[5][5].position = "f3"; board[5][6].position = "g3"; board[5][7].position = "h3";
    board[6][0].position = "a2"; board[6][1].position = "b2"; board[6][2].position = "c2"; board[6][3].position = "d2"; board[6][4].position = "e2"; board[6][5].position = "f2"; board[6][6].position = "g2"; board[6][7].position = "h2";
    board[7][0].position = "a1"; board[7][1].position = "b1"; board[7][2].position = "c1"; board[7][3].position = "d1"; board[7][4].position = "e1"; board[7][5].position = "f1"; board[7][6].position = "g1"; board[7][7].position = "h1";
    //printf("here\n");

    for (int i = RANK - 1; i >= 0; i--) {
        for (int j = FILEA; j < FILEA + RANK; j++) {
            board[i][j - FILEA].colour = '-';
            board[i][j - FILEA].hasPiece = false;
            board[i][j - FILEA].piece = '-';
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
        cell* space = &board[row][col];
        char* move = (char*)malloc(sizeof(piece->position) + sizeof(space->position));
        //printf("%s\n", (char*)move);
        if (move != NULL) { strCpy(move, piece->position); strCat(move, space->position); }
        else { return; }
        if (!space->hasPiece) { //if the space is empty, add it moves
            vectorPushBack(moves, move);
        }
        else {
            if (piece->colour != space->colour) { //if the piece is of the enmy colour, add to moves
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

/**
 * @brief Update the current state of the board, usually after moving a piece.
 *
 * @param state Pointer to state object to be updated
 * @param piece Pointer to space of piece that was moved.
 * @param flag Boolean indicating whether a pawn was moved or piece was captured.
 *
 * @return Nothing. State object is updated.
 */
__device__ __host__ void updateState(cell board[][RANK], state* state, cell* piece, bool flag) {
    //create new fen string representing board by traversing in row-major order
    char newFen[STR_BUFFER] = ""; char count;
    for (int i = 0; i < RANK; i++) {
        int empty = 0;
        for (int j = 0; j < RANK; j++) {
            cell* space = &board[i][j];
            if (space->piece != '-') {
                if (empty != 0) { count = empty + '0'; strNcat(newFen, &count, 1); }
                strNcat(newFen, &space->piece, 1);
                empty = 0;
            }
            else empty++;
        }
        if (empty != 0) { count = empty + '0'; strNcat(newFen, &count, 1); }
        char slash = '/';
        if (i != RANK - 1) strNcat(newFen, &slash, 1);
    }
    strNcpy(state->fen, newFen, STR_BUFFER);

    //castling piece types
    char k = 'k', q = 'q', r = 'r';
    if (piece->colour == 'w') { k = toUpper(k); q = toUpper(q); r = toUpper(r); }

    if (piece->piece == k) { //if king moves, remove all castling rights for that side
        char* posPtr = strChr(state->castle, k);
        if (posPtr != NULL) {
            int idxToDel = posPtr - state->castle;
            memMove(&state->castle[idxToDel], &state->castle[idxToDel+1], strLen(state->castle) - idxToDel);
        }
        posPtr = strChr(state->castle, q);
        if (posPtr != NULL) {
            int idxToDel = posPtr - state->castle;
            memMove(&state->castle[idxToDel], &state->castle[idxToDel + 1], strLen(state->castle) - idxToDel);
        }
    }
    else if (piece->piece == r) { //rook moves
        //find which side rook is and remove relevant castling rights flag
        int coords[2] = {0,0};
        toCoords(piece->position, coords);
        if (coords[1] > 5) {
            char* posPtr = strChr(state->castle, k);
            if (posPtr != NULL) {
                int idxToDel = posPtr - state->castle;
                memMove(&state->castle[idxToDel], &state->castle[idxToDel + 1], strLen(state->castle) - idxToDel);
            }
        }
        else {
            char* posPtr = strChr(state->castle, q);
            if (posPtr != NULL) {
                int idxToDel = posPtr - state->castle;
                memMove(&state->castle[idxToDel], &state->castle[idxToDel + 1], strLen(state->castle) - idxToDel);
            }
        }
        //free(coords);
    }
    if (strLen(state->castle) == 0) strCpy(state->castle, "-");
    //printf("castle updated");

    if (state->side == 'w') state->side = 'b'; //switch side to play
    else {
        state->side = 'w';
        state->fullMove++; //if black played, increment full moves
    }

    if (flag) state->halfMove = 0; //if pawn moved or piece captured, reset half moves
    else state->halfMove++; //else increment

    //if at least one king is missing, game is over
    bool blackKing = false, whiteKing = false;
    for (int i = 0; i < BOARD_WIDTH; i++) {
        for (int j = 0; j < BOARD_WIDTH; j++) {
            if (board[i][j].piece == 'k') blackKing = true;
            if (board[i][j].piece == 'K') whiteKing = true;
            if (blackKing && whiteKing) return;
        }
    }
    state->gameOver = false;
    return;
}


__device__ __host__ void printBoard(cell board[][RANK]) {
    for (int i = 0; i < RANK; i++) {
        for (int j = 0; j < RANK; j++) {
            printf("%c\t", board[i][j].piece);
        }
        printf("\n");
    }
    printf("\n");
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
    //printf("Here\n");
    for (int i = 0; i < BOARD_WIDTH; i++) {
        for (int j = 0; j < BOARD_WIDTH; j++) {
            cell piece = board[i][j];
            if (piece.hasPiece && piece.colour != state->side) {
                char c = piece.piece; if (state->side == 'w') c = toLower(c);
                char* pos = strChr(pieceType, c);
                int idx = -1;
                if (pos != NULL) idx = pos - pieceType;
                if (idx == 4) checkSpacesK(board, state, &piece, &enemyMoves);
                else pieceMoves(board, state, &piece, idx, &enemyMoves);
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
    vectorFree(&kingMoves, 0);
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
    //printf("%d\n", index);
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
__device__ __host__ bool checkMove(cell board[][RANK], state* state, char* move, char pieceTypes[]) {
    //get coordinates of start space and goal space in 2D array
    char start[3] = ""; strNcpy(start, &move[0], 2);
    int startCoords[2] = { 0,0 }; toCoords(start, startCoords);
    char goal[3] = ""; strNcpy(goal, &move[2], 2);
    int goalCoords[2] = { 0,0 }; toCoords(goal, goalCoords);

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
        char* posPtr = strChr(pieceTypes, startSpace->piece);
        if (posPtr != NULL) idx = posPtr - pieceTypes;
        else return false;
        pieceMoves(board, state, startSpace, idx, &moves); //generate all moves for respective piece
    }
    else { //piece is white, convert to lowercase for search
        char* posPtr = strChr(pieceTypes, toLower(startSpace->piece));
        if (posPtr != NULL) idx = posPtr - pieceTypes;
        else return false;
        pieceMoves(board, state, startSpace, idx, &moves);
    }

    //search for move in generated moves list
    if (vectorSearch(&moves, move)) { //move was found
        vectorFree(&moves, 1);
        movePiece(board, startSpace, goalSpace); //move piece
        if (idx == 4) { //if king was moved
            //check if castkling moved was made on queen side
            if (startCoords[1] - goalCoords[1] == 2) {
                //move rook accordingly
                cell* oldRook = &board[goalCoords[0]][0];
                cell* newRook = &board[goalCoords[0]][goalCoords[1] + 1];
                movePiece(board, oldRook, newRook);
            } //check if castkling moved was made on king side
            else if (startCoords[1] - goalCoords[1] == -2) {
                //move rook accordingly
                cell* oldRook = &board[goalCoords[0]][RANK - 1];
                cell* newRook = &board[goalCoords[0]][goalCoords[1] - 1];
                movePiece(board, oldRook, newRook);
            }
        }
        updateState(board, state, goalSpace, flag); //update state to indicate move played
        //free(startCoords); free(goalCoords);
        return true;
    }
    //move was not found (illegal move), return false
    vectorFree(&moves, 1);
    return false;
}

/*==================================================================================================================================================*/

__global__ void distributor(state* prevGame, char** moves) {
    int thread = threadIdx.x;

    __shared__ char lPieceTypes[(PIECES)];
    __shared__ int lPieceValues[(PIECES)];

    if (thread == 0) {
        for (int i = 0; i < PIECES; i++) {
            lPieceTypes[i] = pieceType[i];
            lPieceValues[i] = pieceValue[i];
        }
    }
    cell myBoard[BOARD_WIDTH][BOARD_WIDTH];
    char* move = 0; cudaMalloc((void**)&move, 5 * sizeof(char));
    strNcpy(move, moves[thread], 5);
    state myGame; memcpy(&myGame, prevGame, sizeof(state));
    __syncthreads();

    initBoard(myBoard); addPieces(myBoard, (char*)prevGame->fen);
    checkMove(myBoard, &myGame, (char*)move, lPieceTypes);
    cudaFree(move);
    printf("%s\n", (char*)myGame.fen);
}

__global__ void master(state* game, int depth) {
    int tRow = threadIdx.y; int tCol = threadIdx.x;
    int thread = tRow * BOARD_WIDTH + tCol;

    __shared__ cell lBoard[(BOARD_WIDTH)][(BOARD_WIDTH)];
    __shared__ char lPieceTypes[(PIECES)];
    __shared__ int lPieceValues[(PIECES)];

    //lBoard[tRow][tCol] = board[thread];
    if (thread == 0) {
        initBoard(lBoard);
        addPieces(lBoard, game->fen);
    }
    if (thread < PIECES) {
        lPieceTypes[thread] = pieceType[thread];
        lPieceValues[thread] = pieceValue[thread];
    }
    __syncthreads();
    char c = lBoard[tRow][tCol].piece;
    cell myPiece = lBoard[tRow][tCol];
    if (lBoard[tRow][tCol].colour == 'w') c = toLower(c);
    char* pos = strChr(lPieceTypes, c);
    int idx = -1;
    if (pos != NULL) idx = pos - lPieceTypes;
    if (myPiece.hasPiece && myPiece.colour == game->side) {
        vector moves; initVector(&moves);
        pieceMoves(lBoard, game, &myPiece, idx, &moves);
        if (moves.size > 0)
            //distributor << <1, moves.size >> > (game, (char**)moves.items);
            printf("Piece: %c on %s with %d moves\n", lPieceTypes[idx], myPiece.position, moves.size);
        vectorFree(&moves, 1);
    }
}

int main(int argc, char* argv[]) {

    state game = initState(argv[1], *argv[2], argv[3], argv[4], atoi(argv[5]), atoi(argv[6]));
    cell board[BOARD_WIDTH][BOARD_WIDTH];
    initBoard(board);
    //addPieces(board, game.fen);
    //printBoard(board);

    state* dState = 0;
    char* dTest = 0;
    char* hTest = (char*)malloc(5 * sizeof(char));

    cudaMemcpyToSymbol(pieceType, pieceTypes, PIECES * sizeof(char));
    cudaMemcpyToSymbol(pieceValue, pieceValues, PIECES * sizeof(int));

    cudaMalloc((void**)&dState, sizeof(state));
    cudaMalloc((void**)&dTest, 5 * sizeof(char));

    cudaMemcpy(dState, &game, sizeof(state), cudaMemcpyHostToDevice);

    dim3 gridSize(1, 1); dim3 blockSize(BOARD_WIDTH, BOARD_WIDTH);
    master << <gridSize, blockSize >> > (dState, 2);
    cudaMemcpy(hTest, dTest, 5 * sizeof(char), cudaMemcpyDeviceToHost);

    printf("%s\n", (char*)hTest);

    cudaFree(dState);
    cudaFree(dTest);
    free(hTest);

    //printBoard(board); printf("\n");

    return 0;
}

//nvcc chess.cu helpers/board.cu helpers/state.cu -o chess chess.cu -rdc=true -lcudadevrt

    //char** moves = (char**)malloc(27 * sizeof(char));
    //for (int i = 0; i < 27; i++) {
    //    moves[i] = (char*)malloc(5 * sizeof(char));
    //}
    //int* thread = (int*)malloc(sizeof(int));
    //thread[0] = threadIdx.x;
    //preTest << <1, 1 >> > (moves[threadIdx.x], thread);
    //free(thread);
    //printf("%s\n", moves[threadIdx.x]);
    //char* res = moves[0];
    //memcpy(master, res, 5);
    ////for (int i = 0; i < 27; i++) {
    ////    free(moves[i]);
    ////}
    //free(moves);

/*
__global__ void printer(state* game, char* fen, char** moves) {
    int thread = threadIdx.x;

    __shared__ char lPieceTypes[(PIECES)];
    __shared__ int lPieceValues[(PIECES)];

    cell lBoard[BOARD_WIDTH][BOARD_WIDTH];
    state myGame; memcpy(&myGame, game, sizeof(state));
    initBoard(lBoard, fen);
    addPieces(lBoard, fen);

    if (thread == 0) {
        for (int i = 0; i < PIECES; i++) {
            lPieceTypes[i] = pieceType[i];
            lPieceValues[i] = pieceValue[i];
        }
    }
    __syncthreads();
    checkMove(lBoard, &myGame, moves[thread], lPieceTypes);
    //printf("My thread number: %d, My fen: %s, My move: %s\n",thread, myGame.fen, moves[thread]);
}

__global__ void preTest(char** move, int size) {
    /*if (*thread == 0) {
        char* pos = "one";
        memcpy(test, pos, 5);
    }
    if (*thread == 1) {
        char* pos = "two";
        memcpy(test, pos, 5);
    }
    if (*thread == 2) {
        char* pos = "three";
        memcpy(test, pos, 5);
    }*/
    /*for (int i = 0; i < size; i++) {
        printf("%s\n", (char*)move[i]);
    }
    //printer<<<1,1>>>(move[threadIdx.x]);
}

__global__ void test(char* master, state* game) {
    int tRow = threadIdx.y; int tCol = threadIdx.x;
    int thread = tRow * BOARD_WIDTH + tCol;

    __shared__ cell lBoard[(BOARD_WIDTH)][(BOARD_WIDTH)];
    __shared__ char lPieceTypes[(PIECES)];
    __shared__ int lPieceValues[(PIECES)];

    //lBoard[tRow][tCol] = board[thread];
    if (thread == 0) {
        initBoard(lBoard, game->fen);
        addPieces(lBoard, game->fen);
    }
    if (thread < PIECES) {
        lPieceTypes[thread] = pieceType[thread];
        lPieceValues[thread] = pieceValue[thread];
    }
    __syncthreads();
    char c = lBoard[tRow][tCol].piece;
    cell myPiece = lBoard[tRow][tCol];
    if (lBoard[tRow][tCol].colour == 'w') c = toLower(c);
    char* pos = strChr(lPieceTypes, c);
    int idx = -1;
    if (pos != NULL) idx = pos - lPieceTypes;
    if (myPiece.hasPiece && myPiece.colour == game->side) {
        vector moves; initVector(&moves);
        pieceMoves(lBoard, game, &myPiece, idx, &moves);
        //if (tRow == 6 && tCol == 4)
        //    checkMove(lBoard, game, (char*)moves.items[0], lPieceTypes);
        //printBoard(lBoard); printf("\n");
        //printf("%s\n", game->fen);
        //char* nextFen = 0; cudaMalloc((void**)&nextFen, STR_BUFFER * sizeof(char));
        //memcpy(nextFen, game->fen, STR_BUFFER * sizeof(char));
        if (moves.size > 0) printer << <1, moves.size >> > (game, game->fen, (char**)moves.items);
        //cudaFree(nextFen);
        //printf("%d\n", moves.size);
        vectorFree(&moves, 1);
    }
}
*/
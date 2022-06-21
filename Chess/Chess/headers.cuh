#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <ctype.h>

//definitions used across project
#define PIECES 6                //number of piece types
#define RANK 8                  //number of ranks (and files)
#define FILEA 'a'               //starting file count
#define STR_BUFFER 128          //buffer for fen string
#define BOARD_WIDTH 8           //size of board
#define VECTOR_INIT_CAPACITY 4  //initial vector capacity

//structure for vector
typedef struct vector {
    void** items;
    int capacity;
    int size;
} vector;

//structure for current state of board
typedef struct state {
    char fen[STR_BUFFER];             //fen string representation of board
    char side;              //side to play
    char castle[5] = "";          //castling availability
    char* enPassant;       //en passant move
    int halfMove;           //number of half moves (increments after each side moves) since capture or pawn move
    int fullMove;           //number of full moves, increments after black plays
    bool check = false;     //is there a king in check
    bool gameOver = false;  //is the game over
} state;

//structure to represnt space on board
typedef struct cell {
    char* position;       //name of position eg. a5, b3
    bool hasPiece;  //does space have a piece
    char piece;       //if not, default indication
    char colour;            //if yes, what color is the piece
} cell;

//state functions
state initState(char* fen, char side, char* castle, char* enPassant, int halfMove, int fullMove);

//board functions
//void initBoard(cell board[][RANK], char* fen);
//void printBoard(cell board[][RANK]);

//main functions
__device__ __host__ unsigned int strLen(char* p);
__device__ __host__ char* strCpy(char* destination, const char* source);
__device__ __host__ char* strCat(char* dest, const char* src);
__device__ __host__ char* strChr(char* str, char c);
__device__ __host__ int strCmp(const char* str1, const char* str2);
__device__ __host__ char* strNcpy(char* dest, char* src, size_t num);
__device__ __host__ int charToInt(char* p);
__device__ __host__ int isInt(char c);
__device__ __host__ int isLower(char c);
__device__ __host__ int isUpper(char c);
__device__ __host__ int toUpper(char c);
__device__ __host__ int toLower(char c);
__device__ __host__ void memCpy(void* dest, void* src, size_t n);
__device__ __host__ void* reAlloc(void* ptr, size_t currSize, size_t newSize);
__device__ __host__ void toCoords(char* pos, int* coords);
__device__ __host__ void initVector(vector* v);
__device__ __host__ static void vectorResize(vector* v, int capacity);
__device__ __host__ void vectorPushBack(vector* v, void* item);
__device__ __host__ void* vectorGet(vector* v, int index);
__device__ __host__ void vectorRemove(vector* v, int index);
__device__ __host__ int vectorSearch(vector* v, char* str);
__device__ __host__ void vectorFree(vector* v, int malloced);
__device__ __host__ void addPieces(cell board[][BOARD_WIDTH], char* fen);
__device__ __host__ void checkSpace(cell board[][RANK], cell* piece, int row, int col, vector* moves);
__device__ __host__ void checkSpaceP(cell board[][RANK], cell* piece, int row, int col, vector* moves);
__device__ __host__ void checkSpacesK(cell board[][RANK], state* game, cell* king, vector* moves);
__device__ __host__ void castle(cell board[][RANK], state* state, cell* king, vector* moves);
__device__ __host__ void movePiece(cell board[][RANK], cell* startSpace, cell* goalSpace);
__device__ __host__ void rookMoves(cell board[][RANK], cell* rook, vector* moves);
__device__ __host__ void knightMoves(cell board[][RANK], cell* knight, vector* moves);
__device__ __host__ void bishopMoves(cell board[][RANK], cell* bishop, vector* moves);
__device__ __host__ void queenMoves(cell board[][RANK], cell* queen, vector* moves);
__device__ __host__ void kingMoves(cell board[][RANK], state* state, cell* king, vector* moves);
__device__ __host__ void pawnMoves(cell board[][RANK], cell* pawn, vector* moves);
__device__ __host__ void pieceMoves(cell board[][RANK], state* state, cell* piece, int index, vector* moves);
__global__ void kernel(cell* board, state* game, int* threads);

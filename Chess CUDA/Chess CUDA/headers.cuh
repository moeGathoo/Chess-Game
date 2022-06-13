#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <ctype.h>

//definitions used across project
#define PIECES 6        //number of piece types
#define RANK 8          //number of ranks (and files)
#define FILEA 'a'       //starting file count
#define STR_BUFFER 128  //buffer for fen string

//vector structure
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
    char *position;       //name of position eg. a5, b3
    bool hasPiece = false;  //does space have a piece
    char piece = '-';       //if not, default indication
    char colour;            //if yes, what color is the piece
} cell;

//gobal variables used across project
extern char pieceTypes[6];
extern int pieceValues[6];
extern cell board[RANK][RANK];
extern vector black[PIECES];
extern vector white[PIECES];

//vector functions
void initVector(vector* v);
int vectorSize(vector* v);
void vectorPushBack(vector* v, void* item);
void vectorSet(vector* v, int index, void* item);
void* vectorGet(vector* v, int index);
void vectorSort(vector* v);
bool vectorSearch(vector* v, char* str);
void vectorRemove(vector* v, int index);
void vectorClear();
void vectorFree(vector* v);

//board functions
void addPieces(char* fen);
void initBoard(char* fen);
void resetBoard();
void checkSpace(cell* piece, int row, int col, vector* moves);
void checkSpaceP(cell* piece, int row, int col, vector* moves);
void checkSpacesK(state* game, cell* king, vector* moves);
void castle(state* state, cell* king, vector* moves);
void movePiece(cell* startSpace, cell* goalSpace);
void printBoard();

//state functions
state initState(char* fen, char side, char* castle, char* enPassant, int halfMove, int fullMove);
void toCoords(char* pos, int* coords);
void getPositions();
void updateState(state* state, cell* piece, bool flag);
void printState(state* currState);

//moves functions
void rookMoves(cell* rook, vector* moves);
void knightMoves(cell* knight, vector* moves);
void bishopMoves(cell* bishop, vector* moves);
void queenMoves(cell* queen, vector* moves);
void kingMoves(state* state, cell* king, vector* moves);
void pawnMoves(cell* pawn, vector* moves);
void pieceMoves(state* state, cell* piece, int index, vector* moves);
bool checkMove(state* state, char* move);

//engine functions
void assign(char* bestMove);
int evaluation(state* state);
int advEvaluation(state* state);
int alphaBeta(state currState, int depth, int alpha, int beta, char* bestMove, bool first);
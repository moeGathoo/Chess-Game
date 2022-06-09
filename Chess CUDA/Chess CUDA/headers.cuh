#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

//vector structure
typedef struct vector {
    void** items;
    int capacity;
    int size;
} vector;

//structure for current state of board
typedef struct state {
    char* fen;             //fen string representation of board
    char side;              //side to play
    char* castle;          //castling availability
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

//definitions used across project
#define PIECES 6    //number of piece types
#define RANK 8      //number of ranks (and files)
#define FILE 'a'    //starting file count

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
void vectorRemove(vector* v, int index);
void vectorClear(vector* v);
void vectorFree(vector* v);

//board functions
void addPieces(char* fen);
void initBoard(char* fen);
void printBoard();

//state functions
void getPositions();
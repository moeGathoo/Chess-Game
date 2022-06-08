#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

//definitions used across project
#define PIECES 6    //number of piece types
#define RANK 8      //number of ranks (and files)
#define FILE 'a'    //starting file count

//gobal variables used across project
extern char pieceTypes[6];
extern int pieceValues[6];
//extern cell board[RANK][RANK];
//extern std::vector<cell*> black[PIECES];
//extern std::vector<cell*> white[PIECES];

//structure to represnt space on board
struct cell {
    char position[2];        //name of position eg. a5, b3
    bool hasPiece = false;  //does space have a piece
    char piece = '-';       //if not, default indication
    char colour;            //if yes, what color is the piece
};

__global__ void addKernel(int* c, const int* a, const int* b);
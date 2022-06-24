/**
 * @file headers.h
 * @brief header file containing initialization of all functions
 */

//libraries included and used across project
#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <string>
#include <chrono>

//definitions used across project
#define PIECES 6    //number of piece types
#define RANK 8      //number of ranks (and files)
#define FILE 'a'    //starting file count

using namespace std;

//gobal variables used across project
extern char pieceTypes[6];
extern int pieceValues[6];
extern struct cell board[RANK][RANK];
extern vector<cell*> black[PIECES];
extern vector<cell*> white[PIECES];

//*state functions
//structure for current state of board
struct state{
    string fen;             //fen string representation of board
    char side;              //side to play
    string castle;          //castling availability
    string enPassant;       //en passant move
    int halfMove;           //number of half moves (increments after each side moves) since capture or pawn move
    int fullMove;           //number of full moves, increments after black plays
    bool check = false;     //is there a king in check
    bool gameOver = false;  //is the game over
};
state initState(string fen, char side, string castle, string enPassant, int halfMove, int fullMove);
void toCoords(string pos, int* coords);
void getPositions();
void updateState(state* state, cell* piece, bool flag);
void printState(state* currState);

//*board functions
//structure to represnt space on board
struct cell{
    string position;        //name of position eg. a5, b3
    bool hasPiece = false;  //does space have a piece
    char piece = '-';       //if not, default indication
    char colour;            //if yes, what color is the piece
};
void addPieces(string fen);
void initBoard(string fen);
void resetBoard();
void checkSpace(cell *piece, int row, int col, vector<string>* moves);
void checkSpaceP(cell *piece, int row, int col, vector<string>* moves);
void checkSpacesK(cell *piece, vector<string>* moves);
void castle(state* state, cell* king, vector<string> *moves);
void movePiece(cell* startSpace, cell* goalSpace);
void printBoard();

//*move functions
void rookMoves(cell* rook, vector<string> *moves);
void knightMoves(cell* knight, vector<string> *moves);
void bishopMoves(cell* bishop, vector<string> *moves);
void queenMoves(cell* queen, vector<string> *moves);
void kingMoves(state* state, cell* king, vector<string> *moves);
void pawnMoves(cell* pawn, vector<string> *moves);
void pieceMoves(state* state, cell* piece, int index, vector<string> *moves);
bool checkMove(state *state, string *move);

//engine functions
int evaluation(state* state);
int miniMax(state state, int depth, string* bestMove, bool first);
int advEvaluation(state *state);
int alphaBeta(state currState, int depth, int alpha, int beta, string* bestMove, bool first);

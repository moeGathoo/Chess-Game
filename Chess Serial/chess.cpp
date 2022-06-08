/**
 * @brief Plays a game of chess for 10 full moves
 * compile command:
 * * g++ chess.cpp helpers/state.cpp helpers/board.cpp helpers/moves.cpp helpers/engine.cpp -o chess
 * execute command:
 * * ./chess rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1
 */

#include "headers.h"

//initializes values of all global variables
char pieceTypes[6] = {'r', 'n', 'b', 'q', 'k', 'p'};    //types of pieces on board
int pieceValues[6] = {50, 30, 30, 90, 900, 10};         //piece weightings for search and evaluation algorithms
cell board[RANK][RANK];                                 //board to play on
vector<cell*> black[PIECES];                            //array of vectors for each piece type (pointers to spaces on baord that have the piece)
vector<cell*> white[PIECES];                            //array of vectors for each piece type (pointers to spaces on baord that have the piece)

/**
 * @brief Main function to play game.
 * 
 * @param argc Number of arguments.
 * @param argv Each argument pertains to a particular part of fen notation.
 * ? argv[0]: fen string representation of board
 * ? argv[1]: side to play
 * ? argv[2]: castling rights
 * ? argv[3]: en passant move
 * ? argv[4]: half moves (total moves since pawn move or piece capture)
 * ? argv[5]: full moves (increments by 1 after black moves)
 * 
 * @return int 
 */
int main(int argc, char* argv[]) {
    state game = initState(argv[1], *argv[2], argv[3], argv[4], atoi(argv[5]), atoi(argv[6]));
    //set up board
    initBoard(game.fen);

    //plays game up to 10 full moves using alpha-beta pruning algorithm
    //prints move made
    for (int i = 0; i < 10; i++) {
        string move;
        alphaBeta(game, 4, -10001, 10001, &move, true);
        checkMove(&game, &move);
        cout << move << endl;
        if (game.gameOver) break; //end game of game is over
    }
    return 0;
}
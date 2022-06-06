#include "headers.h"

char pieceTypes[6] = {'r', 'n', 'b', 'q', 'k', 'p'};
int pieceValues[6] = {50, 30, 30, 90, 900, 10};
cell board[RANK][RANK];
vector<cell*> black[PIECES];
vector<cell*> white[PIECES];

int main(int argc, char* argv[]) {
    state game = initState(argv[1], *argv[2], argv[3], argv[4], atoi(argv[5]), atoi(argv[6]));

    initBoard(game.fen);
    // printBoard();
    
    vector<cell*>* pieces; vector<string> moves;
    if (game.side == 'b') pieces = black;
    else pieces = white;
    for (int i = 0; i < PIECES; i++)
        if (pieces[i].size() != 0)
            for (cell* piece : pieces[i])
                pieceMoves(&game, piece, i, &moves);

    for (int i = 0; i < 10; i++) {
        string move;
        miniMax(game, 4, &move, true);
        checkMove(&game, &move);
        // printBoard();
        cout << move << endl;
    }
    return 0;
}

//compile command:
//g++ chess.cpp helpers/state.cpp helpers/board.cpp helpers/moves.cpp helpers/engine.cpp -o chess
//execute command
//./chess rnbqkbnr/ppp2ppp/3pp3/8/6P1/3PP3/PPP2P1P/RNBQKBNR b KQkq - 0 3

    // string move;
    // alphaBeta(game, 4, -10001, 10001, &move, true);
    // cout << endl << move << endl;
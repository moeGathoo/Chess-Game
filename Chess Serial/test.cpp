#include "headers.h"

char pieceTypes[6] = {'r', 'n', 'b', 'q', 'k', 'p'};
int pieceValues[6] = {50, 30, 30, 90, 900, 10};
cell board[RANK][RANK];
vector<cell*> black[PIECES];
vector<cell*> white[PIECES];

int main(int argc, char* argv[]) {
    state game = initState(argv[1], *argv[2],
                                   argv[3], argv[4],
                                   atoi(argv[5]), atoi(argv[6]));

    initBoard(game.fen);
    getPositions();
    // printState(&game);
    // printBoard();
    cout << endl;
    
    vector<cell*>* pieces; vector<string> moves;
    if (game.side == 'b') pieces = black;
    else pieces = white;
    for (int i = 0; i < PIECES; i++)
        if (pieces[0].size() != 0)
            for (cell* piece : pieces[i])
                pieceMoves(&game, piece, i, &moves);

    for (string move: moves)
        cout << move << endl;

    return 0;
}

//compile command:
//g++ test.cpp helpers/state.cpp helpers/board.cpp helpers/moves.cpp helpers/engine.cpp -o test
//execute command
//./test rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1
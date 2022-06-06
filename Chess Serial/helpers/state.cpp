#include "../headers.h"


state initState(string fen, char side, string castle, string enPassant, int halfMove, int fullMove) {
    struct state curr;
    curr.fen = fen;
    curr.side = side;
    curr.castle = castle;
    curr.enPassant = enPassant;
    curr.halfMove = halfMove;
    curr.fullMove = fullMove;
    return curr;
}


void toCoords(string pos, int* coords) {
    int row = RANK - (pos[1] - '0'); 
    int col = pos[0] - FILE; 
    coords[0] = row; coords[1] = col;
}


void toPos(int col, int row, vector<string>* vec) {
    char c = col;
    string pos = string() + c + to_string(row);
    vec->push_back(pos);
}


void getPositions() {
    for (auto& v : black) v.clear();
    for (auto& v : white) v.clear();
    
    for (int i = 0; i < RANK; i++) {
        for (int j = 0; j < RANK; j++) {
            if (board[i][j].piece != '-') {
                char *foo = find(begin(pieceTypes), end(pieceTypes), board[i][j].piece); 
                if (foo != end(pieceTypes)) {
                    int idx = distance(pieceTypes, foo);
                    black[idx].push_back(&board[i][j]);
                }
                else {
                    foo = find(begin(pieceTypes), end(pieceTypes), tolower(board[i][j].piece)); 
                    int idx = distance(pieceTypes, foo);
                    white[idx].push_back(&board[i][j]);
                }
            }
        }
    }
}


void updateState(state* state, cell* piece, bool flag) {
    string newFen = "";
    for (int i = 0; i < RANK; i++) {
        int empty = 0;
        for (int j = 0; j < RANK; j++) {
            cell space = board[i][j];
            if (space.piece != '-') {
                if (empty != 0) newFen += to_string(empty);
                newFen += space.piece;
                empty = 0;
            }
            else empty++;
        }
        if (empty != 0) newFen += to_string(empty);
        if (i != RANK-1) newFen += "/";
    }
    state->fen = newFen;
    
    char k = 'k', q = 'q', r = 'r';
    if (piece->colour == 'w') {k = toupper(k); q = toupper(q); r = toupper(r);}
    if (piece->piece == k) {
        state->castle.erase(remove(state->castle.begin(), state->castle.end(), q), state->castle.end());
        state->castle.erase(remove(state->castle.begin(), state->castle.end(), k), state->castle.end());
    }
    else if (piece->piece == r) {
        int* coords = new int[2];
        toCoords(piece->position, coords);
        if (coords[1] < 5) state->castle.erase(remove(state->castle.begin(), state->castle.end(), q), state->castle.end());
        else state->castle.erase(remove(state->castle.begin(), state->castle.end(), k), state->castle.end());
    }
    if (state->castle.length() == 0) state->castle = "-";
    
    if (state->side == 'w') state->side = 'b';
    else {
        state->side = 'w';
        state->fullMove++;
    }

    if (flag) state->halfMove = 0;
    else state->halfMove++;

    if (white[4].size() == 0 || black[4].size() == 0) state->gameOver = true;

    // vector<cell*>* enemy; vector<string> enemyMoves; string king = "";
    // if (piece->colour == 'b') {enemy = white; king = black[4][0]->position;}
    // else {enemy = black; king = white[4][0]->position;}
    // for (int i = 0; i < PIECES; i++) {
    //     if (i == 4) checkSpacesK(enemy[4][0], &enemyMoves);
    //     else if (enemy[i].size() != 0) {
    //         for (int j = 0; j < enemy[i].size(); j++)
    //             pieceMoves(state, enemy[i][j], i, &enemyMoves);
    //     }
    // }
    // for (string enemyMove : enemyMoves) {
    //     string enemy = enemyMove.substr(2,4);
    //     if (enemy == king) {
    //         state->gameOver = true;
    //         break;
    //     }
    // }

    // vector<string> bKingMoves, wKingMoves;
    // pieceMoves(state, black[4][0], 4, &bKingMoves);
    // pieceMoves(state, white[4][0], 4, &wKingMoves);
    // if (wKingMoves.size() == 0 || bKingMoves.size() == 0) state->gameOver = true;
}


void printState(state* currState) {
    cout << currState->fen << " ";
    cout << currState->side << " ";
    cout << currState->castle << " ";
    cout << currState->enPassant << " ";
    cout << currState->halfMove << " ";
    cout << currState->fullMove << endl;
}
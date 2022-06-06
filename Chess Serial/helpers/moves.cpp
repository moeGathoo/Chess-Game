#include "../headers.h"


void rookMoves(cell* rook, vector<string> *moves) {
    int coords[2] = {0, 0};
    toCoords(rook->position, coords);

    int i = 1;
    while (coords[0]+i<RANK && !board[coords[0]+i][coords[1]].hasPiece) i++;
    for (int j = 1; j <= i; j++) checkSpace(rook, coords[0]+j, coords[1], moves);
    
    i = 1;
    while (coords[0]-i>=0 && !board[coords[0]-i][coords[1]].hasPiece) i++;
    for (int j = 1; j <= i; j++) checkSpace(rook, coords[0]-j, coords[1], moves);
    
    i = 1;
    while (coords[1]+i<RANK && !board[coords[0]][coords[1]+i].hasPiece) i++;
    for (int j = 1; j <= i; j++) checkSpace(rook, coords[0], coords[1]+j, moves);
    
    i = 1;
    while (coords[1]-i>=0 && !board[coords[0]][coords[1]-i].hasPiece) i++;
    for (int j = 1; j <= i; j++) checkSpace(rook, coords[0], coords[1]-j, moves);

    sort(moves->begin(), moves->end());
}


void knightMoves(cell* knight, vector<string> *moves) {
    int coords[2] = {0, 0};
    toCoords(knight->position, coords);

    checkSpace(knight, coords[0]-1, coords[1]-2, moves);
    checkSpace(knight, coords[0]-2, coords[1]-1, moves);
    checkSpace(knight, coords[0]-2, coords[1]+1, moves);
    checkSpace(knight, coords[0]-1, coords[1]+2, moves);
    checkSpace(knight, coords[0]+1, coords[1]+2, moves);
    checkSpace(knight, coords[0]+2, coords[1]+1, moves);
    checkSpace(knight, coords[0]+1, coords[1]-2, moves);
    checkSpace(knight, coords[0]+2, coords[1]-1, moves);

    sort(moves->begin(), moves->end());
}


void bishopMoves(cell* bishop, vector<string> *moves) {
    int coords[2] = {0, 0};
    toCoords(bishop->position, coords);
    
    int i = 1;
    while (coords[0]+i<RANK && coords[1]+i<RANK && !board[coords[0]+i][coords[1]+i].hasPiece) i++;
    for (int j = 1; j <= i; j++) checkSpace(bishop, coords[0]+j, coords[1]+j, moves);

    i = 1;
    while (coords[0]+i<RANK && coords[1]-i>-1 && !board[coords[0]+i][coords[1]-i].hasPiece) i++;
    for (int j = 1; j <= i; j++) checkSpace(bishop, coords[0]+j, coords[1]-j, moves);

    i = 1;
    while (coords[0]-i>-1 && coords[1]+i<RANK && !board[coords[0]-i][coords[1]+i].hasPiece) i++;
    for (int j = 1; j <= i; j++) checkSpace(bishop, coords[0]-j, coords[1]+j, moves);
    
    i = 1;
    while (coords[0]-i>-1 && coords[1]-i>-1 && !board[coords[0]-i][coords[1]-i].hasPiece) i++;
    for (int j = 1; j <= i; j++) checkSpace(bishop, coords[0]-j, coords[1]-j, moves);

    sort(moves->begin(), moves->end());
}


void queenMoves(cell* queen, vector<string> *moves) {
    bishopMoves(queen, moves);
    rookMoves(queen, moves);
    sort(moves->begin(), moves->end());
}


void kingMoves(state* state, cell* king, vector<string> *moves) {
    checkSpacesK(king, moves);
    vector<cell*>* enemy; vector<string> enemyMoves;
    if (king->colour == 'b') enemy = white;
    else enemy = black;

    for (int i = 0; i < PIECES; i++) {
        if (i == 4) checkSpacesK(enemy[4][0], &enemyMoves);
        else if (enemy[i].size() != 0) {
            for (int j = 0; j < enemy[i].size(); j++)
                pieceMoves(state, enemy[i][j], i, &enemyMoves);
        }
    }

    for (int i = 0; i < moves->size(); i++) {
        string move = moves->at(i);
        string goal = move.substr(2,4);
        for (string enemyMove : enemyMoves) {
            string enemy = enemyMove.substr(2,4);
            if (goal == enemy) {
                moves->erase(moves->begin() + i);
                i--;
                break;
            }
        }
    }

    castle(state, king, moves);

    sort(moves->begin(), moves->end());
}


void pawnMoves(cell* pawn, vector<string> *moves) {
    int coords[2] = {0, 0};
    toCoords(pawn->position, coords); 

    if (pawn->colour == 'b') { 
        if (!board[coords[0]+1][coords[1]].hasPiece) moves->push_back(pawn->position + board[coords[0]+1][coords[1]].position);
        if (coords[0] == 1 && !board[coords[0]+1][coords[1]].hasPiece)
            if (!board[coords[0]+2][coords[1]].hasPiece) moves->push_back(pawn->position + board[coords[0]+2][coords[1]].position);
        
        if (board[coords[0]+1][coords[1]-1].hasPiece) checkSpaceP(pawn, coords[0]+1, coords[1]-1, moves);
        if (board[coords[0]+1][coords[1]+1].hasPiece) checkSpaceP(pawn, coords[0]+1, coords[1]+1, moves);
    }
    else { 
        if (!board[coords[0]-1][coords[1]].hasPiece) moves->push_back(pawn->position + board[coords[0]-1][coords[1]].position);
        if (coords[0] == 6 && !board[coords[0]-1][coords[1]].hasPiece)
            if (!board[coords[0]-2][coords[1]].hasPiece) moves->push_back(pawn->position + board[coords[0]-2][coords[1]].position);

        if (board[coords[0]-1][coords[1]-1].hasPiece) checkSpaceP(pawn, coords[0]-1, coords[1]-1, moves);
        if (board[coords[0]-1][coords[1]+1].hasPiece) checkSpaceP(pawn, coords[0]-1, coords[1]+1, moves);
    }

    sort(moves->begin(), moves->end());
}


void pieceMoves(state* state, cell* piece, int index, vector<string> *moves) {
    switch (index) {
        case 0:
            rookMoves(piece, moves);
            break;
        case 1:
            knightMoves(piece, moves);
            break;
        case 2:
            bishopMoves(piece, moves);
            break;
        case 3:
            queenMoves(piece, moves);
            break;
        case 4:
            kingMoves(state, piece, moves);
            break;
        case 5:
            pawnMoves(piece, moves);
            break;
    }
}


bool checkMove(state *state, string *move) {
    string start = move->substr(0,2);
    int* startCoords = new int[2]; toCoords(start, startCoords);
    string goal = move->substr(2,3);
    int* goalCoords = new int[2]; toCoords(goal, goalCoords);

    cell *startSpace = &board[startCoords[0]][startCoords[1]];
    cell *goalSpace = &board[goalCoords[0]][goalCoords[1]];

    bool flag = false;
    if (startSpace->piece == 'p' || startSpace->piece == 'P' || goalSpace->hasPiece) flag = true;
    int idx;
    vector<string> moves;
    if (startSpace->colour == 'b') {
        idx = find(begin(pieceTypes), end(pieceTypes), startSpace->piece) - begin(pieceTypes);
        pieceMoves(state, startSpace, idx, &moves);
    }
    else {
        idx = find(begin(pieceTypes), end(pieceTypes), tolower(startSpace->piece)) - begin(pieceTypes);
        pieceMoves(state, startSpace, idx, &moves);
    }
    
    if (find(moves.begin(), moves.end(), *move) != moves.end()) {
        movePiece(startSpace, goalSpace);
        if (idx == 4) {
            if (startCoords[1] - goalCoords[1] == 2) {
                cell *oldRook = &board[goalCoords[0]][0];
                cell *newRook = &board[goalCoords[0]][goalCoords[1]+1];
                movePiece(oldRook, newRook);
            }
            else if (startCoords[1] - goalCoords[1] == -2) {
                cell *oldRook = &board[goalCoords[0]][RANK-1];
                cell *newRook = &board[goalCoords[0]][goalCoords[1]-1];
                movePiece(oldRook, newRook);
            }
        }
        updateState(state, goalSpace, flag);
        delete[] startCoords; delete[] goalCoords;
        return true;
    }
    delete[] startCoords; delete[] goalCoords;
    return false;
}


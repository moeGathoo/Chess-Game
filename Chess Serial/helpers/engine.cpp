#include "../headers.h"

/**
 * @brief Evaluates the board for the current side to play.
 * 
 * @param state 
 * @return int 
 */
int evaluation(state* state) {
    int rawScore = 0;
    if (black[4].size() == 0 && white[4].size() == 0) return 0;
    if (black[4].size() != 0 && white[4].size() == 0) rawScore = -1 * pieceValues[4];
    else if (black[4].size() == 0 && white[4].size() != 0) rawScore = pieceValues[4];
    else {
        int bScore = 0, wScore = 0;
        for (int i = 0; i < PIECES; i++) {
            if (i == 4) continue;
            wScore += pieceValues[i] * white[i].size();
            bScore += pieceValues[i] * black[i].size();
        }
        rawScore = wScore - bScore;
    }
    if (state->side == 'b') return -1 * (rawScore);
    return rawScore;
}


int miniMax(state currState, int depth, string* bestMove, bool first) {
    if (currState.gameOver || depth <= 0) {
        return evaluation(&currState);
    }

    int value = -10000000;
    vector<cell*>* pieces; vector<string> moves;
    if (currState.side == 'b') pieces = black;
    else pieces = white;
    for (int i = 0; i < PIECES; i++)
        if (pieces[i].size() != 0)
            for (cell* piece : pieces[i])
                pieceMoves(&currState, piece, i, &moves);

    for (string move : moves) {
        string start = move.substr(0,2);
        int* startCoords = new int[2]; toCoords(start, startCoords);
        string goal = move.substr(2,3);
        int* goalCoords = new int[2]; toCoords(goal, goalCoords);
        cell *startSpace = &board[startCoords[0]][startCoords[1]];
        cell *goalSpace = &board[goalCoords[0]][goalCoords[1]];
        delete[] startCoords; delete[] goalCoords;

        state nextState = currState;
        movePiece(startSpace, goalSpace);
        updateState(&nextState, goalSpace, false);
        int eval = (-1)*miniMax(nextState, depth-1, bestMove, false);
        movePiece(goalSpace, startSpace);
        resetBoard();
        addPieces(currState.fen);
        if (eval > value) {
            value = eval;
            if (first) *bestMove = move;
        }
        movePiece(goalSpace, startSpace);
    }
    return value;
}


int advEvaluation(state *state) {
    int material = evaluation(state);
    if (material == 10000 || material == -10000) return material;

    vector<string> blackMoves, whiteMoves;
    for (int i = 0; i < PIECES; i++) {
        if (black[i].size() != 0)
            for (cell* piece: black[i])
                pieceMoves(state, piece, i, &blackMoves);
        if (white[i].size() != 0)
            for (cell* piece: white[i])
                pieceMoves(state, piece, i, &whiteMoves);
    }
    int mobility = whiteMoves.size() - blackMoves.size();

    int blackScore = 0;
    for (string blackMove : blackMoves) {
        string goal = blackMove.substr(2,3);
        int* goalCoords = new int[2]; toCoords(goal, goalCoords);
        if (board[goalCoords[0]][goalCoords[1]].hasPiece) {
            blackScore++;
            if (board[goalCoords[0]][goalCoords[1]].piece == 'K')
                blackScore += 10;
        }
    }
    int whiteScore = 0;
    for (string whiteMove : whiteMoves) {
        string goal = whiteMove.substr(2,3);
        int* goalCoords = new int[2]; toCoords(goal, goalCoords);
        if (board[goalCoords[0]][goalCoords[1]].hasPiece) {
            whiteScore++;
            if (board[goalCoords[0]][goalCoords[1]].piece == 'k')
                whiteScore += 10;
        }
    }
    int attack = whiteScore - blackScore;

    int rawScore = 0;
    bool flag = true;
    for (int i = 0; i < PIECES; i++) {
        if (i != 5) {
            if (black[i].size() != 0 || white[i].size() != 0) {
                flag = false;
                break;
            }
        }
    }
    if (black[4].size() == 1 && white[4].size() == 1 && flag) rawScore = 0;
    else rawScore = material + mobility + attack;
    if (state->side == 'b') return (-1)*rawScore;
    return rawScore;
}


int alphaBeta(state currState, int depth, int alpha, int beta, string *bestMove, bool first) {
    if (currState.gameOver || depth <= 0) {
        return advEvaluation(&currState);
    }

    vector<cell*>* pieces; vector<string> moves;
    if (currState.side == 'b') pieces = black;
    else pieces = white;
    for (int i = 0; i < PIECES; i++)
        if (pieces[i].size() != 0)
            for (cell* piece : pieces[i])
                pieceMoves(&currState, piece, i, &moves);

    for (auto move : moves) {
        string start = move.substr(0,2);
        int* startCoords = new int[2]; toCoords(start, startCoords);
        string goal = move.substr(2,3);
        int* goalCoords = new int[2]; toCoords(goal, goalCoords);
        cell *startSpace = &board[startCoords[0]][startCoords[1]];
        cell *goalSpace = &board[goalCoords[0]][goalCoords[1]];
        delete[] startCoords; delete[] goalCoords;

        state nextState = currState;
        movePiece(startSpace, goalSpace);
        updateState(&nextState, goalSpace, false);
        int eval = (-1)*alphaBeta(nextState, depth-1, (-1)*beta, (-1)*alpha, bestMove, false);
        movePiece(goalSpace, startSpace);
        resetBoard();
        addPieces(currState.fen);
        if (eval >= beta){
            return beta;
        }
        if (eval > alpha){
            alpha = eval;
            if (first) *bestMove = move;
        }
    }

    return alpha;
}


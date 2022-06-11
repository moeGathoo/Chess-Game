/**
 * @file engine.cpp
 * @brief all functions pertaining to search and evaluation
 */
#include "../headers.h"

/**
 * @brief Evaluates the board for the current side to play.
 * 
 * @param state Pointer to state object containing current board information.
 * 
 * @return int - Score for evaluation of the current side to play's board.
 */
int evaluation(state* state) {
    int rawScore = 0;
    if (black[4].size() == 0 && white[4].size() == 0) return 0; //game over if no kings present
    //if one king is missing for one side, the other side wins, game over
    if (black[4].size() != 0 && white[4].size() == 0) rawScore = -1 * pieceValues[4];
    else if (black[4].size() == 0 && white[4].size() != 0) rawScore = pieceValues[4];
    else { //both kings present, game continues
        int bScore = 0, wScore = 0;
        //calculate score per side based on pieces present multiplied by their respective weightings
        for (int i = 0; i < PIECES; i++) {
            if (i == 4) continue; //don't include king in evaluation
            wScore += pieceValues[i] * white[i].size();
            bScore += pieceValues[i] * black[i].size();
        }
        rawScore = wScore - bScore; //calculate raw score
    }
    //raw score based on side playing
    //returns rawScore if white is playing, else returns rawScore * (-1)
    if (state->side == 'b') return -1 * (rawScore);
    return rawScore;
}

/**
 * @brief Performs minimax search algorithm for current side playing.
 * Searches all possible possible for current side to take (all possible moves to make)
 * in order to maximize move score while minimizing enemy move score.
 * 
 * @param currState State object containing current state of the board.
 * @param depth The depth to which the algorithm must search.
 * @param bestMove Pointer to a string containing the best move to make.
 * @param first Boolean indicating whether we are at root node of the search tree (true when called the first time, false otherwise)
 * 
 * @return int - Returns the maximum score out of all possible paths to take. The best recorded move based
 * off the returned score is stored in the bestMove variable.
 */
int miniMax(state currState, int depth, string* bestMove, bool first) {
    //stop searching if game state is over or depth of 0 is reached
    if (currState.gameOver || depth <= 0) {
        return evaluation(&currState);
    }

    int value = -10000000; //any large value to store maximum move score
    //determine which side is playing
    //generate all possible moves available for said side's current board
    vector<cell*>* pieces; vector<string> moves;
    if (currState.side == 'b') pieces = black;
    else pieces = white;
    for (int i = 0; i < PIECES; i++)
        if (pieces[i].size() != 0)
            for (cell* piece : pieces[i])
                pieceMoves(&currState, piece, i, &moves);

    //execute each move generated
    for (string move : moves) {
        //get moves start and goal spaces to move pieces based on positions in 'move' string
        string start = move.substr(0,2);
        int* startCoords = new int[2]; toCoords(start, startCoords);
        string goal = move.substr(2,3);
        int* goalCoords = new int[2]; toCoords(goal, goalCoords);
        cell *startSpace = &board[startCoords[0]][startCoords[1]];
        cell *goalSpace = &board[goalCoords[0]][goalCoords[1]];
        delete[] startCoords; delete[] goalCoords;

        state nextState = currState; //state after move is made
        movePiece(startSpace, goalSpace); //move piece from start space to goal space
        updateState(&nextState, goalSpace, (goalSpace->piece == 'p' || goalSpace->piece == 'P')); //update next state indicating move has been made
        int eval = (-1)*miniMax(nextState, depth-1, bestMove, false); //evaulate state of board after move has been made
        //undo move to restore current state's board
        movePiece(goalSpace, startSpace);
        resetBoard();
        addPieces(currState.fen);
        //determine if score returned is greater than current score, if so and at root node, set bestMove
        if (eval > value) {
            value = eval;
            if (first) *bestMove = move;
        }
    }
    return value;
}

/**
 * @brief Advanced evaulation function calculated from
 * all pieces on the board, possible moves available and attacking pieces.
 * 
 * @param state Pointer to state object with current board's information.
 * 
 * @return int - Advanced score based on criteria mentioned above.
 */
int advEvaluation(state *state) {
    //material score: weightings of pieces on board
    int material = evaluation(state);
    if (material == 10000 || material == -10000) return material;

    //mobility score: number of moves each side has to play
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

    //attack score: number of opposition pieces a player is threatening to attack
    int* goalCoords = new int[2];
    int blackScore = 0;
    for (string blackMove : blackMoves) {
        string goal = blackMove.substr(2,3);
        toCoords(goal, goalCoords);
        if (board[goalCoords[0]][goalCoords[1]].hasPiece) {
            blackScore++;
            //bonus points if piece being threatened is oppositions king
            if (board[goalCoords[0]][goalCoords[1]].piece == 'K')
                blackScore += 10;
        }
    }
    int whiteScore = 0;
    for (string whiteMove : whiteMoves) {
        string goal = whiteMove.substr(2,3);
        toCoords(goal, goalCoords);
        if (board[goalCoords[0]][goalCoords[1]].hasPiece) {
            whiteScore++;
            if (board[goalCoords[0]][goalCoords[1]].piece == 'k')
                whiteScore += 10;
        }
    }
    delete[] goalCoords;
    int attack = whiteScore - blackScore;

    //overall score
    int rawScore = 0;
    bool flag = true;
    for (int i = 0; i < PIECES; i++) {
        //if only king pieces are on the board, rawScore is 0, game over
        if (i != 5) { 
            if (black[i].size() != 0 || white[i].size() != 0) {
                flag = false;
                break;
            }
        }
    }
    //if one king is missing, the other side wins, game over
    if (black[4].size() == 1 && white[4].size() == 1 && flag) rawScore = 0;
    else rawScore = material + mobility + attack;
    //score from white's perpective, for black: rawScore * (-1)
    if (state->side == 'b') return (-1)*rawScore;
    return rawScore;
}

/**
 * @brief Alpha-beta pruning algorithms. Works the same way as minimax algorithm,
 * but faster as it cuts off portions from the search tree to search if a threshold score
 * is not met.
 * 
 * @param currState State object containing current board information.
 * @param depth Depth to which the algorithm must search.
 * @param alpha Alpha score value.
 * @param beta Beta score value.
 * @param bestMove Pointer to string that will store best move.
 * @param first Boolean indicating whether or not we are at root node (true when first called, false otherwise)
 * 
 * @return int - Score after having search tree. The best move to make based on the score is stored
 * in the bestMove string.
 */
int alphaBeta(state currState, int depth, int alpha, int beta, string *bestMove, bool first) {
    //stop searching if game is over or max depth reached
    //return advanvced evaluation of board
    if (currState.gameOver || depth <= 0) {
        return advEvaluation(&currState);
    }

    //determine which side is playing and generate all moves possible to play
    vector<cell*>* pieces; vector<string> moves;
    if (currState.side == 'b') pieces = black;
    else pieces = white;
    for (int i = 0; i < PIECES; i++)
        if (pieces[i].size() != 0)
            for (cell* piece : pieces[i])
                pieceMoves(&currState, piece, i, &moves);

    //execute each move generated
    for (auto move : moves) {
        //get moves start and goal spaces to move pieces based on positions in 'move' string
        string start = move.substr(0,2);
        int* startCoords = new int[2]; toCoords(start, startCoords);
        string goal = move.substr(2,3);
        int* goalCoords = new int[2]; toCoords(goal, goalCoords);
        cell *startSpace = &board[startCoords[0]][startCoords[1]];
        cell *goalSpace = &board[goalCoords[0]][goalCoords[1]];
        delete[] startCoords; delete[] goalCoords;

        state nextState = currState; //next state which will store board information after making move
        movePiece(startSpace, goalSpace); //move piece
        updateState(&nextState, goalSpace, false); //updates next state indicating move
        int eval = (-1)*alphaBeta(nextState, depth-1, (-1)*beta, (-1)*alpha, bestMove, false); //calculate score of baord after move has been made
        //undo move to restore board's original state
        resetBoard();
        addPieces(currState.fen);
        //stop searching tree if evaluation is greater than beta threshold
        if (eval >= beta){
            return beta;
        }
        //assign current best move if at root node
        if (eval > alpha){
            alpha = eval;
            if (first) *bestMove = move;
        }
    }

    return alpha;
}


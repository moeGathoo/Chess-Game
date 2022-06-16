#include "../headers.h"

int parallelAlphaBeta(state currState, int size, int rank, int depth, int alpha, int beta, string *bestMove, bool first) {
    vector<string> myMoves; int pieceCount = 0;
    int flags[NUM_PROCS] = {0}; vector<string> moves;
    for (int r = 0; r < SIZE; r++) {
        cell* space = &board[r][rank];
        if (space->colour == currState.side) {
            char piece = space->piece; if (currState.side == 'w') piece = tolower(piece);
            char *foo = find(begin(pieceTypes), end(pieceTypes), piece); 
            if (foo != end(pieceTypes)) { //lowercas piece type found, therefore black piece type
                int idx = distance(pieceTypes, foo); //index of piece type
                pieceMoves(&currState, space, idx, &myMoves);
            }
            pieceCount++;
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);

    for (string move : myMoves) {
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

    // int flag = 0; if (pieceCount == 0) flag = 1;
    // vector<const char*> thingy;
    // for (int i = 0; i < myMoves.size(); i++) {
    //     thingy.push_back((const char*)myMoves[i].c_str());

    // MPI_Gather(&flag, 1, MPI_INT, &flags, 1, MPI_INT, 0, MPI_COMM_WORLD);
    // if (rank == 0)
    //     for (int i = 0; i < NUM_PROCS; i++)
    //         if (flags[i] == 1) {
    //             for (int i = 1; i < )
    //             break;
    //         }

    // int buff;
    // if (rank == 0) buff = 77;
    // cout << "Before bcast (rank " << rank << "): " << buff << endl;
    // MPI_Bcast(&buff, 1, MPI_INT, 0, MPI_COMM_WORLD);
    // cout << "After bcast (rank " << rank << "): " << buff << endl;
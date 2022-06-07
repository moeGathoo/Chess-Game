/**
 * @file state.cpp
 * @brief all function definitons pertaining to state of board
 */
#include "../headers.h"

/**
 * @brief Initializes a state based on current baord based on fen information.
 * 
 * @param fen Fen string representation of board.
 * @param side Side to play.
 * @param castle Castling rights for each side.
 * @param enPassant En passant move available.
 * @param halfMove Total moves since pawn move or piece capture.
 * @param fullMove Total number of moves incremented after black moves.
 * 
 * @return state State object with current board information.
 */
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

/**
 * @brief Converts position of space to corresponding row and column in 2D array.
 * 
 * @param pos String position to convert e.g. e5
 * @param coords Pointer to integer array created to store row and column.
 * 
 * @return Nothing. The converted values are stored in the array,
 * e.g. e5 corresponds to row 3, column 4.
 */
void toCoords(string pos, int* coords) {
    int row = RANK - (pos[1] - '0');    //convert rank number to int   
    int col = pos[0] - FILE;            //convert file letter to int
    coords[0] = row; coords[1] = col;
}

/**
 * @brief Converts row and column numbers to position on board.
 * 
 * @param col Column number.
 * @param row Row number.
 * @param vec Pointer to vector storing string positions.
 * 
 * @return Nothing. Once converted, the position string is added to the vector.
 */
void toPos(int col, int row, vector<string>* vec) {
    char c = col;
    string pos = string() + c + to_string(row);
    vec->push_back(pos);
}

/**
 * @brief Get the positions of all pieces on the board.
 * 
 * @param None
 * 
 * @return Nothing. The respective vectors storing the different piece
 * type's positions are updated with piece positions
 */
void getPositions() {
    //clear all piece type's vectors
    for (auto& v : black) v.clear();
    for (auto& v : white) v.clear();
    
    for (int i = 0; i < RANK; i++) {
        for (int j = 0; j < RANK; j++) {
            if (board[i][j].piece != '-') { //if board space does not have piece
                //find index of piece in pieceTypes array to determine which vector to update
                char *foo = find(begin(pieceTypes), end(pieceTypes), board[i][j].piece); 
                if (foo != end(pieceTypes)) { //lowercas piece type found, therefore black piece type
                    int idx = distance(pieceTypes, foo); //index of piece type
                    black[idx].push_back(&board[i][j]);  //update vector with pointer to space containing piece
                }
                else {
                    //lowercase piece not found, therefore white piece type
                    foo = find(begin(pieceTypes), end(pieceTypes), tolower(board[i][j].piece));
                    int idx = distance(pieceTypes, foo);
                    white[idx].push_back(&board[i][j]);
                }
            }
        }
    }
}

/**
 * @brief Update the current state of the board, usually after moving a piece.
 * 
 * @param state Pointer to state object to be updated
 * @param piece Pointer to space of piece that was moved.
 * @param flag Boolean indicating whether a pawn was moved or piece was captured.
 * 
 * @return Nothing. State object is updated.
 */
void updateState(state* state, cell* piece, bool flag) {
    //create new fen string representing board by traversing in row-major order
    string newFen = "";
    for (int i = 0; i < RANK; i++) {
        int empty = 0; //number of consecutive empty spaces
        for (int j = 0; j < RANK; j++) {
            cell space = board[i][j];
            if (space.piece != '-') { //piece found
                if (empty != 0) newFen += to_string(empty); //add empty spaces counted
                newFen += space.piece; //add piece to fen string
                empty = 0; //reset empty space counter
            }
            else empty++; //no piece found, increment empty spaces found
        }
        if (empty != 0) newFen += to_string(empty); //add empty spaces number at end of rank
        if (i != RANK-1) newFen += "/"; //if not in the last rank, add '/' to indicate end of rank in fen string
    }
    state->fen = newFen; //assign new fen string generated
    
    //castling piece types
    char k = 'k', q = 'q', r = 'r';
    if (piece->colour == 'w') {k = toupper(k); q = toupper(q); r = toupper(r);}
    
    if (piece->piece == k) { //if king moves, remove all castling rights for that side
        state->castle.erase(remove(state->castle.begin(), state->castle.end(), q), state->castle.end());
        state->castle.erase(remove(state->castle.begin(), state->castle.end(), k), state->castle.end());
    }
    else if (piece->piece == r) { //rook moves
        //find which side rook is and remove relevant castling rights flag
        int* coords = new int[2];
        toCoords(piece->position, coords);
        if (coords[1] < 5) state->castle.erase(remove(state->castle.begin(), state->castle.end(), q), state->castle.end());
        else state->castle.erase(remove(state->castle.begin(), state->castle.end(), k), state->castle.end());
    }
    if (state->castle.length() == 0) state->castle = "-";
    
    if (state->side == 'w') state->side = 'b'; //switch side to play
    else {
        state->side = 'w';
        state->fullMove++; //if black played, increment full moves
    }

    if (flag) state->halfMove = 0; //if pawn moved or piece captured, reset half moves
    else state->halfMove++; //else increment

    //if at least one king is missing, game is over
    if (white[4].size() == 0 || black[4].size() == 0) state->gameOver = true;

}

/**
 * @brief Prints out attributes of current state of board.
 * 
 * @param currState Pointer to state object with current state information.
 * 
 * @return Nothing. Current state information is printed to terminal.
 */
void printState(state* currState) {
    cout << currState->fen << " ";
    cout << currState->side << " ";
    cout << currState->castle << " ";
    cout << currState->enPassant << " ";
    cout << currState->halfMove << " ";
    cout << currState->fullMove << endl;
}
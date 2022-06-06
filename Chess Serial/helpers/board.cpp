#include "../headers.h"


void addPieces(string fen) {
    int row = 0, col = 0;
    for (char c : fen) {
        if (c == '/') {row++; col = 0; continue;}
        if (!isdigit(c)) {
            board[row][col].hasPiece = true;
            board[row][col].piece = c;
            if (islower(c)) board[row][col].colour = 'b';
            else board[row][col].colour = 'w';
            col++;
        }
        else {
            int skip = atoi(&c);
            col+=skip;
        }
    }
    getPositions();
}


void initBoard(string fen) {
    for (int i = RANK-1; i >= 0; i--){
        for (int j = FILE; j < FILE + RANK; j++){
            char c = j;
            string pos = string() + c + to_string(RANK-i);
            board[i][j-FILE].position = pos;
        }
    }
    addPieces(fen);
}


void resetBoard(){
    for (int i = 0; i < 7; i++){
        for (int j = 0; j < 7; j++){
            cell *space = &board[i][j];
            space->hasPiece = false;
            space->piece = '-';
            space->colour = '-';
        }
    }
}


void checkSpace(struct cell *piece, int row, int col, vector<string>* moves) {
    if (row >=0 && row < RANK && col >= 0 && col < RANK){ 
        struct cell space = board[row][col];
        if (!space.hasPiece){ 
            moves->push_back(piece->position + space.position); 
        }
        else{
            if (piece->colour != space.colour){ 
                moves->push_back(piece->position + space.position); 
            }
        }
    }
}


void checkSpaceP(struct cell *piece, int row, int col, vector<string>* moves) {
    if (row >=0 && row < RANK && col >= 0 && col < RANK){
        struct cell space = board[row][col];
        if (piece->colour != space.colour){
            moves->push_back(piece->position + space.position);
        }
    }
}


void checkSpacesK(struct cell *king, vector<string>* moves) {
    int coords[2] = {0, 0};
    toCoords(king->position, coords);

    
    checkSpace(king, coords[0]+1, coords[1], moves);
    checkSpace(king, coords[0]-1, coords[1], moves);
    checkSpace(king, coords[0], coords[1]+1, moves);
    checkSpace(king, coords[0], coords[1]-1, moves);
    checkSpace(king, coords[0]+1, coords[1]+1, moves);
    checkSpace(king, coords[0]-1, coords[1]-1, moves);
    checkSpace(king, coords[0]+1, coords[1]-1, moves);
    checkSpace(king, coords[0]-1, coords[1]+1, moves);

    

    sort(moves->begin(), moves->end());
}


void castle(state* state, cell* king, vector<string> *moves) {
    if (state->castle == "-") return;
    bool kSide = false, qSide = false;
    if (king->colour == 'b') {
        if (state->castle.find('k') < 3) kSide = true;
        if (state->castle.find('q') < 3) qSide = true;
    }
    else {
        if (state->castle.find('K') < 3) kSide = true;
        if (state->castle.find('Q') < 3) qSide = true;
    }

    int* coords = new int[2];
    toCoords(king->position, coords);
    if (kSide) {
        if (!board[coords[0]][coords[1]+1].hasPiece &&
            !board[coords[0]][coords[1]+2].hasPiece)
                moves->push_back(king->position + board[coords[0]][coords[1]+2].position);
    }
    if (qSide) {
        if (!board[coords[0]][coords[1]-1].hasPiece &&
            !board[coords[0]][coords[1]-2].hasPiece &&
            !board[coords[0]][coords[1]-3].hasPiece)
                moves->push_back(king->position + board[coords[0]][coords[1]-2].position);
    }
    delete[] coords;
}


void movePiece(cell* startSpace, cell* goalSpace) {
    goalSpace->piece = startSpace->piece; startSpace->piece = '-';
    goalSpace->colour = startSpace->colour; startSpace->colour = '-';
    goalSpace->hasPiece = true; startSpace->hasPiece = false;
    getPositions();
}


void printBoard() {
    for (int i = 0; i < RANK; i++) {
        for (int j = 0; j < RANK; j++)
            cout << board[i][j].piece << "\t";
        cout << endl;
    }
}


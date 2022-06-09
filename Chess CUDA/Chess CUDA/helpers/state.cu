#include "../headers.cuh"

void getPositions() {
	for (int i = 0; i < PIECES; i++) {
		vectorClear(&black[i]);
		vectorClear(&white[i]);
	}

    for (int i = 0; i < RANK; i++) {
        for (int j = 0; j < RANK; j++) {
            if (board[i][j].piece != '-') { //if board space does not have piece
                //find index of piece in pieceTypes array to determine which vector to update
                bool found = false; int idx = -1;
                for (int p = 0; p < PIECES; p++) {
                    if (board[i][j].piece == pieceTypes[p]) {
                        found = true;
                        idx = p;
                        break;
                    }
                }
                if (found)
                    //lowercase piece not found, therefore white piece type
                    vectorPushBack(&black[idx], &board[i][j]);
                else {
                    for (int p = 0; p < PIECES; p++) {
                        if (board[i][j].piece == toupper(pieceTypes[p])) {
                            found = true;
                            idx = p;
                            break;
                        }
                    }
                    vectorPushBack(&white[idx], &board[i][j]);
                }
            }
        }
    }
}
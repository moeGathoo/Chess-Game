#include "../headers.cuh"

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
state initState(char* fen, char side, char* castle, char* enPassant, int halfMove, int fullMove) {
    struct state curr;
    strcpy(curr.fen, fen);
    curr.side = side;
    strcpy(curr.castle, castle);
    curr.enPassant = enPassant;
    curr.halfMove = halfMove;
    curr.fullMove = fullMove;
    return curr;
}
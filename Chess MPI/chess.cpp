/**
 * @brief Plays a game of chess for 10 full moves
 * compile command:
 * * g++ chess.cpp helpers/state.cpp helpers/board.cpp helpers/moves.cpp helpers/engine.cpp -o chess
 * execute command:
 * * ./chess rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1
 */

#include "headers.h"

//initializes values of all global variables
char pieceTypes[6] = {'r', 'n', 'b', 'q', 'k', 'p'};    //types of pieces on board
int pieceValues[6] = {50, 30, 30, 90, 900, 10};         //piece weightings for search and evaluation algorithms
cell board[RANK][RANK];                                 //board to play on
vector<cell*> black[PIECES];                            //array of vectors for each piece type (pointers to spaces on baord that have the piece)
vector<cell*> white[PIECES];                            //array of vectors for each piece type (pointers to spaces on baord that have the piece)

/**
 * @brief Main function to play game.
 * 
 * @param argc Number of arguments.
 * @param argv Each argument pertains to a particular part of fen notation.
 * ? argv[0]: fen string representation of board
 * ? argv[1]: side to play
 * ? argv[2]: castling rights
 * ? argv[3]: en passant move
 * ? argv[4]: half moves (total moves since pawn move or piece capture)
 * ? argv[5]: full moves (increments by 1 after black moves)
 * 
 * @return int 
 */
int main(int argc, char* argv[]) {
    string input[] = {"rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR",
                      "w",
                      "KQkq",
                      "-",
                      "0",
                      "1"};
    state game = initState(input[0], input[1][0], input[2], input[3], stoi(input[4]), stoi(input[5]));
    //set up board
    initBoard(game.fen);
    int rank, size;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    string move; char bestMove[4]; int scores[NUM_PROCS];

    double start, end;
    start = MPI_Wtime();
    int score = parallelAlphaBeta(game, size, rank, 4, -10001, 10001, &move, true);
    MPI_Barrier(MPI_COMM_WORLD);
    end = MPI_Wtime();
    if (rank == 0)
        cout << "Parallel implementation time: " << end-start << endl;

    int max = -10001;
    MPI_Gather(&score, 1, MPI_INT, &scores, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (rank == 0)
        for (int i = 0; i < NUM_PROCS; i++)
            if (scores[i] > max) max = scores[i];

    MPI_Bcast(&max, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (max == score) {
        strncpy(bestMove, (const char*)move.c_str(), 4);
        for (int i = 0; i < NUM_PROCS; i++) {
            if (i == rank) continue;
            MPI_Send(move.c_str(), move.size(), MPI_CHAR, i, 0, MPI_COMM_WORLD);
        }
    }
    else MPI_Recv(bestMove, 4, MPI_CHAR, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    //string play(bestMove);
    //checkMove(&game, &play);
    //printBoard(); cout << endl;
    cout << bestMove << endl;

    MPI_Finalize();

    //plays game up to 10 full moves using alpha-beta pruning algorithm
    //prints move made
    // for (int i = 0; i < 10; i++) {
    //     string move;
    //     alphaBeta(game, 4, -10001, 10001, &move, true);
    //     checkMove(&game, &move);
    //     cout << move << endl;
    //     if (game.gameOver) break; //end game of game is over
    // }

    return 0;
}
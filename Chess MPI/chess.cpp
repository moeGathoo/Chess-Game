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

    //initialize MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    string move; char bestMove[4]; int scores[NUM_PROCS];

    for (int i = 0; i < 10; i++){
        double start, end;
        //have each processor perform alpha beta on their columns pieces
        //time how long they collectively take to return a move 
        start = MPI_Wtime();
        int score = parallelAlphaBeta(game, size, rank, 4, -10001, 10001, &move, true);
        MPI_Barrier(MPI_COMM_WORLD);
        end = MPI_Wtime();
        if (rank == 0)
            cout << "Parallel implementation time: " << end-start << endl;

        //gather evaluated scores from every processor to master processor
        int max = -10001;
        MPI_Gather(&score, 1, MPI_INT, &scores, 1, MPI_INT, 0, MPI_COMM_WORLD);
        if (rank == 0)
            for (int i = 0; i < NUM_PROCS; i++) //get maximum score
                if (scores[i] > max) max = scores[i];

        MPI_Bcast(&max, 1, MPI_INT, 0, MPI_COMM_WORLD); //broadcast maximum score to every processor
        if (max == score) { //find processor with highest score
            strncpy(bestMove, (const char*)move.c_str(), 4);
            for (int i = 0; i < NUM_PROCS; i++) {
                if (i == rank) continue;
                MPI_Send(move.c_str(), move.size(), MPI_CHAR, i, 0, MPI_COMM_WORLD); //send said processors evaluated move to every other processor
            }
        }
        else MPI_Recv(bestMove, 4, MPI_CHAR, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        move = bestMove;
        checkMove(&game, &move); //perform move
        //if (rank == 0) cout << move << endl;
    }

    MPI_Finalize();

    return 0;
}
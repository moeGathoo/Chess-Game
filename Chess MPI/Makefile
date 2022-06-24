CC = mpic++
TARGET = chess
HELPER3 = helpers/moves.cpp
HELPER2 = helpers/board.cpp
HELPER1 = helpers/state.cpp
HELPER4 = helpers/engine.cpp

$(TARGET): $(TARGET).cpp
	$(CC) -o $(TARGET) $(TARGET).cpp $(HELPER1) $(HELPER2) $(HELPER3) $(HELPER4)

clean:
	rm chess

run:
	mpirun -n 8 ./chess
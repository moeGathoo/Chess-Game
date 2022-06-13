CC = g++
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
	./chess rnbqkbnr/pppppppp/8/8/R3P3/2NB3N/PPP2PPP/2BQK2R w Kkq - 0 1
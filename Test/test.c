#include "vector.c"

#define FILE 'a'

int main(void) {
    char *fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR";
    for (int i = 0; i < strlen(fen); i++) {
        char c = fen[i];
        printf("%c\n", c);
    }
}
#include "vector.c"

#define FILE 'a'

int main(void) {
    char fen[] = "KQkq";
    char k = 'q';
    //char* posPtr = strchr(fen, k);
    int idxToDel = strchr(fen, k) - fen;
    //memmove(&fen[idxToDel], &fen[idxToDel + 1], strlen(fen) - idxToDel );

    char *newCastle = "KQk";
    strcpy(fen, newCastle);
    printf("%s\n", fen);
    printf("%c\n", fen[10]);
}
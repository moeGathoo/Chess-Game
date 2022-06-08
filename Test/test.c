#include "vector.c"

int main(void) {
    vector v;

    initVector(&v);
    vectorPushBack(&v, (void*)1);
    vectorPushBack(&v, (void*)2);
    vectorPushBack(&v, (void*)3);
    vectorPushBack(&v, (void*)4);
    vectorPushBack(&v, (void*)5);

    for (int i = 0; i < v.size; i++) {
        printf("%d\n", v.items[i]);
    }
    printf("\n");

    v.items[0] = (void*)10;
    vectorRemove(&v, 2);

    for (int i = 0; i < v.size; i++) {
        printf("%d\n", v.items[i]);
    }

    
    return 0;
}
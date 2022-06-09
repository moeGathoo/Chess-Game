#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#define VECTOR_INIT_CAPACITY 4

typedef struct vector {
    void **items;
    int capacity;
    int size;
} vector;

void initVector(vector *v) {
    v->capacity = VECTOR_INIT_CAPACITY;
    v->size = 0;
    v->items = malloc(sizeof(void *) * v->capacity);
}

int vectorSize(vector *v) {
    return v->size;
}

static void vectorResize(vector *v, int capacity) {
    void** items = realloc(v->items, sizeof(void*)*capacity);
    if (items) {
        v->items = items;
        v->capacity = capacity;
    }
}

void vectorPushBack(vector *v, void *item) {
    if (v->capacity == v->size)
        vectorResize(v, v->capacity*2);
    v->items[v->size++] = item;
}

void vectorSet(vector *v, int index, void* item) {
    if (index >= 0 && index < v->size)
        v->items[index] = item;
}

void *vectorGet(vector *v, int index) {
    if (index >= 0 && index < v->size)
        return v->items[index];
    return NULL;
}

void vectorRemove(vector *v, int index) {
    if (index < 0 || index >= v->size)
        return;
    
    v->items[index] = NULL;
    for (int i = index; i < v->size; i++) {
        v->items[i] = v->items[i+1];
    }
    v->items[v->size] = NULL;
    v->size--;
    if (v->size > 0 && v->size == v->capacity / 4)
        vectorResize(v, v->capacity / 2);
}

void vectorFree(vector *v) {
    free(v->items);
}
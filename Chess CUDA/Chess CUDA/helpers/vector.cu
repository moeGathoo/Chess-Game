#include "../headers.cuh"

#define VECTOR_INIT_CAPACITY 4

static int myCompare(const void* a, const void* b) {
    return strcmp(*(const char**)a, *(const char**)b);
}

void initVector(vector* v) {
    v->capacity = VECTOR_INIT_CAPACITY;
    v->size = 0;
    v->items = (void**)malloc(sizeof(void*) * v->capacity);
}

int vectorSize(vector* v) {
    return v->size;
}

static void vectorResize(vector* v, int capacity) {
    void** items = (void**)realloc(v->items, sizeof(void**) * capacity);
    if (items) {
        v->items = items;
        v->capacity = capacity;
    }
}

void vectorPushBack(vector* v, void* item) {
    if (v->capacity == v->size)
        vectorResize(v, v->capacity * 2);
    v->items[v->size++] = item;
}

void vectorSet(vector* v, int index, void* item) {
    if (index >= 0 && index < v->size)
        v->items[index] = item;
}

void* vectorGet(vector* v, int index) {
    if (index >= 0 && index < v->size)
        return v->items[index];
    return NULL;
}

void vectorSort(vector* v) {
    qsort(v->items, v->size, sizeof(const char*), myCompare);
}

bool vectorSearch(vector* v, char* str) {
    int len = (sizeof(void*)*v->size) / sizeof(v->items[0]);
    for (int i = 0; i < len; i++)
        if (!strcmp((const char*)v->items[i], str))
            return true;
    return false;
}

void vectorRemove(vector* v, int index) {
    if (index < 0 || index >= v->size)
        return;

    v->items[index] = NULL;
    for (int i = index; i < v->size; i++) {
        v->items[i] = v->items[i + 1];
    }
    v->items[v->size] = NULL;
    v->size--;
    if (v->size > 0 && v->size == v->capacity / 4)
        vectorResize(v, v->capacity / 2);
}

void vectorClear() {
    for(int i = 0; i < PIECES; i++) {
        vectorFree(&black[i]);
        vectorFree(&white[i]);
        initVector(&black[i]);
        initVector(&white[i]);
    }
}

void vectorFree(vector* v) {
    free(v->items);
}
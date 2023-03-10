#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int N = 10; 
int M = 5; 
int dificultad; 
int modo; 

__device__ bool pertenece(int* x, int n, int y) {
    bool p = false;
    for (int i = 0; i < n; ++i) {
        p = p || x[i] == y;
    }
    return p;
}

__device__ void buscar_camino(char* tablero, int inicio, int fin, int* visitados, int* x, int* camino, int* y, int N, int M) {
    if (inicio != fin) {
        //Encima, debajo, izq, dcha ||||| Vecino = -1 --> fuera del tablero
        int vecinos[5] = { inicio, inicio - M, inicio + M, inicio - 1, inicio + 1 };
        if (vecinos[1] < 0) {
            vecinos[1] = -1;
        }
        if (vecinos[2] >= N * M) {
            vecinos[2] = -1;
        }
        if (inicio % M == 0) {
            vecinos[3] = -1;
        }
        if ((inicio + 1) % M == 0) {
            vecinos[4] = -1;
        }

        for (int i = 0; i < 5; ++i) {
            if (!pertenece(visitados, *x, vecinos[i])) {
                if (vecinos[i] != -1) {
                    //Se marca como explorado

                    visitados[*x] = vecinos[i];
                    (*x)++;

                    if (tablero[inicio] == tablero[vecinos[i]]) {
                        //En caso de que el vecino sea del mismo tipo, sigo el camino

                        camino[*y] = vecinos[i];
                        (*y)++;
                        buscar_camino(tablero, vecinos[i], fin, visitados, x, camino, y, N, M);
                    }
                }
            }
        }
    }
}

__global__ void encontrar_caminos(char* tablero, int N, int M, int fila, int columna) {
    int selec = fila * M + columna; 
    int id = threadIdx.y * N + threadIdx.x; 

    //Funcion que busque camino
    int* camino = (int*)malloc(N * M * sizeof(int)); 
    int* visitados = (int*)malloc(N * M * sizeof(int));
    int x = 0; 
    int y = 0; 

    for (int i = 0; i < N * M; ++i) {
        camino[i] = -1; 
        visitados[i] = -1; 
    }

    if (tablero[selec] == tablero[id]) {
        buscar_camino(tablero, id, selec, visitados, &x, camino, &y, N, M);
    }
    
    free(visitados); 
}

void cargar_argumentos(int argc, char* argv[]) {
    if (argc != 5) {
        perror("Se esperaban argumentos -a/-m 1/2 n m. "); 
        exit(-1); 
    }
    else {
        int error = 0; 
        if (!strcmp(argv[1], "-a")) {
            modo = 0; 
        }
        else if (!strcmp(argv[1], "-m")){
            modo = 1; 
        }
        else {
            error = 1; 
        }

        if (!strcmp(argv[2], "1")) {
            dificultad = 0; 
        }
        else if (!strcmp(argv[2], "2")) {
            dificultad = 1; 
        }
        else {
            error = 1; 
        }

        if (atoi(argv[3]) < 1 || atoi(argv[4]) < 1) {
            error = 1; 
        }
        else {
            N = atoi(argv[3]); 
            M = atoi(argv[4]); 
        }

        if (error) {
            perror("Valor de argumento invalido. ");
            exit(-1);
        }
    }
}

char generar_elemento() {
    if (dificultad) {
        return  (rand() % (6 - 1 + 1) + 1) + '0';
    }
    else {
        return  (rand() % (4 - 1 + 1) + 1) + '0';
    }
    
}

void mostrar_tablero(char* tablero, int n, int m) {
    printf("\nTABLERO: \n"); 
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            printf("%2c", tablero[m * i + j]); 
        }
        printf("\n"); 
    }
}


int main(int argc, char* argv[]){

    //cargar_argumentos(argc, argv); 
    int tam_tablero = sizeof(char) * N * M; 
    char* tablero = (char*)malloc(tam_tablero);

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < M; j++) {
            tablero[M * i + j] = generar_elemento();
        }
    }

    mostrar_tablero(tablero, N, M); 

    char* d_tablero;
    cudaMalloc((void**)&d_tablero, sizeof(char) * N * M); 
    cudaMemcpy(d_tablero, tablero, sizeof(char) * N * M, cudaMemcpyHostToDevice); 

    dim3 bloque(N, M); 
    encontrar_caminos<<<1, bloque>>>(d_tablero, N, M, 2, 3);
    
    free(tablero); 

    return 0;
}

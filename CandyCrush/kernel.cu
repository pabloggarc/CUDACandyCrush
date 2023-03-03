#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int N; 
int M; 
int dificultad; 
int modo; 

__global__ void buscar_camino(char* tablero, int fila, int columna) {
    int selec = fila * N + columna; 
    int id = threadIdx.y * N + threadIdx.x; 
    int vecinos[4] = {id - N, id + N, id - 1, id + 1}; //Encima, debajo, izq, dcha
    
    if (vecinos[0] < 0) {
        vecinos[0] = -1; //fuera
    }
    if (vecinos[1] < N * M - 1) {
        vecinos[1] = -1; 
    }
    if(vecinos[2])
    

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

    cargar_argumentos(argc, argv); 
    char* tablero = (char*)malloc(sizeof(char) * N * M);

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < M; j++) {
            tablero[M * i + j] = generar_elemento();
        }
    }

    mostrar_tablero(tablero, N, M); 

    return 0;
}

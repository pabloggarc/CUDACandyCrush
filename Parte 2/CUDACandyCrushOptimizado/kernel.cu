#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <device_functions.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <curand_kernel.h>
#include <time.h>
#include <math.h>

//Variables generales del juego

int N;
int M;
int vidas = 5;
int modo;
int dificultad;

//Funciones auxiliares (DEVICE)

/*
   a y b: límites del intervalo del que se quiere obtener un número aleatorio
   Salida R perteneciente a [a, b]
*/

__device__ int aleatorio(int a, int b) {
    curandState state;
    curand_init(clock64(), threadIdx.y * blockDim.x + threadIdx.x, 0, &state);
    return curand(&state) % (b - a + 1) + a;
}

/*
    x: vector de enteros
    n: longitud del vector
    y: elemento a encontrar

    Salida: determina si y pertenece a x

*/

__host__ __device__ bool pertenece(int* x, int n, int y) {
    bool p = false;
    for (int i = 0; i < n; ++i) {
        p = p || x[i] == y;
    }
    return p;
}

/*
    tablero: tablero del juego
    inicio: casilla en desde la que se empieza a buscar
    fin: casilla a la que se quiere llegar
    visitados: casillas por las que se ha intentado ir
    x: longitud de visitados
    camino: casillas que forman parte del camino
    y: longitud de camino
    N y M: dimensiones del tablero

    Salida: camino que lleva de inicio a fin (mediante el puntero a camino)

*/

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

/*
    tablero: tablero del juego
    N y M: dimensiones del tablero
    fila y columna: del elemento tocado

    Salida: tablero con las posiciones a borrar sustituidas por 'X' (mediante el puntero a tablero)

*/

__global__ void encontrar_caminos(char* tablero, char* new_tablero, int selec, int fila, int col, int N, int M) {
    int fila2 = blockIdx.y * blockDim.y + threadIdx.y;
    int col2 = blockIdx.x * blockDim.x + threadIdx.x;

    int id = fila2 * M + col2;

    //verificar que no sale del tablero
    if (fila2 < N && col2 < M) {

        //Funcion que busque camino

        int* camino = (int*)malloc(N * M * sizeof(int));
        int* visitados = (int*)malloc(N * M * sizeof(int));

        int x = 1;
        int y = 1;


        for (int i = 0; i < N * M; ++i) {
            camino[i] = -1;
            visitados[i] = -1;
        }

        camino[0] = id;
        visitados[0] = id;

        new_tablero[id] = tablero[id];


        if (tablero[selec] == tablero[id]) {
            buscar_camino(tablero, id, selec, visitados, &x, camino, &y, N, M);
        }

        if (pertenece(camino, N * M, selec) && x > 1) {
            for (int i = 0; i < N * M; ++i) {
                int id_camino = camino[i];
                if (id_camino != -1) {
                    new_tablero[id_camino] = 'X';
                }
            }
        }

        free(visitados);
    }
}

/*
    tablero: tablero general del juego con casillas marcadas para borrar
    N y M: dimensiones del tablero
    dif: dificultad de la partida

    Salida: tablero con casillas borradas, desplazadas, y caramelos nuevos introducidos
            en caso de que corresponda (mediante puntero a tablero)

*/

__global__ void recolocar_tablero(char* tablero, char* tablero_aux, int* dif, int N, int M) {

    int fila2 = blockIdx.y * blockDim.y + threadIdx.y;
    int col2 = blockIdx.x * blockDim.x + threadIdx.x;
    int id = fila2 * M + col2;

    if (fila2 < N && col2 < M) {

        tablero_aux[id] = tablero[id]; 

        int X_debajo = 0;
        int noX_encima = 0;

        char valor_anterior = tablero[id];
        for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < N * M; i += M) {
            if (i < id && tablero[i] != 'X') {
                noX_encima++;
            }
            if (i > id && tablero[i] == 'X') {
                X_debajo++;
            }
        }

        if (id + M * X_debajo < N * M && X_debajo > 0 && valor_anterior != 'X') {
            tablero_aux[id + M * X_debajo] = valor_anterior;
        }

        if (valor_anterior == 'X') {
            X_debajo++;
        }

        if (X_debajo - noX_encima > 0) {

            if (*dif) {
                tablero_aux[id] = aleatorio(1, 6) + '0';
            }
            else {
                tablero_aux[id] = aleatorio(1, 4) + '0';
            }
        }

    }
}

/*
    tablero: tablero general del juego con casillas marcadas para borrar
    tablero_aux: como no hay syncthreds, escribimos en otro tablero
    fila y columna: del elemento tocado
    borrados: número de casillas marcadas con X
    eleccion: aleatorio entre 1 y 6
    selec: id del elemento seleccionado
    N y M: dimensiones del tablero

    Salida: tablero con casillas borradas, desplazadas, y caramelos nuevos introducidos
            en caso de que corresponda (mediante puntero a tablero_aux)

*/

__global__ void bloquesEspeciales(char* tablero, char* tablero_aux, int fila, int columna, int* borrados, int* eleccion, int N, int M, int selec) {

    int fila2 = blockIdx.y * blockDim.y + threadIdx.y;
    int col2 = blockIdx.x * blockDim.x + threadIdx.x;
    int id = fila2 * M + col2;

    if (fila2 < N && col2 < M) {
        char o_propio = tablero[id];
        char objeto = tablero[selec];
        tablero_aux[id] = tablero[id]; 

        if (objeto == 'B') {
            //Borro fila o columna

            if(id == selec)
            printf("\n\t\t\tOBJETO especial, se aplica el efecto de la bomba ");

            if (*eleccion % 2) {
                if (id == selec)
                printf("borrar FILA\n");
                if (threadIdx.y + blockDim.y * blockIdx.y == fila) {
                    tablero_aux[id] = 'X';
                    atomicAdd(borrados, 1);
                }
            }
            else {
                if (id == selec)
                printf("borrar COLUMNA\n");
                if (threadIdx.x + blockDim.x * blockIdx.x == columna) {
                    tablero_aux[id] = 'X';
                    atomicAdd(borrados, 1);
                }
            }
        }   

        if (objeto == 'T') {
            //Borro todo en un radio de 4 desde el elemento seleccionado

            if (fabsf((double)(threadIdx.x + blockDim.x * blockIdx.x) - columna) < 4.0 && fabsf((double)(threadIdx.y + blockDim.y * blockIdx.y) - fila) < 4.0) {

                tablero_aux[id] = 'X';
                atomicAdd(borrados, 1);

                if (id == selec) {
                    printf("\n\t\t\tOBJETO especial, se aplica el efecto del TNT\n");
                }
            }
        }
        
        if (objeto == 'R') {
            //Borro todos los elementos del tipo

            char tipo = *eleccion + '0';
            if (tablero[id] == tipo) {
                tablero_aux[id] = 'X'; 
                atomicAdd(borrados, 1);
            }

            if (id == selec) {
                tablero_aux[id] = 'X';
                printf("\n\t\tOBJETO especial, se aplica el efecto del rompecabezas, borrar caramelos tipo %c\n", tipo);
            }
        }
    }
}

/*
    tablero: tablero general del juego con casillas marcadas para borrar
    N y M: dimensiones del tablero

    Salida: cantidad de casillas marcadas con X (mediante puntero borrados)

*/

__global__ void contar_borrados(char* tablero, int* borrados, int M) {
    int fila = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int id = fila * M + col;

    if (tablero[id] == 'X') {
        atomicAdd(borrados, 1); 
    }
}

//Funciones auxiliares (HOST)

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
        else if (!strcmp(argv[1], "-m")) {
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
    printf("\t\t");
    printf("%4s", "");
    for (int j = 0; j < m; j++) {
        printf("%3d", j);
    }
    printf("\n");
    printf("\t\t");
    printf("%4s", "");
    for (int j = 0; j < m; j++) {
        printf("%3c", '-');
    }
    printf("\n");
    for (int i = 0; i < n; i++) {
        printf("\t\t%3d|", i);
        for (int j = 0; j < m; j++) {
            printf("%3c", tablero[m * i + j]);
        }
        printf("\n");
    }
    printf("\n");
}


void calcular_dimensiones_optimas(dim3* grid, dim3* bloques) {
    //Obtener las propiedades del dispositivo
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);

    //Calcular bloques e hilos según características del SM
    int max_hilos_SM = deviceProp.maxThreadsPerMultiProcessor;
    int max_bloques_SM = deviceProp.maxBlocksPerMultiProcessor;

    int hilos_bloque = floor(max_hilos_SM / max_bloques_SM);
    int anchura_bloque = floor(sqrt(pow(2, ceil(log2(hilos_bloque)))));
    *bloques = dim3(anchura_bloque, anchura_bloque);

    int bloques_x = ceil(M / (float)anchura_bloque);
    int bloques_y = ceil(N / (float)anchura_bloque);
    *grid = dim3(bloques_x, bloques_y);

    printf("Info CUDA: \n"); 
    printf("Grid --> (%d, %d)\n", bloques_x, bloques_y);
    printf("Bloques --> (%d, %d)\n", anchura_bloque, anchura_bloque);
}

//Flujo principal

int main(int argc, char* argv[]) {
    srand(time(NULL)); //semilla para la ejecucion automatica
    cargar_argumentos(argc, argv);

    dim3 dim_grid; 
    dim3 dim_bloque; 
    calcular_dimensiones_optimas(&dim_grid, &dim_bloque); 

    int tam_tablero = sizeof(char) * N * M;
    char* tablero = (char*)malloc(tam_tablero);
    int posicion = 0; //esta en host


    for (int i = 0; i < N; i++) {
        for (int j = 0; j < M; j++) {
            tablero[M * i + j] = generar_elemento();
        }
    }

    int* d_dif;
    cudaMalloc((void**)&d_dif, sizeof(int));
    cudaMemcpy(d_dif, &dificultad, sizeof(int), cudaMemcpyHostToDevice);

    while (vidas > 0) {
        mostrar_tablero(tablero, N, M);

        char* d_tablero;
        int* d_X;
        int h_X = 0; 
        char* d_rompe;

        cudaMalloc((void**)&d_tablero, sizeof(char) * N * M);
        cudaMalloc((void**)&d_X, sizeof(int));
        cudaMalloc((void**)&d_rompe, sizeof(char));
        cudaMemcpy(d_tablero, tablero, sizeof(char) * N * M, cudaMemcpyHostToDevice);
        cudaMemcpy(d_X, &h_X, sizeof(int), cudaMemcpyHostToDevice);


        int fila;
        int col;
        //Pedir fila y columna al usuario

        if (modo) {

            //Ejecucion manual
            printf("Selecciona fila y columna de la casilla a eliminar: ");
            scanf("%d %d", &fila, &col);

            //Comprobar que la fila y columna son validas
            while (fila < 0 || fila >= N || col < 0 || col >= M) {
                printf("Introduce una fila y columna validas:\n");
                printf("Selecciona fila y columna de la casilla a eliminar: ");
                scanf("%d %d", &fila, &col);
            }
        }
        else {
            //Ejecucion automatica

            //Generar fila y columna aleatorias
            fila = rand() % N;
            col = rand() % M;
            printf("Seleccionada fila %d y columna %d\n", fila, col);
        }

        int seleccionado = fila * M + col;
        int borrados = 0;
        int especial_usado = 0;

        //Comprobar si el elemento seleccionado es un número

        char* d_tablero_aux;
        cudaMalloc((void**)&d_tablero_aux, tam_tablero);

        if (tablero[seleccionado] >= 49 && tablero[seleccionado] <= 54) {
            encontrar_caminos <<<dim_grid, dim_bloque>>> (d_tablero, d_tablero_aux, seleccionado, fila, col, N, M);
            cudaDeviceSynchronize();

            contar_borrados <<<dim_grid, dim_bloque>>> (d_tablero_aux, d_X, M);
            cudaDeviceSynchronize();
        }
        else {
            int h_eleccion = generar_elemento() - '0';
            int* d_eleccion; 
            cudaMalloc((void**)&d_eleccion, sizeof(int));
            cudaMemcpy(d_eleccion, &h_eleccion, sizeof(int), cudaMemcpyHostToDevice);

            bloquesEspeciales <<<dim_grid, dim_bloque>>> (d_tablero, d_tablero_aux, fila, col, d_X, d_eleccion, N, M, seleccionado);
            cudaDeviceSynchronize();

            especial_usado++;
        }

        cudaMemcpy(tablero, d_tablero_aux, sizeof(char) * N * M, cudaMemcpyDeviceToHost);
        cudaMemcpy(&borrados, d_X, sizeof(int), cudaMemcpyDeviceToHost);

        //Decidimos qué pasa en función de los que se han borrado

        printf("\nCaramelos eliminados: %d\n", borrados);

        if (borrados == 0){
            vidas--;
            printf("\nNo hay suficientes caramelos juntos, pierdes una vida!\n");
        }
        else if (borrados == 5 & !especial_usado) {
            tablero[seleccionado] = 'B';
        }
        else if (borrados == 6 & !especial_usado) {
            tablero[seleccionado] = 'T';
        }
        else if (borrados >= 7 & !especial_usado) {
            tablero[seleccionado] = 'R';
        }

        mostrar_tablero(tablero, N, M);
        cudaMemcpy(d_tablero, tablero, sizeof(char) * N * M, cudaMemcpyHostToDevice);

        //Bajamos caramelos y metemos nuevos

        recolocar_tablero <<<dim_grid, dim_bloque >>> (d_tablero, d_tablero_aux, d_dif, N, M);
        cudaDeviceSynchronize();

        cudaMemcpy(tablero, d_tablero_aux, sizeof(char) * N * M, cudaMemcpyDeviceToHost);
        cudaFree(d_tablero);
        cudaFree(d_tablero_aux); 

        printf("Vidas: %d\n", vidas);
    }
    printf("\nFIN DEL JUEGO, TE HAS QUEDADO SIN VIDAS");

    free(tablero);
    cudaFree(d_dif);

    return 0;
}

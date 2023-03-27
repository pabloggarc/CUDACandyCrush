#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <device_functions.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <curand_kernel.h>

//Variables generales del juego

int N;
int M;
int vidas = 5; 
int modo;
int dificultad; 
__constant__ int nuevos_caramelos_facil[4] = { 1, 2, 3, 4 };
__constant__ int nuevos_caramelos_dificil[6] = { 1, 2, 3, 4, 5, 6 };

//Funciones auxiliares (DEVICE)

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

/*
    tablero: tablero del juego
    N y M: dimensiones del tablero
    fila y columna: del elemento tocado

    Salida: tablero con las posiciones a borrar sustituidas por 'X' (mediante el puntero a tablero)

*/

__global__ void encontrar_caminos(char* tablero, int N, int M, int fila, int columna) {
    int selec = fila * M + columna;
    int id = threadIdx.y * N + threadIdx.x;

    //Funcion que busque camino
    int* camino = (int*)malloc(N * M * sizeof(int));
    int* visitados = (int*)malloc(N * M * sizeof(int));
    int x = 1;
    int y = 1;

    for (int i = 1; i < N * M; ++i) {
        camino[i] = -1;
        visitados[i] = -1;
    }

    camino[0] = id;
    visitados[0] = id;

    if (tablero[selec] == tablero[id]) {
        buscar_camino(tablero, id, selec, visitados, &x, camino, &y, N, M);
    }

    if (pertenece(camino, N * M, selec)) {
        for (int i = 0; i < N * M; ++i) {
            int id_camino = camino[i];
            if (id_camino != -1) {
                tablero[id_camino] = 'X';
            }
        }
    }

    free(visitados);
}

/*
    tablero: tablero general del juego con casillas marcadas para borrar
    N y M: dimensiones del tablero
    dif: dificultad de la partida

    Salida: tablero con casillas borradas, desplazadas, y caramelos nuevos introducidos
            en caso de que corresponda (mediante puntero a tablero)

*/

__global__ void recolocar_tablero(char* tablero, int N, int M, int* dif) {
    int id = threadIdx.y * N + threadIdx.x;
    int X_debajo = 0; 
    int noX_encima = 0; 

    char valor_anterior = tablero[id]; 
    for (int i = threadIdx.x; i < N * M; i += M) {
        if (i < id && tablero[i] != 'X') {
            noX_encima++; 
        }
        if (i > id && tablero[i] == 'X') {
            X_debajo++; 
        }
    }

    __syncthreads();

    if (id + M * X_debajo < N * M && X_debajo > 0 && valor_anterior != 'X') {
        tablero[id + M * X_debajo] = valor_anterior; 
    }
    
    if (valor_anterior == 'X') {
        X_debajo++; 
    }

    if (X_debajo - noX_encima > 0) {
        if (*dif) {
            tablero[id] = nuevos_caramelos_dificil[id % 6] + '0'; 
        }
        else {
            tablero[id] = nuevos_caramelos_facil[id % 4] + '0';
        }
    }

}


/*void bloquesEspeciales(char* tablero, int N, int M, int fila, int columna, int longitud) //longitud del camino
{
    int indiceSeleccionado = fila * M + columna;

    if(longitud == 5){ // Si hay cinco bloques del mismo color, uno al lado del otro, cuando lo toca obtiene una bomba. Cuando
                        //lo toca, borra todos los bloques en esa fila o columna de forma aleatoria.
        //genero bloque BOMBA
        tablero[indiceSeleccionado] = 'B';

    }
    else if(longitud == 6){
        //genero bloque TNT
        tablero[indiceSeleccionado] = 'T';
    }
    else if(longitud >= 7){
        //genero bloque ROMPECABEZAS
        tablero[indiceSeleccionado] = 'R';
    }
}*/

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

//Flujo principal

int main(int argc, char* argv[]) {

    srand(time(NULL)); //semilla para la ejecucion automatica
    cargar_argumentos(argc, argv); //aqui ya que es N y M
    int tam_tablero = sizeof(char) * N * M;
    char* tablero = (char*)malloc(tam_tablero);
    int cuantos_hay = 0; //esta en host
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

        cudaMalloc((void**)&d_tablero, sizeof(char) * N * M);

        cudaMemcpy(d_tablero, tablero, sizeof(char) * N * M, cudaMemcpyHostToDevice);


        int fila;
        int col;
        //Pedir fila y columna al usuario
        if (modo == 1)
        {
            //Ejecucion manual
            printf("Selecciona fila y columna de la casilla a eliminar: ");
            scanf("%d %d", &fila, &col);
            //Comprobar que la fila y columna son validas
            while (fila < 0 || fila >= N || col < 0 || col >= M)
            {
                printf("Introduce una fila y columna validas:\n");
                printf("Selecciona fila y columna de la casilla a eliminar: ");
                scanf("%d %d", &fila, &col);
            }
        }
        else
        {
            //Ejecucion automatica

            //Generar fila y columna aleatorias
            fila = rand() % N;
            col = rand() % M;
            printf("Seleccionada fila %d y columna %d\n", fila, col);
        }

       
        

        dim3 bloque(N, M);
        encontrar_caminos << <1, bloque >> > (d_tablero, N, M, fila, col); //genera camino
        cudaMemcpy(tablero, d_tablero, sizeof(char) * N * M, cudaMemcpyDeviceToHost);

        //voy a contar las X del tablero que me ha retornado el kernel encontrar_caminos
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < M; j++) {
                if (tablero[M * i + j] == 'X') { 
                    cuantos_hay++;
                   
                }
            }
        }

        //obtengo el valor de la posicion seleccionada
        posicion = tablero[M * fila + col];


        printf("\nTermino de cambiar los valores por X\n");
        mostrar_tablero(tablero, N, M);



        //si cuantos_hay es 1 (solo hay una X) y por tanto se resta la vida porque no puede formar caminos:
        if (cuantos_hay == 1)
        {
            //resto una vida
            vidas--;
            printf("\nNo hay caminos, pierdes una vida. Te quedan %d vidas.\n", vidas);
            mostrar_tablero(tablero, N, M);
        }

        if (cuantos_hay == 5) { // Si hay cinco bloques del mismo color, uno al lado del otro, cuando lo toca obtiene una bomba. Cuando 
                            //lo toca, borra todos los bloques en esa fila o columna de forma aleatoria.
            //genero bloque BOMBA

            posicion = 'B';
            printf("entro en el if del bloque bomba");
            mostrar_tablero(tablero, N, M);
        }

        if (cuantos_hay == 6) {
            //genero bloque TNT
            posicion = 'T';
            printf("entro en el if del bloque tnt");
            mostrar_tablero(tablero, N, M);
        }

        if (cuantos_hay >= 7) {
            //genero bloque ROMPECABEZAS
            posicion = 'R';
            printf("entro en el if del bloque de rafael rico");
            mostrar_tablero(tablero, N, M);
        }

        printf("\nsi hay camino, print para debuggear");
        recolocar_tablero << <1, bloque >> > (d_tablero, N, M, d_dif); //borra camino
        cudaMemcpy(tablero, d_tablero, sizeof(char) * N * M, cudaMemcpyDeviceToHost);
        cudaFree(d_tablero);
        cuantos_hay = 0; //reinicio el contador de Xs

        //imprimo las vidas 
        printf("Vidas: %d", vidas);
    }
    printf("\nFIN DEL JUEGO, TE HAS QUEDADO SIN VIDAS, mira wasap");

    free(tablero);
    cudaFree(d_dif);

    return 0;
}

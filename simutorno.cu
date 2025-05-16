#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <cuda_runtime.h>
#include "simutorno.h"
#include <chrono>

// Variables globales
int PuntosVueltaHelicoide;
float PasoHelicoide;

// Función para crear la superficie
void CrearSuperficie(TSurf* S, int u, int v) {
    S->UPoints = u;
    S->VPoints = v;
    
    // Reservar memoria para el array 2D
    S->Buffer = (TPoints3D**)malloc(v * sizeof(TPoints3D*));
    for (int i = 0; i < v; i++) {
        S->Buffer[i] = (TPoints3D*)malloc(u * sizeof(TPoints3D));
    }
}

// Función para liberar la memoria de la superficie
void BorrarSuperficie(TSurf* S) {
    if (S->Buffer != NULL) {
        for (int i = 0; i < S->VPoints; i++) {
            free(S->Buffer[i]);
        }
        free(S->Buffer);
        S->Buffer = NULL;
    }
    S->UPoints = 0;
    S->VPoints = 0;
}

// Función para leer la superficie desde un archivo
int LeerSuperficie(char* filename, TSurf* S) {
    FILE* f = fopen(filename, "r");
    if (f == NULL) {
        printf("Error al abrir el archivo %s\n", filename);
        return 0;
    }
    
    // Variables para leer datos
    char line[256];
    int sectionNum = 0, pointsPerSection = 0;
    float step = 0;
    int pointsPerRound = 0;
    
    // Leer parámetros
    while (fgets(line, sizeof(line), f)) {
        if (strstr(line, "SECTION NUMBER") != NULL) {
            sscanf(line, "%*[^:]: %d", &sectionNum);
        } else if (strstr(line, "POINTS PER SECTION") != NULL) {
            sscanf(line, "%*[^:]: %d", &pointsPerSection);
        } else if (strstr(line, "STEP") != NULL) {
            sscanf(line, "%*[^:]: %f", &step);
        } else if (strstr(line, "POINTS PER ROUND") != NULL) {
            sscanf(line, "%*[^:]: %d", &pointsPerRound);
        } else if (strstr(line, "POINTS") != NULL) {
            // Empiezan los datos de puntos
            break;
        }
    }
    
    // Guardar variables globales
    PasoHelicoide = step;
    PuntosVueltaHelicoide = pointsPerRound;
    
    // Crear superficie
    CrearSuperficie(S, sectionNum, pointsPerSection);
    
    // Leer puntos
    for (int v = 0; v < S->VPoints; v++) {
        for (int u = 0; u < S->UPoints; u++) {
            float x, y, z;
            if (fscanf(f, "%f %f %f", &x, &y, &z) != 3) {
                printf("Error en formato de archivo\n");
                fclose(f);
                return 0;
            }
            S->Buffer[v][u].x = x;
            S->Buffer[v][u].y = y;
            S->Buffer[v][u].z = z;
        }
    }
    
    fclose(f);
    return 1;
}

// Implementación en CPU de la simulación
void SimulacionTornoCPU(TSurf S, float** CPUBufferMenorY) {
    // Reserva de la malla de salida (mismos puntos que la original)
    *CPUBufferMenorY = (float*)malloc(S.UPoints * S.VPoints * sizeof(float));

    // Rotación total (360 grados en radianes)
    //float anguloTotal = 2.0f * M_PI;
    int pasos = PuntosVueltaHelicoide;

    // Para cada punto de la superficie
    for (int v = 0; v < S.VPoints; v++) {
        for (int u = 0; u < S.UPoints; u++) {
            float y_original = S.Buffer[v][u].y;
            float z_original = S.Buffer[v][u].z;

            float minY = y_original;

            // Simular la rotación sobre el eje X
            for (int i = 1; i <= pasos; i++) {
                float angulo = i * PasoHelicoide;  // PasoHelicoide ya viene en radianes

                // Rotación sobre el eje X: solo cambian Y y Z
                float y_rot = y_original * cos(angulo) - z_original * sin(angulo);
                float z_rot = y_original * sin(angulo) + z_original * cos(angulo);

                if (y_rot < minY) {
                    minY = y_rot;
                }
            }

            // Guardar el valor mínimo de Y encontrado para este punto
            (*CPUBufferMenorY)[v * S.UPoints + u] = minY;
        }
    }
}

// Kernel CUDA para la simulación
__global__ void tornoKernel(TPoints3D* buffer, float* menorY, 
                           int uPoints, int vPoints, 
                           int puntosVuelta, float paso) {

    int u = threadIdx.x + blockIdx.x * blockDim.x;
    int v = threadIdx.y + blockIdx.y * blockDim.y;

    if (u >= uPoints || v >= vPoints) return;

    int idx = v * uPoints + u;

    float y_original = buffer[idx].y;
    float z_original = buffer[idx].z;
    float minY = y_original;

    for (int i = 1; i <= puntosVuelta; i++) {
        float angulo = i * paso;

        float y_rot = y_original * cosf(angulo) - z_original * sinf(angulo);

        if (y_rot < minY) {
            minY = y_rot;
        }
    }

    menorY[idx] = minY;
    
}

// Implementación en GPU de la simulación
void SimulacionTornoGPU(TSurf S, float** GPUBufferMenorY) {
    int numPuntos = S.UPoints * S.VPoints;

    // Reservar memoria en GPU
    TPoints3D* GPU_buffer;
    float* GPU_menorY;
    // Asignamos memoria en la GPU para la malla de puntos (S.Buffer)
    cudaMalloc((void**)&GPU_buffer, numPuntos * sizeof(TPoints3D));
    // Asignamos memoria en la GPU para el buffer de resultados
    cudaMalloc((void**)&GPU_menorY, numPuntos * sizeof(float));

    // Crear array plano de TPoints3D para copiar desde la CPU
    TPoints3D* buffer_plano = (TPoints3D*)malloc(numPuntos * sizeof(TPoints3D));
    for (int v = 0; v < S.VPoints; ++v) {
        for (int u = 0; u < S.UPoints; ++u) {
            buffer_plano[v * S.UPoints + u] = S.Buffer[v][u];
        }
    }

    // Copiar superficie a GPU
    cudaMemcpy(GPU_buffer, buffer_plano, numPuntos * sizeof(TPoints3D), cudaMemcpyHostToDevice);

    // Definir bloques e hilos
    dim3 blockSize(16, 16);  //16x16 es un estandar razonable, pero se podría buscar otras configuraciónes según el tamaño de la malla
    dim3 gridSize((S.UPoints + blockSize.x - 1) / blockSize.x,
                  (S.VPoints + blockSize.y - 1) / blockSize.y);

    // Ejecutar kernel de simulación
    tornoKernel<<<gridSize, blockSize>>>(GPU_buffer, GPU_menorY,
                                         S.UPoints, S.VPoints,
                                         PuntosVueltaHelicoide, PasoHelicoide);

    // Esperar finalización y chequear errores
    cudaDeviceSynchronize();

    // Reservar salida en CPU
    *GPUBufferMenorY = (float*)malloc(numPuntos * sizeof(float));
    // Copiar resultados a CPU
    cudaMemcpy(*GPUBufferMenorY, GPU_menorY, numPuntos * sizeof(float), cudaMemcpyDeviceToHost);

    // Liberar recursos
    free(buffer_plano);
    cudaFree(GPU_buffer);
    cudaFree(GPU_menorY);
}

// Función de prueba
void runTest(char* filename) {
    TSurf superficie;
    float *CPUBuffer = NULL, *GPUBuffer = NULL;

    if (!LeerSuperficie(filename, &superficie)) {
        printf("Error leyendo la superficie\n");
        return;
    }
    
    // Ejecutar CPU
    //TODO: Registrar tiempo ejecución CPU    
    auto start_cpu = std::chrono::high_resolution_clock::now();
    SimulacionTornoCPU(superficie, &CPUBuffer);
    auto end_cpu = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration_CPU = end_cpu - start_cpu;

    // Ejecutar GPU
    //TODO: Registrar tiempo ejecución GPU
    auto start_gpu = std::chrono::high_resolution_clock::now();
    SimulacionTornoGPU(superficie, &GPUBuffer);
    auto end_gpu = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration_GPU = end_gpu - start_gpu;

    // Comparar resultados
    int errores = 0;
    int total = superficie.UPoints * superficie.VPoints;
    for (int i = 0; i < total; i++) {
        if (fabs(CPUBuffer[i] - GPUBuffer[i]) > 1e-3) {
            errores++;
            if (errores < 10)  // mostrar los primeros errores
                printf("Error en punto %d: CPU = %f, GPU = %f\n", i, CPUBuffer[i], GPUBuffer[i]);
        }
		/*else 
			printf("Correcto punto %d: CPU = %f, GPU = %f\n", i, CPUBuffer[i], GPUBuffer[i]);*/
    }

    if (errores == 0){
        //printf("¡Correcto! CPU y GPU coinciden.\n");
        printf("Tiempo en CPU | Tiempo en GPU\n");
        printf("-----------------------------\n");        
        printf("    %f  | %f \n", duration_CPU, duration_GPU);
    }
    else
        printf("Diferencias detectadas en %d puntos\n", errores);

    free(CPUBuffer);
    free(GPUBuffer);
    BorrarSuperficie(&superficie);
}

// Programa principal
int main(int argc, char** argv) {
    if (argc < 2) {
        printf("Uso: %s archivo.for\n", argv[0]);
        return 1;
    }
    
    runTest(argv[1]);
    return 0;
}

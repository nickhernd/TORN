#include "simutorno.h"
#include <cmath>
#include <cstdlib>
#include <cstring>

// Variables globales
int PuntosVueltaHelicoide = 0;
float PasoHelicoide = 0.0f;

// Función para crear la superficie
void CrearSuperficie(TSurf* S, int u, int v) {
    S->UPoints = u;
    S->VPoints = v;

    S->Buffer = (TPoints3D**)malloc(v * sizeof(TPoints3D*));
    for (int i = 0; i < v; i++) {
        S->Buffer[i] = (TPoints3D*)malloc(u * sizeof(TPoints3D));
    }
}

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

int LeerSuperficie(char* filename, TSurf* S) {
    FILE* f = fopen(filename, "r");
    if (f == NULL) {
        printf("Error al abrir el archivo %s\n", filename);
        return 0;
    }

    char line[256];
    int sectionNum = 0, pointsPerSection = 0;
    float step = 0;
    int pointsPerRound = 0;

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
            break;
        }
    }

    PasoHelicoide = step;
    PuntosVueltaHelicoide = pointsPerRound;

    CrearSuperficie(S, sectionNum, pointsPerSection);

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

// STUB: Simulación CPU (versión simplificada)
void SimulacionTornoCPU(TSurf S, float** CPUBufferMenorY) {
    *CPUBufferMenorY = (float*)malloc(S.UPoints * S.VPoints * sizeof(float));

    int pasos = PuntosVueltaHelicoide;

    for (int v = 0; v < S.VPoints; v++) {
        for (int u = 0; u < S.UPoints; u++) {
            float y_original = S.Buffer[v][u].y;
            float z_original = S.Buffer[v][u].z;

            float minY = y_original;

            for (int i = 1; i <= pasos; i++) {
                float angulo = i * PasoHelicoide;

                float y_rot = y_original * cos(angulo) - z_original * sin(angulo);

                if (y_rot < minY) {
                    minY = y_rot;
                }
            }

            (*CPUBufferMenorY)[v * S.UPoints + u] = minY;
        }
    }
}

// STUB: Simulación GPU (usa la misma lógica que CPU para demo)
void SimulacionTornoGPU(TSurf S, float** GPUBufferMenorY) {
    printf("DEMO MODE: Usando simulacion CPU (CUDA no disponible)\n");
    SimulacionTornoCPU(S, GPUBufferMenorY);
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
    auto start_cpu = std::chrono::high_resolution_clock::now();
    SimulacionTornoCPU(superficie, &CPUBuffer);
    auto end_cpu = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration_CPU = end_cpu - start_cpu;

    // En modo demo, GPU es igual a CPU
    auto start_gpu = std::chrono::high_resolution_clock::now();
    SimulacionTornoGPU(superficie, &GPUBuffer);
    auto end_gpu = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration_GPU = end_gpu - start_gpu;

    // Comparar resultados
    int errores = 0;
    int total = superficie.UPoints * superficie.VPoints;
    for (int i = 0; i < total; i++) {
        if (fabs(CPUBuffer[i] - GPUBuffer[i]) > 1e-5) {
            errores++;
        }
    }

    if (errores == 0){
        printf("Correcto! CPU y GPU (stub) coinciden.\n");
        printf("Tiempo en CPU | Tiempo en GPU\n");
        printf("-----------------------------\n");
        printf("%f | %f \n", duration_CPU.count(), duration_GPU.count());
    }
    else
        printf("Diferencias detectadas en %d puntos\n", errores);

    free(CPUBuffer);
    free(GPUBuffer);
    BorrarSuperficie(&superficie);
}

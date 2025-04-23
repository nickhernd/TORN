#ifndef SIMUTORNO_H
#define SIMUTORNO_H

// Estructura para puntos 3D
typedef struct {
    float x;
    float y;
    float z;
} TPoints3D;

// Estructura para la superficie
typedef struct {
    int UPoints;
    int VPoints;
    TPoints3D** Buffer;
} TSurf;

// Variables globales (definidas en simutorno.cu)
extern int PuntosVueltaHelicoide;
extern float PasoHelicoide;

// Funciones auxiliares
void CrearSuperficie(TSurf* S, int u, int v);
void BorrarSuperficie(TSurf* S);
int LeerSuperficie(char* filename, TSurf* S);

// Funciones de simulación
void SimulacionTornoCPU(TSurf S, float* CPUBufferMenorY);
void SimulacionTornoGPU(TSurf S, float* GPUBufferMenorY);

#endif // SIMUTORNO_H

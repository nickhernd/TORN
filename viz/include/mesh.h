#ifndef MESH_H
#define MESH_H

#include <glad/gl.h>
#include <glm/glm.hpp>
#include <vector>
#include "../../simutorno.h"

// Estructura de vértice
struct Vertex {
    glm::vec3 Position;
    glm::vec3 Normal;
    glm::vec3 Color;
};

// Estructura de malla
typedef struct {
    std::vector<Vertex> vertices;
    std::vector<unsigned int> indices;

    // Datos de OpenGL
    unsigned int VAO;
    unsigned int VBO;
    unsigned int EBO;
} Mesh;

// Funciones de malla
Mesh* CreateMesh(std::vector<Vertex> vertices, std::vector<unsigned int> indices);
Mesh* CreateMeshFromSurface(TSurf surface);
Mesh* CreateMeshFromSurfaceWithHeatmap(TSurf surface, float* depthBuffer, float minDepth, float maxDepth);
void DestroyMesh(Mesh* mesh);
void SetupMesh(Mesh* mesh);
void DrawMesh(Mesh* mesh);

// Funciones auxiliares
glm::vec3 CalculateNormal(glm::vec3 p1, glm::vec3 p2, glm::vec3 p3);
glm::vec3 HeatmapColor(float value, float min, float max);

#endif // MESH_H

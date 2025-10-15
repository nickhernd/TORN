#include "../include/mesh.h"
#include <iostream>
#include <algorithm>

Mesh* CreateMesh(std::vector<Vertex> vertices, std::vector<unsigned int> indices) {
    Mesh* mesh = new Mesh();
    mesh->vertices = vertices;
    mesh->indices = indices;
    SetupMesh(mesh);
    return mesh;
}

Mesh* CreateMeshFromSurface(TSurf surface) {
    std::vector<Vertex> vertices;
    std::vector<unsigned int> indices;

    // Convertir puntos de la superficie a vértices
    for (int v = 0; v < surface.VPoints; v++) {
        for (int u = 0; u < surface.UPoints; u++) {
            Vertex vertex;
            vertex.Position = glm::vec3(
                surface.Buffer[v][u].x,
                surface.Buffer[v][u].y,
                surface.Buffer[v][u].z
            );

            // Calcular normal aproximada
            glm::vec3 normal(0.0f, 1.0f, 0.0f);
            if (u > 0 && u < surface.UPoints - 1 && v > 0 && v < surface.VPoints - 1) {
                glm::vec3 p1(surface.Buffer[v][u-1].x, surface.Buffer[v][u-1].y, surface.Buffer[v][u-1].z);
                glm::vec3 p2(surface.Buffer[v][u+1].x, surface.Buffer[v][u+1].y, surface.Buffer[v][u+1].z);
                glm::vec3 p3(surface.Buffer[v-1][u].x, surface.Buffer[v-1][u].y, surface.Buffer[v-1][u].z);
                glm::vec3 p4(surface.Buffer[v+1][u].x, surface.Buffer[v+1][u].y, surface.Buffer[v+1][u].z);

                glm::vec3 tangentU = p2 - p1;
                glm::vec3 tangentV = p4 - p3;
                normal = glm::normalize(glm::cross(tangentU, tangentV));
            }

            vertex.Normal = normal;
            vertex.Color = glm::vec3(0.7f, 0.7f, 0.7f); // Color gris por defecto

            vertices.push_back(vertex);
        }
    }

    // Generar índices para triángulos
    for (int v = 0; v < surface.VPoints - 1; v++) {
        for (int u = 0; u < surface.UPoints - 1; u++) {
            int topLeft = v * surface.UPoints + u;
            int topRight = topLeft + 1;
            int bottomLeft = (v + 1) * surface.UPoints + u;
            int bottomRight = bottomLeft + 1;

            // Primer triángulo
            indices.push_back(topLeft);
            indices.push_back(bottomLeft);
            indices.push_back(topRight);

            // Segundo triángulo
            indices.push_back(topRight);
            indices.push_back(bottomLeft);
            indices.push_back(bottomRight);
        }
    }

    return CreateMesh(vertices, indices);
}

Mesh* CreateMeshFromSurfaceWithHeatmap(TSurf surface, float* depthBuffer, float minDepth, float maxDepth) {
    std::vector<Vertex> vertices;
    std::vector<unsigned int> indices;

    // Convertir puntos de la superficie a vértices con heatmap
    for (int v = 0; v < surface.VPoints; v++) {
        for (int u = 0; u < surface.UPoints; u++) {
            int idx = v * surface.UPoints + u;

            Vertex vertex;
            vertex.Position = glm::vec3(
                surface.Buffer[v][u].x,
                surface.Buffer[v][u].y,
                surface.Buffer[v][u].z
            );

            // Calcular normal
            glm::vec3 normal(0.0f, 1.0f, 0.0f);
            if (u > 0 && u < surface.UPoints - 1 && v > 0 && v < surface.VPoints - 1) {
                glm::vec3 p1(surface.Buffer[v][u-1].x, surface.Buffer[v][u-1].y, surface.Buffer[v][u-1].z);
                glm::vec3 p2(surface.Buffer[v][u+1].x, surface.Buffer[v][u+1].y, surface.Buffer[v][u+1].z);
                glm::vec3 p3(surface.Buffer[v-1][u].x, surface.Buffer[v-1][u].y, surface.Buffer[v-1][u].z);
                glm::vec3 p4(surface.Buffer[v+1][u].x, surface.Buffer[v+1][u].y, surface.Buffer[v+1][u].z);

                glm::vec3 tangentU = p2 - p1;
                glm::vec3 tangentV = p4 - p3;
                normal = glm::normalize(glm::cross(tangentU, tangentV));
            }

            vertex.Normal = normal;

            // Color basado en profundidad (heatmap)
            vertex.Color = HeatmapColor(depthBuffer[idx], minDepth, maxDepth);

            vertices.push_back(vertex);
        }
    }

    // Generar índices (igual que antes)
    for (int v = 0; v < surface.VPoints - 1; v++) {
        for (int u = 0; u < surface.UPoints - 1; u++) {
            int topLeft = v * surface.UPoints + u;
            int topRight = topLeft + 1;
            int bottomLeft = (v + 1) * surface.UPoints + u;
            int bottomRight = bottomLeft + 1;

            indices.push_back(topLeft);
            indices.push_back(bottomLeft);
            indices.push_back(topRight);

            indices.push_back(topRight);
            indices.push_back(bottomLeft);
            indices.push_back(bottomRight);
        }
    }

    return CreateMesh(vertices, indices);
}

void DestroyMesh(Mesh* mesh) {
    if (mesh) {
        glDeleteVertexArrays(1, &mesh->VAO);
        glDeleteBuffers(1, &mesh->VBO);
        glDeleteBuffers(1, &mesh->EBO);
        delete mesh;
    }
}

void SetupMesh(Mesh* mesh) {
    glGenVertexArrays(1, &mesh->VAO);
    glGenBuffers(1, &mesh->VBO);
    glGenBuffers(1, &mesh->EBO);

    glBindVertexArray(mesh->VAO);

    // VBO
    glBindBuffer(GL_ARRAY_BUFFER, mesh->VBO);
    glBufferData(GL_ARRAY_BUFFER, mesh->vertices.size() * sizeof(Vertex),
                 &mesh->vertices[0], GL_STATIC_DRAW);

    // EBO
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, mesh->EBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, mesh->indices.size() * sizeof(unsigned int),
                 &mesh->indices[0], GL_STATIC_DRAW);

    // Atributos de vértices
    // Position
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)0);

    // Normal
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex),
                          (void*)offsetof(Vertex, Normal));

    // Color
    glEnableVertexAttribArray(2);
    glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex),
                          (void*)offsetof(Vertex, Color));

    glBindVertexArray(0);
}

void DrawMesh(Mesh* mesh) {
    glBindVertexArray(mesh->VAO);
    glDrawElements(GL_TRIANGLES, mesh->indices.size(), GL_UNSIGNED_INT, 0);
    glBindVertexArray(0);
}

glm::vec3 CalculateNormal(glm::vec3 p1, glm::vec3 p2, glm::vec3 p3) {
    glm::vec3 u = p2 - p1;
    glm::vec3 v = p3 - p1;
    return glm::normalize(glm::cross(u, v));
}

glm::vec3 HeatmapColor(float value, float min, float max) {
    // Normalizar valor entre 0 y 1
    float t = (value - min) / (max - min);
    t = std::clamp(t, 0.0f, 1.0f);

    // Heatmap: azul -> cian -> verde -> amarillo -> rojo
    glm::vec3 color;

    if (t < 0.25f) {
        // Azul a cian
        float s = t / 0.25f;
        color = glm::vec3(0.0f, s, 1.0f);
    } else if (t < 0.5f) {
        // Cian a verde
        float s = (t - 0.25f) / 0.25f;
        color = glm::vec3(0.0f, 1.0f, 1.0f - s);
    } else if (t < 0.75f) {
        // Verde a amarillo
        float s = (t - 0.5f) / 0.25f;
        color = glm::vec3(s, 1.0f, 0.0f);
    } else {
        // Amarillo a rojo
        float s = (t - 0.75f) / 0.25f;
        color = glm::vec3(1.0f, 1.0f - s, 0.0f);
    }

    return color;
}

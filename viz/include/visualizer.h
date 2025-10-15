#ifndef VISUALIZER_H
#define VISUALIZER_H

#include <glad/gl.h>
#include <GLFW/glfw3.h>
#include "../../simutorno.h"

// Estructura del visualizador
typedef struct {
    GLFWwindow* window;
    int width;
    int height;
    const char* title;
    bool isRunning;

    // Callbacks
    void (*onResize)(int width, int height);
    void (*onKeyPress)(int key, int action);
    void (*onMouseMove)(double xpos, double ypos);
    void (*onMouseScroll)(double xoffset, double yoffset);
} Visualizer;

// Funciones del visualizador
Visualizer* CreateVisualizer(int width, int height, const char* title);
void DestroyVisualizer(Visualizer* viz);
bool InitVisualizer(Visualizer* viz);
void UpdateVisualizer(Visualizer* viz);
bool ShouldCloseVisualizer(Visualizer* viz);
void ClearScreen(float r, float g, float b, float a);
void SwapBuffers(Visualizer* viz);
void PollEvents();

// Funciones auxiliares
double GetTime();
void SetVSync(bool enabled);

#endif // VISUALIZER_H

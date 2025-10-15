#ifndef GUI_MANAGER_H
#define GUI_MANAGER_H

#include <imgui.h>
#include <string>
#include <vector>
#include "../../simutorno.h"

// Estado de la GUI
struct GUIState {
    // Ventanas
    bool showMainPanel;
    bool showMetrics;
    bool showSettings;
    bool showFileDialog;
    bool showAbout;

    // Configuración de visualización
    bool useHeatmap;
    bool wireframeMode;
    bool showNormals;
    float backgroundColor[3];
    float lightPosition[3];
    float lightColor[3];

    // Configuración de cámara
    float cameraSpeed;
    float mouseSensitivity;
    float fov;

    // Información de la simulación
    int surfacePoints;
    int surfaceTriangles;
    float cpuTime;
    float gpuTime;
    float speedup;
    float minDepth;
    float maxDepth;

    // Control de simulación
    bool autoRotate;
    float rotationSpeed;
    bool animateSimulation;
    float animationProgress;

    // Archivo actual
    char currentFile[256];
    std::vector<std::string> recentFiles;

    // FPS
    float fps;
    float frameTime;
};

// Funciones del GUI Manager
void InitGUI(GLFWwindow* window);
void ShutdownGUI();
void BeginGUIFrame();
void EndGUIFrame();

// Paneles
void RenderMainPanel(GUIState* state);
void RenderMetricsPanel(GUIState* state);
void RenderSettingsPanel(GUIState* state);
void RenderFileDialog(GUIState* state);
void RenderAboutPanel(GUIState* state);

// Utilidades
void ApplyDarkTheme();
void SaveGUIConfig(const char* filename);
void LoadGUIConfig(const char* filename);

#endif // GUI_MANAGER_H

#include <iostream>
#include <string>
#include "simutorno.h"
#include "viz/include/visualizer.h"
#include "viz/include/camera.h"
#include "viz/include/shader.h"
#include "viz/include/mesh.h"
#include "gui/include/gui_manager.h"

// Estado de la aplicación
struct AppState {
    Camera* camera;
    float deltaTime;
    float lastFrame;
    bool firstMouse;
    float lastX;
    float lastY;
    GUIState guiState;

    // Meshes
    Mesh* meshOriginal;
    Mesh* meshHeatmap;

    // Datos de simulación
    TSurf superficie;
    float* depthBuffer;
};

AppState appState;

// Callbacks
void onResize(int width, int height) {
    glViewport(0, 0, width, height);
}

void onKeyPress(int key, int action) {
    if (action == GLFW_PRESS) {
        if (key == GLFW_KEY_F) {
            appState.guiState.wireframeMode = !appState.guiState.wireframeMode;
            glPolygonMode(GL_FRONT_AND_BACK, appState.guiState.wireframeMode ? GL_LINE : GL_FILL);
        }
        if (key == GLFW_KEY_H) {
            appState.guiState.useHeatmap = !appState.guiState.useHeatmap;
        }
    }
}

void onMouseMove(double xpos, double ypos) {
    if (appState.firstMouse) {
        appState.lastX = xpos;
        appState.lastY = ypos;
        appState.firstMouse = false;
    }

    float xoffset = xpos - appState.lastX;
    float yoffset = appState.lastY - ypos;

    appState.lastX = xpos;
    appState.lastY = ypos;

    ProcessMouseMovement(appState.camera, xoffset, yoffset);
}

void onMouseScroll(double xoffset, double yoffset) {
    ProcessMouseScroll(appState.camera, yoffset);
}

void processInput(Visualizer* viz) {
    GLFWwindow* window = viz->window;

    float speed = appState.guiState.cameraSpeed;

    if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
        ProcessKeyboard(appState.camera, FORWARD, appState.deltaTime);
    if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
        ProcessKeyboard(appState.camera, BACKWARD, appState.deltaTime);
    if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
        ProcessKeyboard(appState.camera, LEFT, appState.deltaTime);
    if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
        ProcessKeyboard(appState.camera, RIGHT, appState.deltaTime);
    if (glfwGetKey(window, GLFW_KEY_SPACE) == GLFW_PRESS)
        ProcessKeyboard(appState.camera, UP, appState.deltaTime);
    if (glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS)
        ProcessKeyboard(appState.camera, DOWN, appState.deltaTime);

    // Aplicar velocidad de cámara desde GUI
    appState.camera->MovementSpeed = speed;
}

void initGUIState() {
    // Ventanas
    appState.guiState.showMainPanel = true;
    appState.guiState.showMetrics = true;
    appState.guiState.showSettings = false;
    appState.guiState.showFileDialog = false;
    appState.guiState.showAbout = false;

    // Visualización
    appState.guiState.useHeatmap = true;
    appState.guiState.wireframeMode = false;
    appState.guiState.showNormals = false;
    appState.guiState.backgroundColor[0] = 0.1f;
    appState.guiState.backgroundColor[1] = 0.1f;
    appState.guiState.backgroundColor[2] = 0.15f;
    appState.guiState.lightPosition[0] = 10.0f;
    appState.guiState.lightPosition[1] = 10.0f;
    appState.guiState.lightPosition[2] = 10.0f;
    appState.guiState.lightColor[0] = 1.0f;
    appState.guiState.lightColor[1] = 1.0f;
    appState.guiState.lightColor[2] = 1.0f;

    // Cámara
    appState.guiState.cameraSpeed = 2.5f;
    appState.guiState.mouseSensitivity = 0.1f;
    appState.guiState.fov = 45.0f;

    // Animación
    appState.guiState.autoRotate = false;
    appState.guiState.rotationSpeed = 1.0f;
    appState.guiState.animateSimulation = false;
    appState.guiState.animationProgress = 0.0f;

    strcpy(appState.guiState.currentFile, "ejemplo.for");
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cout << "Uso: " << argv[0] << " archivo.for" << std::endl;
        return 1;
    }

    std::cout << "=== TORN Visualizer con GUI ===" << std::endl;
    std::cout << "Cargando superficie desde: " << argv[1] << std::endl;

    // Cargar superficie
    if (!LeerSuperficie(argv[1], &appState.superficie)) {
        std::cerr << "Error al cargar la superficie" << std::endl;
        return 1;
    }

    std::cout << "Superficie cargada: " << appState.superficie.UPoints << "x"
              << appState.superficie.VPoints << " puntos" << std::endl;

    // Ejecutar simulación CUDA
    std::cout << "Ejecutando simulación CUDA..." << std::endl;

    auto start_gpu = std::chrono::high_resolution_clock::now();
    SimulacionTornoGPU(appState.superficie, &appState.depthBuffer);
    auto end_gpu = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration_GPU = end_gpu - start_gpu;

    std::cout << "Simulación completada en " << duration_GPU.count() << "s" << std::endl;

    // Encontrar min/max para el heatmap
    float minDepth = appState.depthBuffer[0];
    float maxDepth = appState.depthBuffer[0];
    int totalPoints = appState.superficie.UPoints * appState.superficie.VPoints;
    for (int i = 1; i < totalPoints; i++) {
        if (appState.depthBuffer[i] < minDepth) minDepth = appState.depthBuffer[i];
        if (appState.depthBuffer[i] > maxDepth) maxDepth = appState.depthBuffer[i];
    }
    std::cout << "Rango de profundidad: [" << minDepth << ", " << maxDepth << "]" << std::endl;

    // Inicializar visualizador
    Visualizer* viz = CreateVisualizer(1600, 900, "TORN Visualizer - Simulacion de Torno 3D con GUI");
    if (!InitVisualizer(viz)) {
        std::cerr << "Error al inicializar el visualizador" << std::endl;
        return 1;
    }

    // Configurar callbacks
    viz->onResize = onResize;
    viz->onKeyPress = onKeyPress;
    viz->onMouseMove = onMouseMove;
    viz->onMouseScroll = onMouseScroll;

    // Inicializar GUI
    InitGUI(viz->window);
    initGUIState();

    // Inicializar estado
    appState.camera = CreateCamera(
        glm::vec3(0.0f, 5.0f, 10.0f),
        glm::vec3(0.0f, 1.0f, 0.0f),
        -90.0f, -20.0f
    );
    appState.deltaTime = 0.0f;
    appState.lastFrame = 0.0f;
    appState.firstMouse = true;
    appState.lastX = 800.0f;
    appState.lastY = 450.0f;

    // Guardar datos en GUI State
    strcpy(appState.guiState.currentFile, argv[1]);
    appState.guiState.surfacePoints = totalPoints;
    appState.guiState.gpuTime = duration_GPU.count();
    appState.guiState.cpuTime = 0.0f; // Por ahora
    appState.guiState.speedup = 1.0f; // Por ahora
    appState.guiState.minDepth = minDepth;
    appState.guiState.maxDepth = maxDepth;

    // Crear shaders
    Shader* basicShader = CreateShader("shaders/basic.vert", "shaders/basic.frag");
    if (!basicShader) {
        std::cerr << "Error al cargar shaders" << std::endl;
        return 1;
    }

    // Crear mallas
    std::cout << "Generando mallas..." << std::endl;
    appState.meshOriginal = CreateMeshFromSurface(appState.superficie);
    appState.meshHeatmap = CreateMeshFromSurfaceWithHeatmap(appState.superficie, appState.depthBuffer, minDepth, maxDepth);
    appState.guiState.surfaceTriangles = appState.meshOriginal->indices.size() / 3;
    std::cout << "Mallas generadas: " << appState.meshOriginal->vertices.size() << " vertices, "
              << appState.guiState.surfaceTriangles << " triangulos" << std::endl;

    SetVSync(true);

    std::cout << "\n=== Controles ===" << std::endl;
    std::cout << "WASD: Mover camara | Mouse: Rotar | Scroll: Zoom" << std::endl;
    std::cout << "Espacio/Shift: Subir/Bajar | H: Heatmap | F: Wireframe | ESC: Salir\n" << std::endl;

    // Auto-rotación
    float rotation = 0.0f;

    // Loop principal
    while (!ShouldCloseVisualizer(viz)) {
        // Delta time
        float currentFrame = GetTime();
        appState.deltaTime = currentFrame - appState.lastFrame;
        appState.lastFrame = currentFrame;

        // Actualizar FPS en GUI
        appState.guiState.fps = 1.0f / appState.deltaTime;
        appState.guiState.frameTime = appState.deltaTime * 1000.0f;

        // Input
        processInput(viz);

        // Auto-rotación
        if (appState.guiState.autoRotate) {
            rotation += appState.guiState.rotationSpeed * appState.deltaTime;
        }

        // Render
        ClearScreen(
            appState.guiState.backgroundColor[0],
            appState.guiState.backgroundColor[1],
            appState.guiState.backgroundColor[2],
            1.0f
        );

        // Aplicar wireframe desde GUI
        glPolygonMode(GL_FRONT_AND_BACK, appState.guiState.wireframeMode ? GL_LINE : GL_FILL);

        // Usar shader
        UseShader(basicShader);

        // Configurar matrices
        glm::mat4 projection = glm::perspective(
            glm::radians(appState.guiState.fov),
            1600.0f / 900.0f,
            0.1f, 100.0f
        );
        glm::mat4 view = GetViewMatrix(appState.camera);
        glm::mat4 model = glm::mat4(1.0f);

        // Aplicar auto-rotación
        if (appState.guiState.autoRotate) {
            model = glm::rotate(model, rotation, glm::vec3(0.0f, 1.0f, 0.0f));
        }

        SetMat4(basicShader, "projection", projection);
        SetMat4(basicShader, "view", view);
        SetMat4(basicShader, "model", model);

        // Configurar luces desde GUI
        SetVec3(basicShader, "lightPos",
                appState.guiState.lightPosition[0],
                appState.guiState.lightPosition[1],
                appState.guiState.lightPosition[2]);
        SetVec3v(basicShader, "viewPos", appState.camera->Position);
        SetVec3(basicShader, "lightColor",
                appState.guiState.lightColor[0],
                appState.guiState.lightColor[1],
                appState.guiState.lightColor[2]);
        SetBool(basicShader, "useVertexColor", appState.guiState.useHeatmap);

        // Dibujar malla apropiada
        if (appState.guiState.useHeatmap) {
            DrawMesh(appState.meshHeatmap);
        } else {
            DrawMesh(appState.meshOriginal);
        }

        // Renderizar GUI
        BeginGUIFrame();
        RenderMainPanel(&appState.guiState);
        RenderMetricsPanel(&appState.guiState);
        RenderSettingsPanel(&appState.guiState);
        RenderFileDialog(&appState.guiState);
        RenderAboutPanel(&appState.guiState);
        EndGUIFrame();

        SwapBuffers(viz);
        PollEvents();
    }

    // Limpieza
    std::cout << "Cerrando..." << std::endl;
    ShutdownGUI();
    DestroyMesh(appState.meshOriginal);
    DestroyMesh(appState.meshHeatmap);
    DestroyShader(basicShader);
    DestroyCamera(appState.camera);
    DestroyVisualizer(viz);
    BorrarSuperficie(&appState.superficie);
    free(appState.depthBuffer);

    std::cout << "Adios!" << std::endl;
    return 0;
}

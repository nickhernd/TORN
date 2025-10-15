#include <iostream>
#include <string>
#include "simutorno.h"
#include "viz/include/visualizer.h"
#include "viz/include/camera.h"
#include "viz/include/shader.h"
#include "viz/include/mesh.h"

// Estado de la aplicación
struct AppState {
    Camera* camera;
    float deltaTime;
    float lastFrame;
    bool firstMouse;
    float lastX;
    float lastY;
    bool wireframeMode;
    bool showHeatmap;
};

AppState appState;

// Callbacks
void onResize(int width, int height) {
    glViewport(0, 0, width, height);
}

void onKeyPress(int key, int action) {
    if (action == GLFW_PRESS) {
        if (key == GLFW_KEY_F) {
            appState.wireframeMode = !appState.wireframeMode;
            glPolygonMode(GL_FRONT_AND_BACK, appState.wireframeMode ? GL_LINE : GL_FILL);
            std::cout << "Wireframe: " << (appState.wireframeMode ? "ON" : "OFF") << std::endl;
        }
        if (key == GLFW_KEY_H) {
            appState.showHeatmap = !appState.showHeatmap;
            std::cout << "Heatmap: " << (appState.showHeatmap ? "ON" : "OFF") << std::endl;
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
    float yoffset = appState.lastY - ypos; // Invertido porque Y va de abajo hacia arriba

    appState.lastX = xpos;
    appState.lastY = ypos;

    ProcessMouseMovement(appState.camera, xoffset, yoffset);
}

void onMouseScroll(double xoffset, double yoffset) {
    ProcessMouseScroll(appState.camera, yoffset);
}

void processInput(Visualizer* viz) {
    GLFWwindow* window = viz->window;

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
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cout << "Uso: " << argv[0] << " archivo.for" << std::endl;
        return 1;
    }

    std::cout << "=== TORN Visualizer ===" << std::endl;
    std::cout << "Cargando superficie desde: " << argv[1] << std::endl;

    // Cargar superficie
    TSurf superficie;
    if (!LeerSuperficie(argv[1], &superficie)) {
        std::cerr << "Error al cargar la superficie" << std::endl;
        return 1;
    }

    std::cout << "Superficie cargada: " << superficie.UPoints << "x" << superficie.VPoints << " puntos" << std::endl;

    // Ejecutar simulación CUDA
    std::cout << "Ejecutando simulación CUDA..." << std::endl;
    float* depthBuffer = nullptr;
    SimulacionTornoGPU(superficie, &depthBuffer);
    std::cout << "Simulación completada" << std::endl;

    // Encontrar min/max para el heatmap
    float minDepth = depthBuffer[0];
    float maxDepth = depthBuffer[0];
    int totalPoints = superficie.UPoints * superficie.VPoints;
    for (int i = 1; i < totalPoints; i++) {
        if (depthBuffer[i] < minDepth) minDepth = depthBuffer[i];
        if (depthBuffer[i] > maxDepth) maxDepth = depthBuffer[i];
    }
    std::cout << "Rango de profundidad: [" << minDepth << ", " << maxDepth << "]" << std::endl;

    // Inicializar visualizador
    Visualizer* viz = CreateVisualizer(1280, 720, "TORN Visualizer - Simulacion de Torno 3D");
    if (!InitVisualizer(viz)) {
        std::cerr << "Error al inicializar el visualizador" << std::endl;
        return 1;
    }

    // Configurar callbacks
    viz->onResize = onResize;
    viz->onKeyPress = onKeyPress;
    viz->onMouseMove = onMouseMove;
    viz->onMouseScroll = onMouseScroll;

    // Inicializar estado
    appState.camera = CreateCamera(
        glm::vec3(0.0f, 5.0f, 10.0f),
        glm::vec3(0.0f, 1.0f, 0.0f),
        -90.0f, -20.0f
    );
    appState.deltaTime = 0.0f;
    appState.lastFrame = 0.0f;
    appState.firstMouse = true;
    appState.lastX = 640.0f;
    appState.lastY = 360.0f;
    appState.wireframeMode = false;
    appState.showHeatmap = true;

    // Crear shaders
    Shader* basicShader = CreateShader("shaders/basic.vert", "shaders/basic.frag");
    if (!basicShader) {
        std::cerr << "Error al cargar shaders" << std::endl;
        return 1;
    }

    // Crear mallas
    std::cout << "Generando mallas..." << std::endl;
    Mesh* meshOriginal = CreateMeshFromSurface(superficie);
    Mesh* meshHeatmap = CreateMeshFromSurfaceWithHeatmap(superficie, depthBuffer, minDepth, maxDepth);
    std::cout << "Mallas generadas: " << meshOriginal->vertices.size() << " vertices, "
              << meshOriginal->indices.size() / 3 << " triangulos" << std::endl;

    SetVSync(true);

    std::cout << "\n=== Controles ===" << std::endl;
    std::cout << "WASD: Mover camara" << std::endl;
    std::cout << "Mouse: Rotar vista" << std::endl;
    std::cout << "Scroll: Zoom" << std::endl;
    std::cout << "Espacio: Subir" << std::endl;
    std::cout << "Shift: Bajar" << std::endl;
    std::cout << "F: Toggle wireframe" << std::endl;
    std::cout << "H: Toggle heatmap" << std::endl;
    std::cout << "ESC: Salir\n" << std::endl;

    // Loop principal
    while (!ShouldCloseVisualizer(viz)) {
        // Delta time
        float currentFrame = GetTime();
        appState.deltaTime = currentFrame - appState.lastFrame;
        appState.lastFrame = currentFrame;

        // Input
        processInput(viz);

        // Render
        ClearScreen(0.1f, 0.1f, 0.15f, 1.0f);

        // Usar shader
        UseShader(basicShader);

        // Configurar matrices
        glm::mat4 projection = glm::perspective(
            glm::radians(appState.camera->Zoom),
            1280.0f / 720.0f,
            0.1f, 100.0f
        );
        glm::mat4 view = GetViewMatrix(appState.camera);
        glm::mat4 model = glm::mat4(1.0f);

        SetMat4(basicShader, "projection", projection);
        SetMat4(basicShader, "view", view);
        SetMat4(basicShader, "model", model);

        // Configurar luces
        SetVec3v(basicShader, "lightPos", glm::vec3(10.0f, 10.0f, 10.0f));
        SetVec3v(basicShader, "viewPos", appState.camera->Position);
        SetVec3(basicShader, "lightColor", 1.0f, 1.0f, 1.0f);
        SetBool(basicShader, "useVertexColor", appState.showHeatmap);

        // Dibujar malla apropiada
        if (appState.showHeatmap) {
            DrawMesh(meshHeatmap);
        } else {
            DrawMesh(meshOriginal);
        }

        SwapBuffers(viz);
        PollEvents();
    }

    // Limpieza
    std::cout << "Cerrando..." << std::endl;
    DestroyMesh(meshOriginal);
    DestroyMesh(meshHeatmap);
    DestroyShader(basicShader);
    DestroyCamera(appState.camera);
    DestroyVisualizer(viz);
    BorrarSuperficie(&superficie);
    free(depthBuffer);

    std::cout << "Adios!" << std::endl;
    return 0;
}

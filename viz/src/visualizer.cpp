#include "../include/visualizer.h"
#include <iostream>
#include <cstdlib>

// Callbacks globales para GLFW
static Visualizer* g_currentViz = nullptr;

static void framebuffer_size_callback(GLFWwindow* window, int width, int height) {
    glViewport(0, 0, width, height);
    if (g_currentViz && g_currentViz->onResize) {
        g_currentViz->width = width;
        g_currentViz->height = height;
        g_currentViz->onResize(width, height);
    }
}

static void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods) {
    if (g_currentViz && g_currentViz->onKeyPress) {
        g_currentViz->onKeyPress(key, action);
    }

    // ESC para cerrar
    if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS) {
        glfwSetWindowShouldClose(window, true);
    }
}

static void cursor_position_callback(GLFWwindow* window, double xpos, double ypos) {
    if (g_currentViz && g_currentViz->onMouseMove) {
        g_currentViz->onMouseMove(xpos, ypos);
    }
}

static void scroll_callback(GLFWwindow* window, double xoffset, double yoffset) {
    if (g_currentViz && g_currentViz->onMouseScroll) {
        g_currentViz->onMouseScroll(xoffset, yoffset);
    }
}

Visualizer* CreateVisualizer(int width, int height, const char* title) {
    Visualizer* viz = new Visualizer();
    viz->width = width;
    viz->height = height;
    viz->title = title;
    viz->window = nullptr;
    viz->isRunning = false;
    viz->onResize = nullptr;
    viz->onKeyPress = nullptr;
    viz->onMouseMove = nullptr;
    viz->onMouseScroll = nullptr;

    return viz;
}

void DestroyVisualizer(Visualizer* viz) {
    if (viz) {
        if (viz->window) {
            glfwDestroyWindow(viz->window);
        }
        glfwTerminate();
        delete viz;
    }
}

bool InitVisualizer(Visualizer* viz) {
    if (!viz) return false;

    // Inicializar GLFW
    if (!glfwInit()) {
        std::cerr << "Error: No se pudo inicializar GLFW" << std::endl;
        return false;
    }

    // Configurar OpenGL version (4.5 Core)
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 5);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_SAMPLES, 4); // Anti-aliasing

#ifdef __APPLE__
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif

    // Crear ventana
    viz->window = glfwCreateWindow(viz->width, viz->height, viz->title, nullptr, nullptr);
    if (!viz->window) {
        std::cerr << "Error: No se pudo crear la ventana GLFW" << std::endl;
        glfwTerminate();
        return false;
    }

    glfwMakeContextCurrent(viz->window);

    // Cargar funciones OpenGL con GLAD
    if (!gladLoadGL((GLADloadfunc)glfwGetProcAddress)) {
        std::cerr << "Error: No se pudo inicializar GLAD" << std::endl;
        return false;
    }

    // Configurar viewport
    glViewport(0, 0, viz->width, viz->height);

    // Configurar callbacks
    g_currentViz = viz;
    glfwSetFramebufferSizeCallback(viz->window, framebuffer_size_callback);
    glfwSetKeyCallback(viz->window, key_callback);
    glfwSetCursorPosCallback(viz->window, cursor_position_callback);
    glfwSetScrollCallback(viz->window, scroll_callback);

    // Configurar OpenGL
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_MULTISAMPLE); // Anti-aliasing

    // Opcional: capturar el cursor para controles FPS
    // glfwSetInputMode(viz->window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);

    viz->isRunning = true;

    std::cout << "OpenGL Version: " << glGetString(GL_VERSION) << std::endl;
    std::cout << "GLSL Version: " << glGetString(GL_SHADING_LANGUAGE_VERSION) << std::endl;
    std::cout << "Renderer: " << glGetString(GL_RENDERER) << std::endl;

    return true;
}

void UpdateVisualizer(Visualizer* viz) {
    if (!viz || !viz->window) return;
    // Aquí se pueden añadir actualizaciones por frame si es necesario
}

bool ShouldCloseVisualizer(Visualizer* viz) {
    if (!viz || !viz->window) return true;
    return glfwWindowShouldClose(viz->window);
}

void ClearScreen(float r, float g, float b, float a) {
    glClearColor(r, g, b, a);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
}

void SwapBuffers(Visualizer* viz) {
    if (viz && viz->window) {
        glfwSwapBuffers(viz->window);
    }
}

void PollEvents() {
    glfwPollEvents();
}

double GetTime() {
    return glfwGetTime();
}

void SetVSync(bool enabled) {
    glfwSwapInterval(enabled ? 1 : 0);
}

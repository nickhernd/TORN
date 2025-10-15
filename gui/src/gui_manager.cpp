#include "../include/gui_manager.h"
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>
#include <GLFW/glfw3.h>
#include <iostream>
#include <fstream>

void InitGUI(GLFWwindow* window) {
    // Setup Dear ImGui context
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
    io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;

    // Setup style
    ApplyDarkTheme();

    // Setup Platform/Renderer backends
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init("#version 450");

    std::cout << "ImGui inicializado correctamente" << std::endl;
}

void ShutdownGUI() {
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
}

void BeginGUIFrame() {
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();
}

void EndGUIFrame() {
    ImGui::Render();
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
}

void RenderMainPanel(GUIState* state) {
    if (!state->showMainPanel) return;

    ImGui::SetNextWindowPos(ImVec2(10, 10), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(ImVec2(350, 500), ImGuiCond_FirstUseEver);

    ImGui::Begin("Control Panel", &state->showMainPanel);

    // Sección de archivo
    if (ImGui::CollapsingHeader("Archivo", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::Text("Archivo actual:");
        ImGui::TextWrapped("%s", state->currentFile);

        if (ImGui::Button("Cargar archivo...", ImVec2(-1, 0))) {
            state->showFileDialog = true;
        }

        if (ImGui::TreeNode("Archivos recientes")) {
            for (const auto& file : state->recentFiles) {
                if (ImGui::Selectable(file.c_str())) {
                    strncpy(state->currentFile, file.c_str(), sizeof(state->currentFile) - 1);
                    // Aquí se debería recargar el archivo
                }
            }
            ImGui::TreePop();
        }
    }

    ImGui::Separator();

    // Sección de visualización
    if (ImGui::CollapsingHeader("Visualización", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::Checkbox("Heatmap", &state->useHeatmap);
        ImGui::SameLine();
        ImGui::Checkbox("Wireframe", &state->wireframeMode);

        ImGui::Checkbox("Mostrar normales", &state->showNormals);

        ImGui::Text("Color de fondo:");
        ImGui::ColorEdit3("##bgcolor", state->backgroundColor);

        ImGui::Text("Posición de luz:");
        ImGui::SliderFloat3("##lightpos", state->lightPosition, -20.0f, 20.0f);

        ImGui::Text("Color de luz:");
        ImGui::ColorEdit3("##lightcolor", state->lightColor);
    }

    ImGui::Separator();

    // Sección de cámara
    if (ImGui::CollapsingHeader("Cámara")) {
        ImGui::SliderFloat("Velocidad", &state->cameraSpeed, 0.5f, 10.0f);
        ImGui::SliderFloat("Sensibilidad", &state->mouseSensitivity, 0.01f, 0.5f);
        ImGui::SliderFloat("FOV", &state->fov, 30.0f, 120.0f);

        if (ImGui::Button("Reset Cámara")) {
            // Aquí se resetearía la cámara
        }
    }

    ImGui::Separator();

    // Sección de animación
    if (ImGui::CollapsingHeader("Animación")) {
        ImGui::Checkbox("Auto-rotar", &state->autoRotate);
        if (state->autoRotate) {
            ImGui::SliderFloat("Velocidad", &state->rotationSpeed, 0.1f, 5.0f);
        }

        ImGui::Checkbox("Animar simulación", &state->animateSimulation);
        if (state->animateSimulation) {
            ImGui::SliderFloat("Progreso", &state->animationProgress, 0.0f, 1.0f);
        }
    }

    ImGui::Separator();

    // Botones de ventanas
    if (ImGui::Button("Métricas", ImVec2(-1, 0))) {
        state->showMetrics = !state->showMetrics;
    }

    if (ImGui::Button("Configuración", ImVec2(-1, 0))) {
        state->showSettings = !state->showSettings;
    }

    if (ImGui::Button("Acerca de", ImVec2(-1, 0))) {
        state->showAbout = !state->showAbout;
    }

    ImGui::End();
}

void RenderMetricsPanel(GUIState* state) {
    if (!state->showMetrics) return;

    ImGui::SetNextWindowPos(ImVec2(370, 10), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(ImVec2(300, 400), ImGuiCond_FirstUseEver);

    ImGui::Begin("Métricas", &state->showMetrics);

    // FPS
    ImGui::Text("Rendimiento");
    ImGui::Separator();
    ImGui::Text("FPS: %.1f", state->fps);
    ImGui::Text("Frame Time: %.3f ms", state->frameTime);

    ImGui::Spacing();

    // Información de la superficie
    ImGui::Text("Superficie");
    ImGui::Separator();
    ImGui::Text("Puntos: %d", state->surfacePoints);
    ImGui::Text("Triángulos: %d", state->surfaceTriangles);

    ImGui::Spacing();

    // Información de simulación
    ImGui::Text("Simulación CUDA");
    ImGui::Separator();
    ImGui::Text("Tiempo CPU: %.3f ms", state->cpuTime * 1000.0f);
    ImGui::Text("Tiempo GPU: %.3f ms", state->gpuTime * 1000.0f);
    ImGui::Text("Speedup: %.1fx", state->speedup);

    ImGui::Spacing();

    // Profundidad
    ImGui::Text("Profundidad de Corte");
    ImGui::Separator();
    ImGui::Text("Mínima: %.3f", state->minDepth);
    ImGui::Text("Máxima: %.3f", state->maxDepth);
    ImGui::Text("Rango: %.3f", state->maxDepth - state->minDepth);

    ImGui::Spacing();

    // Gráfico de FPS
    static float fps_history[90] = {};
    static int fps_index = 0;
    fps_history[fps_index] = state->fps;
    fps_index = (fps_index + 1) % 90;

    ImGui::PlotLines("##fps", fps_history, 90, 0, "FPS History", 0.0f, 120.0f, ImVec2(0, 80));

    ImGui::End();
}

void RenderSettingsPanel(GUIState* state) {
    if (!state->showSettings) return;

    ImGui::SetNextWindowPos(ImVec2(680, 10), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(ImVec2(300, 300), ImGuiCond_FirstUseEver);

    ImGui::Begin("Configuración", &state->showSettings);

    if (ImGui::CollapsingHeader("Render", ImGuiTreeNodeFlags_DefaultOpen)) {
        static bool vsync = true;
        if (ImGui::Checkbox("VSync", &vsync)) {
            glfwSwapInterval(vsync ? 1 : 0);
        }

        static bool msaa = true;
        ImGui::Checkbox("Anti-aliasing (MSAA)", &msaa);

        static int shadowQuality = 1;
        ImGui::Combo("Calidad de sombras", &shadowQuality, "Baja\0Media\0Alta\0\0");
    }

    if (ImGui::CollapsingHeader("GPU")) {
        ImGui::Text("Dispositivo: NVIDIA GPU");
        ImGui::Text("Arquitectura: sm_75");

        static int blockSize = 16;
        ImGui::SliderInt("Block Size", &blockSize, 8, 32);
    }

    if (ImGui::CollapsingHeader("Interfaz")) {
        static float scale = 1.0f;
        if (ImGui::SliderFloat("Escala UI", &scale, 0.5f, 2.0f)) {
            ImGui::GetIO().FontGlobalScale = scale;
        }

        if (ImGui::Button("Guardar configuración")) {
            SaveGUIConfig("config.ini");
        }

        ImGui::SameLine();

        if (ImGui::Button("Cargar configuración")) {
            LoadGUIConfig("config.ini");
        }
    }

    ImGui::End();
}

void RenderFileDialog(GUIState* state) {
    if (!state->showFileDialog) return;

    ImGui::SetNextWindowPos(ImVec2(400, 200), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(ImVec2(500, 300), ImGuiCond_FirstUseEver);

    ImGui::Begin("Seleccionar Archivo", &state->showFileDialog);

    ImGui::Text("Archivos .for disponibles:");
    ImGui::Separator();

    // Lista de archivos (simulada)
    const char* files[] = {
        "ejemplo.for",
        "ejemplo_big.for",
        "ejemplo_extraBig.for"
    };

    for (int i = 0; i < 3; i++) {
        if (ImGui::Selectable(files[i], false)) {
            strncpy(state->currentFile, files[i], sizeof(state->currentFile) - 1);
            state->showFileDialog = false;

            // Agregar a recientes
            bool found = false;
            for (const auto& f : state->recentFiles) {
                if (f == files[i]) {
                    found = true;
                    break;
                }
            }
            if (!found && state->recentFiles.size() < 5) {
                state->recentFiles.push_back(files[i]);
            }
        }
    }

    ImGui::Separator();

    if (ImGui::Button("Cancelar", ImVec2(120, 0))) {
        state->showFileDialog = false;
    }

    ImGui::End();
}

void RenderAboutPanel(GUIState* state) {
    if (!state->showAbout) return;

    ImGui::SetNextWindowPos(ImVec2(450, 250), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(ImVec2(400, 300), ImGuiCond_FirstUseEver);

    ImGui::Begin("Acerca de TORN", &state->showAbout);

    ImGui::Text("TORN Visualizer");
    ImGui::Separator();

    ImGui::Text("Version: 1.0.0");
    ImGui::Text("Fecha: Octubre 2025");

    ImGui::Spacing();
    ImGui::TextWrapped("Sistema de visualización 3D para simulaciones de torneado CNC aceleradas por GPU.");

    ImGui::Spacing();
    ImGui::Text("Tecnologías:");
    ImGui::BulletText("CUDA - Simulación paralela en GPU");
    ImGui::BulletText("OpenGL 4.5 - Renderizado 3D");
    ImGui::BulletText("GLFW - Sistema de ventanas");
    ImGui::BulletText("GLM - Matemáticas 3D");
    ImGui::BulletText("Dear ImGui - Interfaz gráfica");

    ImGui::Spacing();
    ImGui::Separator();

    if (ImGui::Button("Cerrar", ImVec2(-1, 0))) {
        state->showAbout = false;
    }

    ImGui::End();
}

void ApplyDarkTheme() {
    ImVec4* colors = ImGui::GetStyle().Colors;

    colors[ImGuiCol_Text]                   = ImVec4(1.00f, 1.00f, 1.00f, 1.00f);
    colors[ImGuiCol_TextDisabled]           = ImVec4(0.50f, 0.50f, 0.50f, 1.00f);
    colors[ImGuiCol_WindowBg]               = ImVec4(0.10f, 0.10f, 0.10f, 0.94f);
    colors[ImGuiCol_ChildBg]                = ImVec4(0.00f, 0.00f, 0.00f, 0.00f);
    colors[ImGuiCol_PopupBg]                = ImVec4(0.19f, 0.19f, 0.19f, 0.92f);
    colors[ImGuiCol_Border]                 = ImVec4(0.19f, 0.19f, 0.19f, 0.29f);
    colors[ImGuiCol_BorderShadow]           = ImVec4(0.00f, 0.00f, 0.00f, 0.24f);
    colors[ImGuiCol_FrameBg]                = ImVec4(0.05f, 0.05f, 0.05f, 0.54f);
    colors[ImGuiCol_FrameBgHovered]         = ImVec4(0.19f, 0.19f, 0.19f, 0.54f);
    colors[ImGuiCol_FrameBgActive]          = ImVec4(0.20f, 0.22f, 0.23f, 1.00f);
    colors[ImGuiCol_TitleBg]                = ImVec4(0.00f, 0.00f, 0.00f, 1.00f);
    colors[ImGuiCol_TitleBgActive]          = ImVec4(0.06f, 0.06f, 0.06f, 1.00f);
    colors[ImGuiCol_TitleBgCollapsed]       = ImVec4(0.00f, 0.00f, 0.00f, 1.00f);
    colors[ImGuiCol_MenuBarBg]              = ImVec4(0.14f, 0.14f, 0.14f, 1.00f);
    colors[ImGuiCol_ScrollbarBg]            = ImVec4(0.05f, 0.05f, 0.05f, 0.54f);
    colors[ImGuiCol_ScrollbarGrab]          = ImVec4(0.34f, 0.34f, 0.34f, 0.54f);
    colors[ImGuiCol_ScrollbarGrabHovered]   = ImVec4(0.40f, 0.40f, 0.40f, 0.54f);
    colors[ImGuiCol_ScrollbarGrabActive]    = ImVec4(0.56f, 0.56f, 0.56f, 0.54f);
    colors[ImGuiCol_CheckMark]              = ImVec4(0.33f, 0.67f, 0.86f, 1.00f);
    colors[ImGuiCol_SliderGrab]             = ImVec4(0.34f, 0.34f, 0.34f, 0.54f);
    colors[ImGuiCol_SliderGrabActive]       = ImVec4(0.56f, 0.56f, 0.56f, 0.54f);
    colors[ImGuiCol_Button]                 = ImVec4(0.05f, 0.05f, 0.05f, 0.54f);
    colors[ImGuiCol_ButtonHovered]          = ImVec4(0.19f, 0.19f, 0.19f, 0.54f);
    colors[ImGuiCol_ButtonActive]           = ImVec4(0.20f, 0.22f, 0.23f, 1.00f);
    colors[ImGuiCol_Header]                 = ImVec4(0.00f, 0.00f, 0.00f, 0.52f);
    colors[ImGuiCol_HeaderHovered]          = ImVec4(0.00f, 0.00f, 0.00f, 0.36f);
    colors[ImGuiCol_HeaderActive]           = ImVec4(0.20f, 0.22f, 0.23f, 0.33f);
    colors[ImGuiCol_Separator]              = ImVec4(0.28f, 0.28f, 0.28f, 0.29f);
    colors[ImGuiCol_SeparatorHovered]       = ImVec4(0.44f, 0.44f, 0.44f, 0.29f);
    colors[ImGuiCol_SeparatorActive]        = ImVec4(0.40f, 0.44f, 0.47f, 1.00f);
    colors[ImGuiCol_ResizeGrip]             = ImVec4(0.28f, 0.28f, 0.28f, 0.29f);
    colors[ImGuiCol_ResizeGripHovered]      = ImVec4(0.44f, 0.44f, 0.44f, 0.29f);
    colors[ImGuiCol_ResizeGripActive]       = ImVec4(0.40f, 0.44f, 0.47f, 1.00f);
    colors[ImGuiCol_Tab]                    = ImVec4(0.00f, 0.00f, 0.00f, 0.52f);
    colors[ImGuiCol_TabHovered]             = ImVec4(0.14f, 0.14f, 0.14f, 1.00f);
    colors[ImGuiCol_TabActive]              = ImVec4(0.20f, 0.20f, 0.20f, 0.36f);
    colors[ImGuiCol_TabUnfocused]           = ImVec4(0.00f, 0.00f, 0.00f, 0.52f);
    colors[ImGuiCol_TabUnfocusedActive]     = ImVec4(0.14f, 0.14f, 0.14f, 1.00f);
    colors[ImGuiCol_PlotLines]              = ImVec4(1.00f, 0.00f, 0.00f, 1.00f);
    colors[ImGuiCol_PlotLinesHovered]       = ImVec4(1.00f, 0.00f, 0.00f, 1.00f);
    colors[ImGuiCol_PlotHistogram]          = ImVec4(1.00f, 0.00f, 0.00f, 1.00f);
    colors[ImGuiCol_PlotHistogramHovered]   = ImVec4(1.00f, 0.00f, 0.00f, 1.00f);
    colors[ImGuiCol_TableHeaderBg]          = ImVec4(0.00f, 0.00f, 0.00f, 0.52f);
    colors[ImGuiCol_TableBorderStrong]      = ImVec4(0.00f, 0.00f, 0.00f, 0.52f);
    colors[ImGuiCol_TableBorderLight]       = ImVec4(0.28f, 0.28f, 0.28f, 0.29f);
    colors[ImGuiCol_TableRowBg]             = ImVec4(0.00f, 0.00f, 0.00f, 0.00f);
    colors[ImGuiCol_TableRowBgAlt]          = ImVec4(1.00f, 1.00f, 1.00f, 0.06f);
    colors[ImGuiCol_TextSelectedBg]         = ImVec4(0.20f, 0.22f, 0.23f, 1.00f);
    colors[ImGuiCol_DragDropTarget]         = ImVec4(0.33f, 0.67f, 0.86f, 1.00f);
    colors[ImGuiCol_NavHighlight]           = ImVec4(1.00f, 0.00f, 0.00f, 1.00f);
    colors[ImGuiCol_NavWindowingHighlight]  = ImVec4(1.00f, 0.00f, 0.00f, 0.70f);
    colors[ImGuiCol_NavWindowingDimBg]      = ImVec4(1.00f, 0.00f, 0.00f, 0.20f);
    colors[ImGuiCol_ModalWindowDimBg]       = ImVec4(1.00f, 0.00f, 0.00f, 0.35f);

    ImGuiStyle& style = ImGui::GetStyle();
    style.WindowPadding                     = ImVec2(8.00f, 8.00f);
    style.FramePadding                      = ImVec2(5.00f, 2.00f);
    style.CellPadding                       = ImVec2(6.00f, 6.00f);
    style.ItemSpacing                       = ImVec2(6.00f, 6.00f);
    style.ItemInnerSpacing                  = ImVec2(6.00f, 6.00f);
    style.TouchExtraPadding                 = ImVec2(0.00f, 0.00f);
    style.IndentSpacing                     = 25;
    style.ScrollbarSize                     = 15;
    style.GrabMinSize                       = 10;
    style.WindowBorderSize                  = 1;
    style.ChildBorderSize                   = 1;
    style.PopupBorderSize                   = 1;
    style.FrameBorderSize                   = 1;
    style.TabBorderSize                     = 1;
    style.WindowRounding                    = 7;
    style.ChildRounding                     = 4;
    style.FrameRounding                     = 3;
    style.PopupRounding                     = 4;
    style.ScrollbarRounding                 = 9;
    style.GrabRounding                      = 3;
    style.LogSliderDeadzone                 = 4;
    style.TabRounding                       = 4;
}

void SaveGUIConfig(const char* filename) {
    std::ofstream file(filename);
    if (file.is_open()) {
        file << "# TORN GUI Config\n";
        // Aquí se guardarían los parámetros
        file.close();
        std::cout << "Configuración guardada: " << filename << std::endl;
    }
}

void LoadGUIConfig(const char* filename) {
    std::ifstream file(filename);
    if (file.is_open()) {
        // Aquí se cargarían los parámetros
        file.close();
        std::cout << "Configuración cargada: " << filename << std::endl;
    }
}

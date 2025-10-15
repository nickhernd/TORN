# Compiladores
NVCC = nvcc
CXX = g++

# Flags
NVCC_FLAGS = -arch=sm_75 -O3
CXX_FLAGS = -std=c++17 -O3 -Wall
INCLUDE_FLAGS = -I. -Iexternal/glad/include -Iviz/include -Igui/include -Iexternal/imgui -Iexternal/imgui/backends
LIBS = -lglfw -lGL -lm -ldl -lpthread

# Ejecutables
TARGET_ORIGINAL = simutorno
TARGET_VIZ = torn-viz
TARGET_GUI = torn-gui

# Archivos fuente originales
CUDA_SRCS = simutorno.cu
CUDA_OBJS = simutorno.o

# Archivos fuente del visualizador
VIZ_SRCS = main_viz.cpp viz/src/visualizer.cpp viz/src/shader.cpp viz/src/mesh.cpp viz/src/camera.cpp
GLAD_SRC = external/glad/src/gl.c
VIZ_OBJS = main_viz.o visualizer.o shader.o mesh.o camera.o glad.o

# Archivos fuente de ImGui
IMGUI_DIR = external/imgui
IMGUI_SRCS = $(IMGUI_DIR)/imgui.cpp $(IMGUI_DIR)/imgui_demo.cpp $(IMGUI_DIR)/imgui_draw.cpp \
             $(IMGUI_DIR)/imgui_tables.cpp $(IMGUI_DIR)/imgui_widgets.cpp \
             $(IMGUI_DIR)/backends/imgui_impl_glfw.cpp $(IMGUI_DIR)/backends/imgui_impl_opengl3.cpp
IMGUI_OBJS = imgui.o imgui_demo.o imgui_draw.o imgui_tables.o imgui_widgets.o \
             imgui_impl_glfw.o imgui_impl_opengl3.o

# Archivos fuente de la GUI
GUI_SRCS = gui/src/gui_manager.cpp
GUI_OBJS = gui_manager.o

# Todos los objetos del visualizador
ALL_VIZ_OBJS = $(VIZ_OBJS) $(IMGUI_OBJS) $(GUI_OBJS)

# Regla principal: compilar todos
all: $(TARGET_ORIGINAL) $(TARGET_VIZ) $(TARGET_GUI)

# Compilar solo el simulador original
original: $(TARGET_ORIGINAL)

# Compilar solo el visualizador básico
viz: $(TARGET_VIZ)

# Compilar solo el visualizador con GUI
gui: $(TARGET_GUI)

# Ejecutable original
$(TARGET_ORIGINAL): $(CUDA_OBJS)
	$(NVCC) $(NVCC_FLAGS) -o $@ $^

# Ejecutable visualizador básico (CUDA + OpenGL)
$(TARGET_VIZ): $(VIZ_OBJS) $(CUDA_OBJS)
	$(NVCC) $(NVCC_FLAGS) -o $@ $^ $(LIBS)

# Ejecutable visualizador con GUI (CUDA + OpenGL + ImGui)
$(TARGET_GUI): main_viz_gui.o $(VIZ_OBJS) $(IMGUI_OBJS) $(GUI_OBJS) $(CUDA_OBJS)
	$(NVCC) $(NVCC_FLAGS) -o $@ $^ $(LIBS)

# Compilar archivos CUDA
%.o: %.cu
	$(NVCC) $(NVCC_FLAGS) $(INCLUDE_FLAGS) -c $< -o $@

# Compilar archivos C++ del visualizador
main_viz.o: main_viz.cpp
	$(CXX) $(CXX_FLAGS) $(INCLUDE_FLAGS) -c $< -o $@

main_viz_gui.o: main_viz_gui.cpp
	$(CXX) $(CXX_FLAGS) $(INCLUDE_FLAGS) -c $< -o $@

visualizer.o: viz/src/visualizer.cpp
	$(CXX) $(CXX_FLAGS) $(INCLUDE_FLAGS) -c $< -o $@

shader.o: viz/src/shader.cpp
	$(CXX) $(CXX_FLAGS) $(INCLUDE_FLAGS) -c $< -o $@

mesh.o: viz/src/mesh.cpp
	$(CXX) $(CXX_FLAGS) $(INCLUDE_FLAGS) -c $< -o $@

camera.o: viz/src/camera.cpp
	$(CXX) $(CXX_FLAGS) $(INCLUDE_FLAGS) -c $< -o $@

# Compilar GLAD (C)
glad.o: $(GLAD_SRC)
	$(CXX) $(CXX_FLAGS) $(INCLUDE_FLAGS) -c $< -o $@

# Compilar archivos de ImGui
imgui.o: $(IMGUI_DIR)/imgui.cpp
	$(CXX) $(CXX_FLAGS) $(INCLUDE_FLAGS) -c $< -o $@

imgui_demo.o: $(IMGUI_DIR)/imgui_demo.cpp
	$(CXX) $(CXX_FLAGS) $(INCLUDE_FLAGS) -c $< -o $@

imgui_draw.o: $(IMGUI_DIR)/imgui_draw.cpp
	$(CXX) $(CXX_FLAGS) $(INCLUDE_FLAGS) -c $< -o $@

imgui_tables.o: $(IMGUI_DIR)/imgui_tables.cpp
	$(CXX) $(CXX_FLAGS) $(INCLUDE_FLAGS) -c $< -o $@

imgui_widgets.o: $(IMGUI_DIR)/imgui_widgets.cpp
	$(CXX) $(CXX_FLAGS) $(INCLUDE_FLAGS) -c $< -o $@

imgui_impl_glfw.o: $(IMGUI_DIR)/backends/imgui_impl_glfw.cpp
	$(CXX) $(CXX_FLAGS) $(INCLUDE_FLAGS) -c $< -o $@

imgui_impl_opengl3.o: $(IMGUI_DIR)/backends/imgui_impl_opengl3.cpp
	$(CXX) $(CXX_FLAGS) $(INCLUDE_FLAGS) -c $< -o $@

# Compilar archivos de GUI
gui_manager.o: gui/src/gui_manager.cpp
	$(CXX) $(CXX_FLAGS) $(INCLUDE_FLAGS) -c $< -o $@

# Limpiar
clean:
	rm -f $(CUDA_OBJS) $(ALL_VIZ_OBJS) main_viz_gui.o $(TARGET_ORIGINAL) $(TARGET_VIZ) $(TARGET_GUI)

# Ejecutar el simulador original
run-original: $(TARGET_ORIGINAL)
	./$(TARGET_ORIGINAL) ejemplo.for

# Ejecutar el visualizador básico
run-viz: $(TARGET_VIZ)
	./$(TARGET_VIZ) ejemplo.for

# Ejecutar el visualizador con GUI
run-gui: $(TARGET_GUI)
	./$(TARGET_GUI) ejemplo.for

# Mostrar ayuda
help:
	@echo "Targets disponibles:"
	@echo "  all          - Compilar todos los programas"
	@echo "  original     - Compilar solo el simulador original"
	@echo "  viz          - Compilar solo el visualizador 3D básico"
	@echo "  gui          - Compilar el visualizador 3D con GUI (ImGui)"
	@echo "  run-original - Ejecutar simulador original"
	@echo "  run-viz      - Ejecutar visualizador 3D básico"
	@echo "  run-gui      - Ejecutar visualizador 3D con GUI (RECOMENDADO)"
	@echo "  clean        - Limpiar archivos compilados"
	@echo ""
	@echo "Antes de compilar el visualizador, instala las dependencias:"
	@echo "  sudo apt-get install libglfw3-dev libglm-dev libgl1-mesa-dev"

.PHONY: all original viz gui clean run-original run-viz run-gui help

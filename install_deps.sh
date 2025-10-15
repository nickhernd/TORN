#!/bin/bash
# Script de instalación de dependencias para TORN Visualizer

echo "Instalando dependencias para visualización 3D..."

# Actualizar repositorios
sudo apt-get update

# Instalar GLFW3
echo "Instalando GLFW3..."
sudo apt-get install -y libglfw3-dev

# Instalar GLM
echo "Instalando GLM (matemáticas OpenGL)..."
sudo apt-get install -y libglm-dev

# Instalar otras dependencias OpenGL
echo "Instalando dependencias OpenGL adicionales..."
sudo apt-get install -y libgl1-mesa-dev libglu1-mesa-dev

echo "Dependencias instaladas correctamente!"
echo ""
echo "Nota: GLAD se incluirá directamente en el proyecto"

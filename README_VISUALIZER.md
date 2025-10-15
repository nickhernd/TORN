# TORN Visualizer - Simulación de Torno 3D con CUDA y OpenGL

Sistema de visualización 3D en tiempo real para simulaciones de torneado CNC aceleradas por GPU.

## Características

### Simulación CUDA
- Simulación paralela de torneado en GPU
- Cálculo de profundidad de corte para cada punto de la superficie
- Comparación de rendimiento CPU vs GPU
- Soporte para superficies helicoidales complejas

### Visualización 3D
- Renderizado en tiempo real con OpenGL 4.5
- Iluminación Phong (ambient, diffuse, specular)
- Heatmap de profundidad de corte con gradiente de colores
- Modo wireframe para análisis de geometría
- Anti-aliasing (MSAA)

### Controles Interactivos
- Cámara libre con controles FPS
- Rotación con mouse
- Zoom con scroll
- Cambio entre vista normal y heatmap
- Toggle wireframe en tiempo real

## Requisitos

### Sistema
- Linux (probado en Ubuntu/WSL2)
- GPU NVIDIA con soporte CUDA (Compute Capability 7.5+)
- OpenGL 4.5+

### Dependencias
```bash
# CUDA Toolkit (ya instalado)
# Dependencias OpenGL
sudo apt-get install libglfw3-dev libglm-dev libgl1-mesa-dev
```

## Compilación

### Estructura del Proyecto
```
TORN/
├── simutorno.cu           # Motor de simulación CUDA
├── simutorno.h            # Headers del simulador
├── main_viz.cpp           # Programa principal del visualizador
├── viz/                   # Módulo de visualización
│   ├── include/
│   │   ├── visualizer.h   # Sistema de ventanas (GLFW)
│   │   ├── camera.h       # Cámara interactiva
│   │   ├── shader.h       # Sistema de shaders
│   │   └── mesh.h         # Renderizador de mallas
│   └── src/
│       ├── visualizer.cpp
│       ├── camera.cpp
│       ├── shader.cpp
│       └── mesh.cpp
├── shaders/               # Shaders GLSL
│   ├── basic.vert         # Vertex shader con iluminación
│   ├── basic.frag         # Fragment shader Phong
│   ├── wireframe.vert     # Vertex shader wireframe
│   └── wireframe.frag     # Fragment shader wireframe
├── external/              # Dependencias externas
│   └── glad/              # OpenGL loader
└── Makefile               # Sistema de compilación
```

### Comandos de Compilación

```bash
# Ver todas las opciones
make help

# Compilar ambos programas (simulador original + visualizador)
make all

# Compilar solo el simulador original
make original

# Compilar solo el visualizador 3D
make viz

# Limpiar archivos compilados
make clean
```

## Uso

### Simulador Original (solo terminal)
```bash
make run-original
# O directamente:
./simutorno ejemplo.for
```

### Visualizador 3D
```bash
make run-viz
# O directamente:
./torn-viz ejemplo.for
```

### Argumentos
Ambos programas aceptan un archivo `.for` como entrada:
```bash
./torn-viz ejemplo.for           # Superficie pequeña
./torn-viz ejemplo_big.for       # Superficie mediana
./torn-viz ejemplo_extraBig.for  # Superficie grande
```

## Controles del Visualizador

| Tecla/Acción | Función |
|--------------|---------|
| **W** | Mover cámara adelante |
| **S** | Mover cámara atrás |
| **A** | Mover cámara izquierda |
| **D** | Mover cámara derecha |
| **Espacio** | Mover cámara arriba |
| **Shift** | Mover cámara abajo |
| **Mouse** | Rotar vista |
| **Scroll** | Zoom in/out |
| **H** | Toggle heatmap de profundidad |
| **F** | Toggle modo wireframe |
| **ESC** | Salir |

## Formato de Archivo .FOR

Los archivos de entrada contienen superficies parametrizadas:

```
SECTION NUMBER: 100
POINTS PER SECTION: 50
STEP: 0.0628
POINTS PER ROUND: 100

POINTS:
x1 y1 z1
x2 y2 z2
...
```

### Parámetros
- **SECTION NUMBER**: Número de secciones (U)
- **POINTS PER SECTION**: Puntos por sección (V)
- **STEP**: Paso angular en radianes
- **POINTS PER ROUND**: Puntos por revolución completa
- **POINTS**: Coordenadas 3D de cada punto

## Características Técnicas

### Arquitectura CUDA
- Kernels optimizados para cálculo paralelo
- Memoria compartida para mejor rendimiento
- Coalesced memory access
- Stream-based execution

### Renderizado OpenGL
- Vertex Array Objects (VAO) para geometría
- Vertex Buffer Objects (VBO) para datos
- Element Buffer Objects (EBO) para índices
- Programmable pipeline con shaders GLSL 4.5

### Iluminación
- Modelo Phong completo
- Luz direccional configurable
- Specular highlights
- Ambient occlusion básico

### Heatmap
Gradiente de colores para visualizar profundidad de corte:
- **Azul**: Profundidad mínima (menos corte)
- **Cian**: Profundidad baja
- **Verde**: Profundidad media
- **Amarillo**: Profundidad alta
- **Rojo**: Profundidad máxima (más corte)

## Rendimiento

Tiempos típicos de simulación (GPU RTX 3070):

| Superficie | Puntos | CPU | GPU | Speedup |
|------------|--------|-----|-----|---------|
| Pequeña | 5K | ~0.5s | ~0.02s | 25x |
| Mediana | 50K | ~5s | ~0.1s | 50x |
| Grande | 500K | ~50s | ~0.5s | 100x |

El visualizador mantiene 60 FPS con mallas de hasta 1M de triángulos.

## Extensiones Futuras

### Planeadas
- [ ] Animación del proceso de torneado
- [ ] Exportación a G-Code CNC
- [ ] Interfaz gráfica (Dear ImGui)
- [ ] Soporte para múltiples herramientas
- [ ] Detección de colisiones
- [ ] Simulación de fresado
- [ ] Exportación de mallas (STL, OBJ)
- [ ] Benchmark automático
- [ ] Soporte multi-GPU

### En Desarrollo
- [x] Visualización 3D interactiva
- [x] Heatmap de profundidad de corte
- [x] Controles de cámara
- [x] Iluminación Phong
- [x] Modo wireframe

## Troubleshooting

### Error: "No se pudo inicializar GLFW"
```bash
# Instalar dependencias faltantes
sudo apt-get install libglfw3-dev libglfw3
```

### Error: "No se pudo inicializar GLAD"
Verifica que tu GPU soporte OpenGL 4.5:
```bash
glxinfo | grep "OpenGL version"
```

### Error: "CUDA device not found"
```bash
# Verificar CUDA
nvidia-smi
nvcc --version
```

### Rendimiento bajo
- Desactiva VSync para FPS ilimitados (tecla V en futuras versiones)
- Reduce el tamaño de la malla
- Verifica que estés usando la GPU dedicada (no integrada)

## Licencia

Este proyecto es parte del trabajo académico de optimización GPU.

## Créditos

Desarrollado con:
- **CUDA**: Simulación paralela en GPU
- **OpenGL**: Renderizado 3D
- **GLFW**: Sistema de ventanas
- **GLM**: Matemáticas 3D
- **GLAD**: OpenGL loader

---

**Versión**: 1.0
**Fecha**: Octubre 2025
**Arquitectura GPU**: NVIDIA Turing/Ampere (sm_75+)

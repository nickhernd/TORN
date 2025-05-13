# Nombre del ejecutable
TARGET = main

# Archivos fuente
SRCS = $(wildcard *.cu)
OBJS = $(SRCS:.cu=.o)

# Compilador de CUDA
NVCC = nvcc

# Flags de compilación
NVCC_FLAGS = -arch=sm_75 -O3

# Reglas principales
all: $(TARGET)

$(TARGET): $(OBJS)
	$(NVCC) $(NVCC_FLAGS) -o $@ $^

%.o: %.cu
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

# Limpiar archivos objetos y ejecutables
clean:
	rm -f $(OBJS) $(TARGET)

# Ejecutar el programa
run: $(TARGET)
	./$(TARGET)

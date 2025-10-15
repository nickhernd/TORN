#include "../include/shader.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <glm/gtc/type_ptr.hpp>

// Leer archivo de shader
std::string ReadShaderFile(const char* path) {
    std::string code;
    std::ifstream file;

    file.exceptions(std::ifstream::failbit | std::ifstream::badbit);
    try {
        file.open(path);
        std::stringstream stream;
        stream << file.rdbuf();
        file.close();
        code = stream.str();
    }
    catch (std::ifstream::failure& e) {
        std::cerr << "Error: No se pudo leer el archivo de shader: " << path << std::endl;
    }

    return code;
}

Shader* CreateShader(const char* vertexPath, const char* fragmentPath) {
    std::string vertexCode = ReadShaderFile(vertexPath);
    std::string fragmentCode = ReadShaderFile(fragmentPath);

    return CreateShaderFromSource(vertexCode.c_str(), fragmentCode.c_str());
}

Shader* CreateShaderFromSource(const char* vertexSource, const char* fragmentSource) {
    Shader* shader = new Shader();

    // Compilar vertex shader
    unsigned int vertex = CompileShader(GL_VERTEX_SHADER, vertexSource);
    if (vertex == 0) {
        delete shader;
        return nullptr;
    }

    // Compilar fragment shader
    unsigned int fragment = CompileShader(GL_FRAGMENT_SHADER, fragmentSource);
    if (fragment == 0) {
        glDeleteShader(vertex);
        delete shader;
        return nullptr;
    }

    // Linkear programa
    shader->ID = glCreateProgram();
    glAttachShader(shader->ID, vertex);
    glAttachShader(shader->ID, fragment);
    glLinkProgram(shader->ID);

    // Verificar errores de linkeo
    int success;
    char infoLog[512];
    glGetProgramiv(shader->ID, GL_LINK_STATUS, &success);
    if (!success) {
        glGetProgramInfoLog(shader->ID, 512, nullptr, infoLog);
        std::cerr << "Error: Fallo al linkear shader program\n" << infoLog << std::endl;
        glDeleteShader(vertex);
        glDeleteShader(fragment);
        delete shader;
        return nullptr;
    }

    // Limpiar shaders individuales
    glDeleteShader(vertex);
    glDeleteShader(fragment);

    return shader;
}

void DestroyShader(Shader* shader) {
    if (shader) {
        glDeleteProgram(shader->ID);
        delete shader;
    }
}

void UseShader(Shader* shader) {
    if (shader) {
        glUseProgram(shader->ID);
    }
}

unsigned int CompileShader(unsigned int type, const char* source) {
    unsigned int shader = glCreateShader(type);
    glShaderSource(shader, 1, &source, nullptr);
    glCompileShader(shader);

    // Verificar errores de compilación
    int success;
    char infoLog[512];
    glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
    if (!success) {
        glGetShaderInfoLog(shader, 512, nullptr, infoLog);
        std::cerr << "Error: Fallo al compilar shader\n" << infoLog << std::endl;
        glDeleteShader(shader);
        return 0;
    }

    return shader;
}

bool CheckCompileErrors(unsigned int shader, const char* type) {
    int success;
    char infoLog[1024];

    if (std::string(type) != "PROGRAM") {
        glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
        if (!success) {
            glGetShaderInfoLog(shader, 1024, nullptr, infoLog);
            std::cerr << "Error de compilación del shader (" << type << "):\n" << infoLog << std::endl;
            return false;
        }
    } else {
        glGetProgramiv(shader, GL_LINK_STATUS, &success);
        if (!success) {
            glGetProgramInfoLog(shader, 1024, nullptr, infoLog);
            std::cerr << "Error de linkeo del programa shader:\n" << infoLog << std::endl;
            return false;
        }
    }
    return true;
}

// Funciones de uniformes
void SetBool(Shader* shader, const char* name, bool value) {
    glUniform1i(glGetUniformLocation(shader->ID, name), (int)value);
}

void SetInt(Shader* shader, const char* name, int value) {
    glUniform1i(glGetUniformLocation(shader->ID, name), value);
}

void SetFloat(Shader* shader, const char* name, float value) {
    glUniform1f(glGetUniformLocation(shader->ID, name), value);
}

void SetVec2(Shader* shader, const char* name, float x, float y) {
    glUniform2f(glGetUniformLocation(shader->ID, name), x, y);
}

void SetVec3(Shader* shader, const char* name, float x, float y, float z) {
    glUniform3f(glGetUniformLocation(shader->ID, name), x, y, z);
}

void SetVec3v(Shader* shader, const char* name, const glm::vec3& value) {
    glUniform3fv(glGetUniformLocation(shader->ID, name), 1, glm::value_ptr(value));
}

void SetVec4(Shader* shader, const char* name, float x, float y, float z, float w) {
    glUniform4f(glGetUniformLocation(shader->ID, name), x, y, z, w);
}

void SetMat4(Shader* shader, const char* name, const glm::mat4& mat) {
    glUniformMatrix4fv(glGetUniformLocation(shader->ID, name), 1, GL_FALSE, glm::value_ptr(mat));
}

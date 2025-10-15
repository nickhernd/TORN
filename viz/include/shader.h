#ifndef SHADER_H
#define SHADER_H

#include <glad/gl.h>
#include <glm/glm.hpp>
#include <string>

// Estructura del shader
typedef struct {
    unsigned int ID;
} Shader;

// Funciones del shader
Shader* CreateShader(const char* vertexPath, const char* fragmentPath);
Shader* CreateShaderFromSource(const char* vertexSource, const char* fragmentSource);
void DestroyShader(Shader* shader);
void UseShader(Shader* shader);

// Funciones de uniformes
void SetBool(Shader* shader, const char* name, bool value);
void SetInt(Shader* shader, const char* name, int value);
void SetFloat(Shader* shader, const char* name, float value);
void SetVec2(Shader* shader, const char* name, float x, float y);
void SetVec3(Shader* shader, const char* name, float x, float y, float z);
void SetVec3v(Shader* shader, const char* name, const glm::vec3& value);
void SetVec4(Shader* shader, const char* name, float x, float y, float z, float w);
void SetMat4(Shader* shader, const char* name, const glm::mat4& mat);

// Funciones auxiliares
unsigned int CompileShader(unsigned int type, const char* source);
bool CheckCompileErrors(unsigned int shader, const char* type);

#endif // SHADER_H

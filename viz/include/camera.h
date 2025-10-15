#ifndef CAMERA_H
#define CAMERA_H

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

// Tipos de movimiento de cámara
enum CameraMovement {
    FORWARD,
    BACKWARD,
    LEFT,
    RIGHT,
    UP,
    DOWN
};

// Valores por defecto de la cámara
const float YAW         = -90.0f;
const float PITCH       =  0.0f;
const float SPEED       =  2.5f;
const float SENSITIVITY =  0.1f;
const float ZOOM        =  45.0f;

// Estructura de la cámara
typedef struct {
    // Atributos de la cámara
    glm::vec3 Position;
    glm::vec3 Front;
    glm::vec3 Up;
    glm::vec3 Right;
    glm::vec3 WorldUp;

    // Ángulos de Euler
    float Yaw;
    float Pitch;

    // Opciones de la cámara
    float MovementSpeed;
    float MouseSensitivity;
    float Zoom;

    // Estado
    bool firstMouse;
    float lastX;
    float lastY;
} Camera;

// Funciones de la cámara
Camera* CreateCamera(glm::vec3 position, glm::vec3 up, float yaw, float pitch);
void DestroyCamera(Camera* camera);
glm::mat4 GetViewMatrix(Camera* camera);
void ProcessKeyboard(Camera* camera, CameraMovement direction, float deltaTime);
void ProcessMouseMovement(Camera* camera, float xoffset, float yoffset, bool constrainPitch = true);
void ProcessMouseScroll(Camera* camera, float yoffset);
void UpdateCameraVectors(Camera* camera);

#endif // CAMERA_H

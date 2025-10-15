#include "../include/camera.h"

Camera* CreateCamera(glm::vec3 position, glm::vec3 up, float yaw, float pitch) {
    Camera* camera = new Camera();

    camera->Position = position;
    camera->WorldUp = up;
    camera->Yaw = yaw;
    camera->Pitch = pitch;

    camera->MovementSpeed = SPEED;
    camera->MouseSensitivity = SENSITIVITY;
    camera->Zoom = ZOOM;

    camera->firstMouse = true;
    camera->lastX = 0.0f;
    camera->lastY = 0.0f;

    // Calcular vectores iniciales
    UpdateCameraVectors(camera);

    return camera;
}

void DestroyCamera(Camera* camera) {
    if (camera) {
        delete camera;
    }
}

glm::mat4 GetViewMatrix(Camera* camera) {
    return glm::lookAt(camera->Position, camera->Position + camera->Front, camera->Up);
}

void ProcessKeyboard(Camera* camera, CameraMovement direction, float deltaTime) {
    float velocity = camera->MovementSpeed * deltaTime;

    switch (direction) {
        case FORWARD:
            camera->Position += camera->Front * velocity;
            break;
        case BACKWARD:
            camera->Position -= camera->Front * velocity;
            break;
        case LEFT:
            camera->Position -= camera->Right * velocity;
            break;
        case RIGHT:
            camera->Position += camera->Right * velocity;
            break;
        case UP:
            camera->Position += camera->Up * velocity;
            break;
        case DOWN:
            camera->Position -= camera->Up * velocity;
            break;
    }
}

void ProcessMouseMovement(Camera* camera, float xoffset, float yoffset, bool constrainPitch) {
    xoffset *= camera->MouseSensitivity;
    yoffset *= camera->MouseSensitivity;

    camera->Yaw += xoffset;
    camera->Pitch += yoffset;

    // Limitar pitch para evitar gimbal lock
    if (constrainPitch) {
        if (camera->Pitch > 89.0f)
            camera->Pitch = 89.0f;
        if (camera->Pitch < -89.0f)
            camera->Pitch = -89.0f;
    }

    // Actualizar vectores de la cámara
    UpdateCameraVectors(camera);
}

void ProcessMouseScroll(Camera* camera, float yoffset) {
    camera->Zoom -= yoffset;

    // Limitar zoom
    if (camera->Zoom < 1.0f)
        camera->Zoom = 1.0f;
    if (camera->Zoom > 45.0f)
        camera->Zoom = 45.0f;
}

void UpdateCameraVectors(Camera* camera) {
    // Calcular el nuevo vector Front
    glm::vec3 front;
    front.x = cos(glm::radians(camera->Yaw)) * cos(glm::radians(camera->Pitch));
    front.y = sin(glm::radians(camera->Pitch));
    front.z = sin(glm::radians(camera->Yaw)) * cos(glm::radians(camera->Pitch));
    camera->Front = glm::normalize(front);

    // Recalcular Right y Up
    camera->Right = glm::normalize(glm::cross(camera->Front, camera->WorldUp));
    camera->Up = glm::normalize(glm::cross(camera->Right, camera->Front));
}

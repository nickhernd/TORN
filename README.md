# 🛠️ Guía para trabajar en el proyecto

Este documento describe el flujo de trabajo que seguimos en el proyecto para garantizar una organización eficiente y colaborativa.

---

## 📝 Tipos de Issues

En este repositorio manejamos tres tipos principales de issues:

- **🪲 Bug Issue**: Se utiliza para reportar errores o fallos encontrados en el sistema. Debe contener una descripción clara del problema, los pasos para reproducirlo y cualquier información relevante (capturas de pantalla, logs, etc.).

- **📌 Requirement Issue**: Describe un nuevo requisito o funcionalidad a desarrollar. Debe incluir detalles sobre el objetivo, especificaciones y cualquier otra información necesaria para su implementación.

- **🛠️ Work Issue**: Son tareas que no forman parte de un requisito específico pero que requieren trabajo, como refactorización de código, optimización o mejoras en la documentación.

Cada issue debe asignarse a un proyecto y moverse dentro del flujo de trabajo establecido.

---

## 🚀 ¿Cómo empezar a trabajar?

### 1️⃣ Crear una nueva rama
- Antes de comenzar cualquier tarea, debes crear una nueva rama basada en `develop`.
> [!WARNING]
> Acuerdate de hacer un pull de develop antes de empezar.

> [!IMPORTANT]
> La rama debe seguir el siguiente formato (Número de la issue sobre la que se trabaja)_(Nombre de la issue) por ejemplo si trabajamos en la issue Seeder/Migration Admin #2048 la rama se llamará 2048_Seeder/Migration_Admin.


### 2️⃣ Realizar cambios y subirlos al repositorio
- Asegúrate de escribir código limpio y seguir las convenciones del proyecto.
- Realiza commits descriptivos que expliquen claramente qué cambios se realizaron.
- Una vez completado el trabajo, sube la rama al repositorio.

### 3️⃣ Crear una Pull Request (PR) hacia `develop`
> [!WARNING]
> Para fusionar los cambios en `develop`, es obligatorio crear una **Pull Request (PR)**.
- La PR debe seguir la plantilla establecida para garantizar una documentación adecuada.
- En la PR:
- **Te asignarás a ti mismo** como responsable de los cambios.
- **Asignarás un revisor**, que debe ser otro compañero de equipo.
- Si la PR está relacionada con diseño, debes asignar **a todos los compañeros** como revisores.
- Espera la aprobación antes de fusionar la PR en `develop`.

---

## 📂 Organización de Issues y Pull Requests

Para una mejor organización, utilizamos **Proyectos de GitHub**, donde clasificamos las tareas, generalmente preasignadas:

- ⚙️ **MTM**: En este proyecto están todas las tareas.

Cada **issue** o **pull request** debe moverse dentro del flujo de trabajo según su estado actual.

---

Siguiendo estas instrucciones, aseguramos un trabajo más eficiente y colaborativo dentro del equipo. 🚀

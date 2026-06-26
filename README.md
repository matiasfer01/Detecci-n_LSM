# Desarrollo de un sistema inteligente de traducción en tiempo real de la Lengua de Señas Mexicana a texto mediante visión por computadora y redes neuronales convolucionales

**Proyecto académico | Licenciatura en Informática | Universidad del Mar (UMAR)**

Este repositorio contiene el desarrollo de un prototipo funcional para la traducción en tiempo real de señas de la Lengua de Señas Mexicana a texto, utilizando visión por computadora y modelos de aprendizaje automático.

## Descripción del Proyecto

El proyecto consiste en el diseño y desarrollo de un sistema inteligente capaz de reconocer señas realizadas con la mano mediante una cámara web. La aplicación captura video en tiempo real, detecta los puntos de referencia de la mano y procesa esta información para identificar letras, señas dinámicas y acciones básicas como espacio o borrar.

La propuesta busca apoyar la comunicación entre personas usuarias de la Lengua de Señas Mexicana y personas que no dominan este lenguaje, mediante una herramienta tecnológica accesible y funcional.

Actualmente, el sistema se encuentra en etapa de prototipo funcional, por lo que está enfocado en:

* La detección de la mano mediante cámara web.
* La extracción de landmarks utilizando MediaPipe.
* El reconocimiento de señas estáticas.
* El reconocimiento de señas dinámicas.
* La conversión de señas reconocidas a texto.
* La generación de sugerencias de palabras.
* La ejecución local mediante un archivo `.exe` para Windows.

## 🛠️ Tecnologías y Conceptos Relacionados

* **Python:** Lenguaje principal utilizado para el desarrollo del sistema.
* **OpenCV:** Captura de video, procesamiento de imágenes e interfaz visual.
* **MediaPipe:** Detección de la mano y extracción de puntos de referencia.
* **NumPy:** Procesamiento numérico de los landmarks.
* **Pandas:** Manejo de datos durante la etapa de entrenamiento.
* **Scikit-learn:** Entrenamiento de modelos de clasificación.
* **Random Forest:** Modelo utilizado para clasificar señas a partir de landmarks.
* **Joblib:** Almacenamiento y carga de modelos entrenados.
* **Pillow:** Renderizado de texto con soporte para caracteres especiales.
* **JSON:** Manejo de palabras y secuencias de señas dinámicas.
* **PyInstaller:** Generación del ejecutable local para Windows.
* **Visión por Computadora:** Captura, análisis e interpretación de imágenes en tiempo real.
* **Aprendizaje Automático:** Clasificación de señas mediante modelos entrenados.

> Nota: el título del proyecto contempla el enfoque de redes neuronales convolucionales como parte del alcance general; sin embargo, el prototipo funcional actual utiliza MediaPipe para extracción de landmarks y modelos de clasificación entrenados con Random Forest.

## 🎯 Objetivo General

Desarrollar un sistema inteligente de traducción en tiempo real de la Lengua de Señas Mexicana a texto, mediante visión por computadora, capaz de detectar la mano del usuario, procesar sus puntos de referencia y clasificar señas estáticas y dinámicas para generar texto en una interfaz visual.

## 🎯 Objetivos Específicos

* Capturar imágenes y video en tiempo real mediante una cámara web.
* Detectar la mano del usuario utilizando MediaPipe.
* Extraer y normalizar landmarks de la mano.
* Construir un conjunto de datos para señas estáticas y dinámicas.
* Entrenar modelos de clasificación para el reconocimiento de señas.
* Implementar una interfaz visual para mostrar la cámara, la seña detectada y el texto generado.
* Integrar acciones básicas como espacio y borrar.
* Generar un ejecutable local para facilitar el uso del sistema en Windows.

## 📁 Estructura General del Proyecto

```text
Prototipo_v1/
│
├── HandDetector.py
├── predictor_dinamicas.py
├── capturar_dinamicas.py
├── entrenar_dinamicas.py
├── vowels.py
├── model.py
│
├── vowels_model.joblib
├── dynamic_signs_model.joblib
├── vowels_landmarks.csv
├── words.json
│
├── dataset/
├── dataset_dinamicas/
```
## ⚙️ Requisitos del Sistema

* Sistema operativo Windows.
* Cámara web integrada o externa.
* Permiso de acceso a la cámara.
* Buena iluminación para mejorar la detección.
* Mantener los archivos `.joblib` y `words.json` junto al ejecutable.

> Si el equipo no cuenta con cámara web, la aplicación no podrá iniciar correctamente el proceso de detección, ya que depende de la captura de video en tiempo real.

## 🔐 Credenciales de Acceso

No aplica.

El sistema no cuenta con módulo de inicio de sesión, usuarios ni contraseñas. La aplicación funciona de manera local y no requiere credenciales para su uso.

## 🗄️ Base de Datos

No aplica.

El sistema no utiliza una base de datos relacional ni no relacional. La información necesaria para su funcionamiento se maneja mediante archivos locales, como modelos entrenados en formato `.joblib`, archivos `.json` y archivos de datos utilizados durante el entrenamiento.

## 📌 Estado del Proyecto

El proyecto se encuentra en etapa de prototipo funcional. Actualmente permite reconocer señas mediante visión por computadora, detectar landmarks de la mano, clasificar señas estáticas y dinámicas, y generar texto en una interfaz visual.

## 📄 Licencia

Este proyecto está bajo la licencia MIT.

## 👤 Autor

Desarrollado por:

**LUIS FERNANDO MATIAS ACEVEDO**

Licenciatura en Informática
Universidad del Mar (UMAR)

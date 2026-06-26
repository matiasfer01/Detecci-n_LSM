import os
import json
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from collections import Counter


# ==============================
# CONFIGURACIÓN GENERAL
# ==============================

# Carpeta donde se encuentra el dataset de señas dinámicas.
# Dentro de esta carpeta debe existir una subcarpeta por cada clase.
DATASET_DIR = "dataset_dinamicas"

# Ruta y nombre del archivo donde se guardará el modelo entrenado.
MODEL_PATH = "dynamic_signs_model.joblib"

# Cantidad de frames que debe tener cada secuencia.
# Debe coincidir con el valor usado durante la captura de datos.
FRAMES_POR_SECUENCIA = 60

# Cantidad de valores por frame.
# MediaPipe genera 21 landmarks y cada landmark tiene 3 coordenadas: x, y, z.
# 21 * 3 = 63.
LANDMARKS_POR_FRAME = 63


def cargar_dataset():
    """
    Carga el dataset de señas dinámicas desde archivos JSON.

    Proceso:
        1. Revisa si existe la carpeta principal del dataset.
        2. Recorre cada subcarpeta, donde cada subcarpeta representa una clase.
        3. Lee los archivos JSON de cada clase.
        4. Convierte cada secuencia en un arreglo NumPy.
        5. Valida que cada secuencia tenga la forma esperada:
           60 frames x 63 valores por frame.
        6. Aplana cada secuencia para que pueda ser usada por el modelo.
        7. Guarda los datos en X y las etiquetas en y.

    Retorna:
        X:
            Arreglo NumPy con las secuencias aplanadas.
            Cada ejemplo tiene 3780 valores:
            60 frames * 63 landmarks.

        y:
            Arreglo NumPy con la clase correspondiente a cada ejemplo.
    """

    # Lista donde se guardarán las características de cada ejemplo.
    X = []

    # Lista donde se guardará la etiqueta o clase de cada ejemplo.
    y = []

    # Verifica si existe la carpeta del dataset.
    if not os.path.exists(DATASET_DIR):
        print(f"No existe la carpeta: {DATASET_DIR}")
        return np.array(X), np.array(y)

    # Recorre cada elemento dentro de la carpeta principal del dataset.
    for clase in os.listdir(DATASET_DIR):
        # Construye la ruta de la carpeta correspondiente a la clase.
        carpeta = os.path.join(DATASET_DIR, clase)

        # Si el elemento no es una carpeta, se ignora.
        if not os.path.isdir(carpeta):
            continue

        # Recorre todos los archivos dentro de la carpeta de la clase.
        for archivo in os.listdir(carpeta):
            # Solo procesa archivos con extensión .json.
            if archivo.endswith(".json"):
                # Construye la ruta completa del archivo JSON.
                ruta = os.path.join(carpeta, archivo)

                try:
                    # Abre y lee el archivo JSON.
                    with open(ruta, "r", encoding="utf-8") as f:
                        secuencia = json.load(f)

                    # Convierte la secuencia cargada en un arreglo NumPy de tipo float32.
                    secuencia = np.array(secuencia, dtype=np.float32)

                    # Valida que la secuencia tenga la forma correcta.
                    # Debe contener 60 frames y 63 valores por frame.
                    if secuencia.shape != (FRAMES_POR_SECUENCIA, LANDMARKS_POR_FRAME):
                        print(f"Archivo ignorado por forma incorrecta: {ruta}")
                        print(f"Forma encontrada: {secuencia.shape}")
                        continue

                    # Aplanar:
                    # Convierte la matriz de 60 x 63 en un solo vector de 3780 datos.
                    # Esto permite que el modelo Random Forest pueda usar la secuencia como entrada.
                    X.append(secuencia.flatten())

                    # Guarda la clase asociada al ejemplo actual.
                    y.append(clase)

                except Exception as e:
                    # Captura errores de lectura o conversión para evitar que el programa se detenga.
                    print(f"Error leyendo {ruta}: {e}")

    # Devuelve las características y etiquetas como arreglos NumPy.
    return np.array(X), np.array(y)


def main():
    """
    Función principal del programa.

    Proceso general:
        1. Carga el dataset desde archivos JSON.
        2. Muestra información general del dataset.
        3. Verifica que existan datos suficientes para entrenar.
        4. Divide los datos en entrenamiento y prueba.
        5. Crea y entrena un modelo Random Forest.
        6. Evalúa el modelo usando accuracy y reporte de clasificación.
        7. Guarda el modelo entrenado en un archivo .joblib.
    """

    # Carga las características y etiquetas del dataset.
    X, y = cargar_dataset()

    # Muestra la cantidad total de ejemplos válidos encontrados.
    print("\nTotal de ejemplos válidos:", len(X))

    # Muestra las clases detectadas dentro del dataset.
    print("Clases encontradas:", sorted(set(y)))

    # Cuenta cuántos ejemplos existen por cada clase.
    conteo = Counter(y)

    print("\nEjemplos por clase:")

    # Imprime el número de ejemplos disponibles para cada clase.
    for clase, cantidad in sorted(conteo.items()):
        print(f"  {clase}: {cantidad}")

    # Si no hay ejemplos válidos, no se puede entrenar el modelo.
    if len(X) == 0:
        print("\nNo hay datos para entrenar.")
        return

    # Para entrenar un clasificador se necesitan al menos dos clases diferentes.
    if len(set(y)) < 2:
        print("\nNecesitas al menos 2 clases para entrenar.")
        return

    # Obtiene la cantidad de ejemplos de la clase con menos datos.
    clase_menor = min(conteo.values())

    # Si alguna clase tiene menos de 2 ejemplos, no se puede hacer división estratificada.
    if clase_menor < 2:
        print("\nError: alguna clase tiene menos de 2 ejemplos.")
        print("Graba más ejemplos de esa clase antes de entrenar.")
        return

    # Divide el dataset en datos de entrenamiento y datos de prueba.
    # test_size=0.25 significa que el 25% se usará para prueba.
    # random_state=42 permite que la división sea reproducible.
    # stratify=y mantiene la proporción de clases en entrenamiento y prueba.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y
    )

    # Crea el modelo Random Forest para clasificación.
    model = RandomForestClassifier(
        # Número de árboles del bosque.
        n_estimators=300,

        # Semilla para obtener resultados reproducibles.
        random_state=42,

        # Balancea el peso de las clases para ayudar cuando hay diferencias en cantidad de ejemplos.
        class_weight="balanced",

        # No se limita la profundidad máxima de los árboles.
        max_depth=None
    )

    print("\nEntrenando modelo...")

    # Entrena el modelo con los datos de entrenamiento.
    model.fit(X_train, y_train)

    # Realiza predicciones sobre los datos de prueba.
    y_pred = model.predict(X_test)

    # Calcula e imprime la precisión general del modelo.
    print("\nAccuracy:", accuracy_score(y_test, y_pred))

    # Imprime métricas por clase:
    # precision, recall, f1-score y support.
    print("\nReporte:")
    print(classification_report(y_test, y_pred))

    # Guarda el modelo entrenado en un archivo .joblib.
    joblib.dump(model, MODEL_PATH)

    print(f"\nModelo guardado como: {MODEL_PATH}")


# Punto de entrada del programa.
# Ejecuta main() solamente si este archivo se ejecuta directamente.
if __name__ == "__main__":
    main()
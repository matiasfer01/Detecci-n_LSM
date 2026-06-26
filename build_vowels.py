import os
import glob
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd

# ============================================================
# CONFIGURACIÓN GENERAL DEL PROGRAMA
# ============================================================

# Ruta donde se encuentra el dataset.
# La estructura esperada es:
# dataset/MyDataset/A
# dataset/MyDataset/B
# dataset/MyDataset/C
# Cada carpeta representa una clase o etiqueta.
DATASET_DIR = os.path.join("dataset", "MyDataset")

# Nombre del archivo CSV que se generará con las características extraídas.
CSV_OUT = "vowels_landmarks.csv"

# Extensiones de imágenes permitidas para procesar.
ALLOWED_EXT = (".jpg", ".jpeg", ".png")

# Inicialización del módulo Hands de MediaPipe.
# Este módulo permite detectar manos y obtener sus puntos clave o landmarks.
mp_hands = mp.solutions.hands


def extract_features_from_image(img_bgr):
    """
    Extrae las características de una mano detectada en una imagen.

    Esta función recibe una imagen en formato BGR, la convierte a RGB
    y utiliza MediaPipe Hands para detectar una mano. Si se detecta una mano,
    obtiene los 21 landmarks de la mano, donde cada landmark contiene
    coordenadas x, y, z.

    Posteriormente, los puntos se normalizan para que el modelo no dependa
    tanto de la posición o tamaño de la mano en la imagen.

    Parámetros:
        img_bgr: Imagen cargada con OpenCV en formato BGR.

    Retorna:
        Un vector de 63 valores:
            21 landmarks * 3 coordenadas = 63 características.

        Retorna None si no se detecta ninguna mano.
    """

    # OpenCV carga las imágenes en formato BGR.
    # MediaPipe trabaja con imágenes en formato RGB, por eso se realiza la conversión.
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # Se crea una instancia del detector de manos de MediaPipe.
    # static_image_mode=True indica que se procesarán imágenes individuales,
    # no video en tiempo real.
    with mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=0.6
    ) as hands:

        # Procesa la imagen y busca landmarks de la mano.
        results = hands.process(img_rgb)

        # Si no se detecta ninguna mano, se devuelve None.
        if not results.multi_hand_landmarks:
            return None

        # Se toma la primera mano detectada.
        hand_lms = results.multi_hand_landmarks[0]

        # Se convierten los 21 landmarks en un arreglo NumPy.
        # Cada punto contiene las coordenadas x, y, z.
        pts = np.array(
            [[lm.x, lm.y, lm.z] for lm in hand_lms.landmark],
            dtype=np.float32
        )

        # ============================================================
        # NORMALIZACIÓN DE LOS LANDMARKS
        # ============================================================

        # Se resta el punto 0, que corresponde a la base de la muñeca.
        # Esto permite que las coordenadas sean relativas a la mano,
        # no a la posición absoluta dentro de la imagen.
        pts = pts - pts[0]

        # Se calcula una escala usando la distancia del punto 9.
        # El punto 9 se encuentra aproximadamente en la base del dedo medio,
        # por lo que sirve como referencia del tamaño de la mano.
        scale = np.linalg.norm(pts[9]) + 1e-6

        # Se divide entre la escala para que las manos grandes o pequeñas
        # tengan una representación más uniforme.
        pts = pts / scale

        # Se aplana la matriz de 21x3 para obtener un vector de 63 valores.
        return pts.flatten()


def main():
    """
    Función principal del programa.

    Esta función recorre todas las carpetas del dataset, carga las imágenes,
    extrae los landmarks de la mano usando MediaPipe, guarda las características
    en una tabla y finalmente genera un archivo CSV.

    El CSV generado contiene:
        - Una columna llamada 'label', que representa la clase de la imagen.
        - 63 columnas de características llamadas f0, f1, f2, ..., f62.
    """

    # Verifica que exista la carpeta del dataset.
    # Si no existe, el programa se detiene y muestra un error.
    if not os.path.exists(DATASET_DIR):
        raise FileNotFoundError(f"No existe la carpeta: {DATASET_DIR}")

    # Obtiene las carpetas dentro del dataset.
    # Cada carpeta representa una etiqueta o clase.
    labels = [
        d for d in os.listdir(DATASET_DIR)
        if os.path.isdir(os.path.join(DATASET_DIR, d))
    ]

    # Ordena las etiquetas alfabéticamente para mantener un orden estable.
    labels = sorted(labels)

    # Lista donde se guardarán todas las filas que después formarán el CSV.
    rows = []

    # Contadores para llevar control del proceso.
    total = 0          # Total de imágenes encontradas.
    usados = 0         # Imágenes donde sí se detectó una mano.
    descartados = 0    # Imágenes descartadas por error o falta de detección.

    # Recorre cada carpeta del dataset.
    for label in labels:
        folder = os.path.join(DATASET_DIR, label)

        # Lista donde se guardarán las rutas de las imágenes encontradas.
        imgs = []

        # Busca imágenes con las extensiones permitidas.
        for ext in ALLOWED_EXT:
            imgs.extend(glob.glob(os.path.join(folder, f"*{ext}")))

        # Procesa cada imagen encontrada en la carpeta actual.
        for img_path in imgs:
            total += 1

            # Carga la imagen usando OpenCV.
            img = cv2.imread(img_path)

            # Si la imagen no se puede leer, se descarta.
            if img is None:
                descartados += 1
                continue

            # Extrae las características de la mano.
            feat = extract_features_from_image(img)

            # Si no se detectó mano en la imagen, se descarta.
            if feat is None:
                descartados += 1
                continue

            # Se crea una fila con la etiqueta de la imagen.
            row = {"label": label}

            # Se agregan las 63 características al diccionario.
            # Cada característica se guarda como f0, f1, f2, ..., f62.
            for i, v in enumerate(feat):
                row[f"f{i}"] = float(v)

            # Se agrega la fila procesada a la lista principal.
            rows.append(row)

            # Aumenta el contador de imágenes utilizadas.
            usados += 1

    # Si no se logró extraer ninguna mano, se detiene el programa.
    if not rows:
        raise RuntimeError(
            "No se pudo extraer ninguna mano. "
            "Revisa iluminación / encuadre / dataset."
        )

    # Convierte la lista de filas en un DataFrame de pandas.
    df = pd.DataFrame(rows)

    # Guarda el DataFrame en un archivo CSV.
    # index=False evita que se agregue una columna extra con el índice.
    df.to_csv(CSV_OUT, index=False)

    # Muestra un resumen del proceso realizado.
    print("=== LISTO ===")
    print(f"Dataset: {DATASET_DIR}")
    print(f"Total imágenes encontradas: {total}")
    print(f"Usadas (mano detectada): {usados}")
    print(f"Descartadas (sin mano/errores): {descartados}")
    print(f"✅ CSV generado: {CSV_OUT}")
    print(f"Columnas: {df.shape[1]} (label + 63 features) | Filas: {df.shape[0]}")


# Punto de entrada del programa.
# Esto permite que main() se ejecute solo cuando el archivo se corre directamente.
if __name__ == "__main__":
    main()

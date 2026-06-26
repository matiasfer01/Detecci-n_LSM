from collections import deque
import numpy as np
import joblib
import os
import time


class PredictorDinamicas:
    """
    Clase encargada de predecir señas dinámicas usando un modelo previamente entrenado.

    Esta clase:
        - Carga un modelo entrenado desde un archivo .joblib.
        - Mantiene una secuencia temporal de landmarks de la mano.
        - Extrae y normaliza los landmarks detectados por MediaPipe.
        - Cuando junta suficientes frames, realiza una predicción.
        - Usa un umbral de confianza para aceptar o rechazar la predicción.
        - Usa un cooldown para evitar repetir la misma predicción muchas veces seguidas.
    """

    def __init__(self, model_path="dynamic_signs_model.joblib"):
        """
        Inicializa el predictor de señas dinámicas.

        Parámetros:
            model_path:
                Ruta del archivo donde está guardado el modelo entrenado.

        Proceso:
            1. Verifica que exista el archivo del modelo.
            2. Carga el modelo con joblib.
            3. Define cuántos frames tendrá cada secuencia.
            4. Crea una cola para almacenar los landmarks de los últimos frames.
            5. Define el umbral mínimo de confianza.
            6. Define el tiempo de espera entre predicciones aceptadas.
        """

        # Verifica si el archivo del modelo existe.
        # Si no existe, detiene el programa y muestra un mensaje indicando qué hacer.
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"No encontré {model_path}. Primero ejecuta entrenar_dinamicas.py"
            )

        # Carga el modelo entrenado desde el archivo .joblib.
        self.model = joblib.load(model_path)

        # Número de frames que debe contener una secuencia completa.
        # Debe coincidir con el número usado durante el entrenamiento.
        self.frames_por_secuencia = 60

        # Cola que almacena los landmarks de los frames recientes.
        # maxlen=60 significa que solo guarda los últimos 60 frames.
        self.secuencia = deque(maxlen=self.frames_por_secuencia)

        # Umbral mínimo de confianza para aceptar una predicción.
        self.threshold = 0.82

        # Tiempo mínimo, en segundos, entre una predicción aceptada y otra.
        # Esto evita que se agregue la misma letra repetidamente muy rápido.
        self.cooldown = 1.2

        # Guarda el tiempo de la última predicción aceptada.
        self.last_time = 0.0

    def extraer_landmarks(self, hand_lms, handedness_label):
        """
        Extrae y normaliza los landmarks de una mano detectada.

        Parámetros:
            hand_lms:
                Landmarks de la mano detectada por MediaPipe.

            handedness_label:
                Etiqueta que indica si la mano detectada es izquierda o derecha.

        Proceso:
            1. Convierte los landmarks de MediaPipe en un arreglo NumPy.
            2. Usa la muñeca como punto de origen.
            3. Normaliza la escala usando el landmark 9.
            4. Si la mano es derecha, invierte el eje X.
            5. Devuelve los landmarks en un vector plano.

        Retorna:
            Un vector de 63 valores:
            21 landmarks * 3 coordenadas (x, y, z).
        """

        # Convierte los landmarks de MediaPipe en una matriz NumPy.
        # Cada landmark contiene tres valores: x, y, z.
        pts = np.array(
            [[lm.x, lm.y, lm.z] for lm in hand_lms.landmark],
            dtype=np.float32
        )

        # Centra todos los puntos tomando como origen el landmark 0,
        # que corresponde a la muñeca.
        pts = pts - pts[0]

        # Calcula una escala usando el landmark 9 como referencia.
        # Se suma 1e-6 para evitar división entre cero.
        scale = np.linalg.norm(pts[9]) + 1e-6

        # Normaliza los puntos para que la escala de la mano sea más uniforme.
        pts = pts / scale

        # Si la mano detectada es derecha, invierte el eje X.
        # Esto ayuda a tratar manos derecha e izquierda de forma más similar.
        if handedness_label.lower() == "right":
            pts[:, 0] *= -1

        # Convierte la matriz de landmarks en un vector plano.
        return pts.flatten()

    def actualizar(self, hand_lms, handedness_label):
        """
        Actualiza la secuencia de frames y realiza una predicción si ya hay suficientes datos.

        Parámetros:
            hand_lms:
                Landmarks de la mano detectada en el frame actual.

            handedness_label:
                Indica si la mano detectada es izquierda o derecha.

        Proceso:
            1. Extrae los landmarks normalizados del frame actual.
            2. Agrega esos landmarks a la secuencia.
            3. Si aún no hay 60 frames, no predice nada.
            4. Si ya hay 60 frames, aplana la secuencia.
            5. Verifica el cooldown para evitar predicciones muy seguidas.
            6. Si el modelo tiene predict_proba, usa confianza.
            7. Si la confianza supera el umbral y no es "Nada", devuelve la letra.
            8. Limpia la secuencia después de aceptar una predicción.

        Retorna:
            La letra dinámica predicha si se cumple el umbral y el cooldown.
            Una cadena vacía si todavía no hay predicción válida.
        """

        # Extrae las características del frame actual.
        features = self.extraer_landmarks(hand_lms, handedness_label)

        # Agrega los landmarks del frame actual a la secuencia.
        self.secuencia.append(features)

        # Si aún no se tienen los 60 frames necesarios, no se realiza predicción.
        if len(self.secuencia) < self.frames_por_secuencia:
            return ""

        # Convierte la secuencia completa en un arreglo NumPy.
        # Luego la aplana y le da forma de una sola muestra para el modelo.
        X = np.array(self.secuencia, dtype=np.float32).flatten().reshape(1, -1)

        # Obtiene el tiempo actual.
        ahora = time.time()

        # Si todavía no ha pasado el tiempo de cooldown, no predice.
        if ahora - self.last_time < self.cooldown:
            return ""

        # Si el modelo permite obtener probabilidades, se usa la confianza de predicción.
        if hasattr(self.model, "predict_proba"):
            # Obtiene las probabilidades de cada clase.
            probs = self.model.predict_proba(X)[0]

            # Obtiene el índice de la clase con mayor probabilidad.
            idx = int(np.argmax(probs))

            # Obtiene la confianza de la predicción.
            conf = float(probs[idx])

            # Obtiene la etiqueta predicha.
            label = self.model.classes_[idx]

            # Solo acepta la predicción si supera el umbral
            # y si la clase predicha no es "Nada".
            if conf >= self.threshold and label != "Nada":
                # Actualiza el tiempo de la última predicción aceptada.
                self.last_time = ahora

                # Limpia la secuencia para comenzar a capturar una nueva seña.
                self.secuencia.clear()

                # Devuelve la letra o clase predicha.
                return label
        else:
            # Si el modelo no tiene predict_proba, predice directamente la clase.
            label = self.model.predict(X)[0]

            # Actualiza el tiempo de la última predicción.
            self.last_time = ahora

            # Limpia la secuencia después de predecir.
            self.secuencia.clear()

            # Devuelve la clase predicha.
            return label

        # Si no se cumple ninguna condición de predicción válida, devuelve cadena vacía.
        return ""

    def limpiar(self):
        """
        Limpia la secuencia de frames almacenada.

        Esta función se usa cuando no se detecta una mano,
        para evitar que frames viejos afecten una futura predicción.
        """

        # Vacía la cola de landmarks acumulados.
        self.secuencia.clear()
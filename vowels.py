import cv2
import mediapipe as mp
import numpy as np
import os
import time

# ================= CONFIG =================

# Carpeta principal donde se guardará el dataset.
# Dentro de esta ruta se crearán subcarpetas por cada etiqueta o clase.
DATASET_DIR = os.path.join("dataset", "MyDataset")

# Diccionario que relaciona teclas del teclado con las etiquetas del dataset.
# Cada tecla permite cambiar la clase actual que se va a capturar.
LABEL_KEYS = {
    ord("a"): "A",
    ord("b"): "B",
    ord("c"): "C",
    ord("d"): "D",
    ord("e"): "E",
    ord("f"): "F",
    ord("g"): "G",
    ord("h"): "H",
    ord("i"): "I",
    ord("l"): "L",
    ord("m"): "M",
    ord("n"): "N",
    ord("o"): "O",
    ord("p"): "P",
    ord("r"): "R",
    ord("s"): "S",
    ord("t"): "T",
    ord("u"): "U",
    ord("v"): "V",
    ord("w"): "W",
    ord("y"): "Y",

    # Etiquetas especiales para acciones del sistema.
    ord("0"): "Borrar",
    ord("1"): "Espacio"
}

# Tiempo mínimo de espera entre guardados manuales.
SAVE_DELAY   = 0.20   # delay modo manual (SPACE)

# Tiempo de espera entre cada foto guardada durante el modo ráfaga.
RAFAGA_DELAY = 0.15   # delay entre fotos en modo ráfaga (segundos)

# Cantidad total de imágenes que se guardarán automáticamente en una ráfaga.
RAFAGA_TOTAL = 200    # cuántas fotos toma la ráfaga automática

# Extensión con la que se guardarán las imágenes.
IMG_EXT = ".jpg"

# =========================================


class HandDatasetCollector:
    """
    Clase encargada de capturar imágenes de señas usando la cámara.

    Esta clase permite:
        - Abrir la cámara.
        - Detectar si hay una mano usando MediaPipe.
        - Cambiar la etiqueta actual con teclas.
        - Guardar imágenes manualmente.
        - Guardar imágenes automáticamente en modo ráfaga.
        - Organizar las imágenes en carpetas por clase.
    """

    def __init__(self):
        """
        Inicializa la cámara, MediaPipe Hands, las etiquetas y las carpetas del dataset.

        Proceso general:
            1. Abre la cámara principal.
            2. Configura MediaPipe para detectar una mano.
            3. Define la etiqueta inicial.
            4. Inicializa los contadores de imágenes por clase.
            5. Crea las carpetas necesarias para guardar el dataset.
        """

        # Abre la cámara principal del equipo.
        self.cap = cv2.VideoCapture(0)

        # Inicializa el módulo Hands de MediaPipe.
        self.mp_hands = mp.solutions.hands

        # Configura MediaPipe Hands.
        # max_num_hands=1 indica que solo se detectará una mano.
        # min_detection_confidence define la confianza mínima para detectar una mano.
        # min_tracking_confidence define la confianza mínima para seguir la mano entre frames.
        self.hands = self.mp_hands.Hands(
            max_num_hands=1,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6
        )

        # Herramienta de MediaPipe para dibujar landmarks y conexiones de la mano.
        self.mp_drawing = mp.solutions.drawing_utils

        # Etiqueta inicial seleccionada al iniciar el programa.
        self.current_label  = "A"

        # Guarda el tiempo en que se realizó el último guardado.
        # Se usa para controlar el delay entre capturas.
        self.last_save_time = 0.0

        # Diccionario donde se almacenará el conteo de imágenes existentes por clase.
        self.counts         = {}

        # -------------------------
        # VARIABLES DEL MODO RÁFAGA
        # -------------------------

        # Indica si el modo ráfaga está activo.
        self.rafaga_activa    = False

        # Cuenta cuántas imágenes se han guardado en la ráfaga actual.
        self.rafaga_guardadas = 0

        # Crea la carpeta principal del dataset si no existe.
        os.makedirs(DATASET_DIR, exist_ok=True)

        # Obtiene todas las etiquetas posibles sin repetirlas.
        all_labels = sorted(set(LABEL_KEYS.values()))

        # Crea una carpeta por cada etiqueta y cuenta cuántas imágenes ya existen.
        for lbl in all_labels:
            # Ruta de la carpeta correspondiente a la etiqueta actual.
            folder = os.path.join(DATASET_DIR, lbl)

            # Crea la carpeta si no existe.
            os.makedirs(folder, exist_ok=True)

            # Cuenta las imágenes existentes dentro de esa carpeta.
            self.counts[lbl] = len([
                f for f in os.listdir(folder)
                if f.lower().endswith((".jpg", ".jpeg", ".png"))
            ])

    def guardar_imagen(self, frame):
        """
        Guarda una imagen en la carpeta de la etiqueta actual.

        Parámetros:
            frame:
                Imagen capturada por la cámara.

        Proceso:
            1. Obtiene la carpeta de la etiqueta actual.
            2. Calcula el siguiente número de imagen.
            3. Construye el nombre del archivo.
            4. Guarda la imagen con OpenCV.
            5. Actualiza el contador si el guardado fue exitoso.

        Retorna:
            True si la imagen se guardó correctamente.
            False si ocurrió algún problema al guardar.
        """

        # Carpeta donde se guardará la imagen según la etiqueta actual.
        folder   = os.path.join(DATASET_DIR, self.current_label)

        # Calcula el número consecutivo de la siguiente imagen.
        idx      = self.counts[self.current_label] + 1

        # Crea el nombre del archivo con formato:
        # Etiqueta_00001.jpg, Etiqueta_00002.jpg, etc.
        filename = f"{self.current_label}_{idx:05d}{IMG_EXT}"

        # Ruta completa donde se guardará la imagen.
        path     = os.path.join(folder, filename)

        # Guarda la imagen en disco.
        ok       = cv2.imwrite(path, frame)

        # Si la imagen se guardó correctamente, actualiza el contador y el tiempo.
        if ok:
            self.counts[self.current_label] += 1
            self.last_save_time = time.time()
            return True

        # Si no se pudo guardar, retorna False.
        return False

    def iniciar(self):
        """
        Inicia el ciclo principal de captura del dataset.

        Controles:
            Letras:
                Cambian la etiqueta actual.
            0:
                Selecciona la etiqueta "Borrar".
            1:
                Selecciona la etiqueta "Espacio".
            SPACE:
                Inicia una ráfaga automática de imágenes.
            F:
                Guarda una imagen manual.
            ESC:
                Sale del programa.

        Proceso general:
            1. Lee frames de la cámara.
            2. Detecta si hay una mano.
            3. Muestra la cámara con landmarks dibujados.
            4. Permite cambiar la etiqueta actual.
            5. Permite guardar imágenes manualmente.
            6. Permite capturar imágenes en modo ráfaga.
        """

        # Mensajes iniciales para el usuario.
        print("=== Captura de Dataset ===")
        print("Teclas letra : a b c d e f g h i l m n o p r s t u v w y  |  0=Borrar  1=Espacio")
        print("SPACE        : iniciar rafaga automatica de 200 fotos")
        print("F            : guardar 1 imagen manual")
        print("ESC          : salir")
        print(f"Guardando en : {DATASET_DIR}\n")

        # Ciclo principal de captura.
        while True:
            # Lee un frame desde la cámara.
            ret, frame = self.cap.read()

            # Si no se puede leer la cámara, se termina el ciclo.
            if not ret:
                break

            # Convierte el frame de BGR a RGB porque MediaPipe trabaja con RGB.
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Procesa la imagen para detectar manos.
            results   = self.hands.process(image_rgb)

            # Bandera que indica si se detectó una mano en el frame actual.
            hand_detected = False

            # Copia del frame para mostrarlo con dibujos sin alterar directamente el original.
            display_frame = frame.copy()

            # Si MediaPipe detecta landmarks de una mano.
            if results.multi_hand_landmarks:
                # Marca que sí hay mano detectada.
                hand_detected = True

                # Dibuja los landmarks y conexiones sobre la copia del frame.
                self.mp_drawing.draw_landmarks(
                    display_frame,
                    results.multi_hand_landmarks[0],
                    self.mp_hands.HAND_CONNECTIONS
                )

            # Voltea el frame horizontalmente para mostrarlo como espejo.
            display_frame = cv2.flip(display_frame, 1)

            # ---- HUD minimo: solo la letra actual ----

            # Muestra en pantalla la etiqueta actual que se está capturando.
            cv2.putText(display_frame, f"{self.current_label}",
                        (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (255, 255, 255), 3)

            # Muestra la ventana de captura.
            cv2.imshow("Captura Dataset - Prototipo v1", display_frame)

            # Lee la tecla presionada por el usuario.
            key = cv2.waitKey(1) & 0xFF

            # Salir del programa con ESC.
            if key == 27:
                break

            # Cambiar etiqueta.
            # Solo permite cambiar la etiqueta si no hay una ráfaga activa.
            if not self.rafaga_activa and key in LABEL_KEYS:
                self.current_label = LABEL_KEYS[key]
                print(f"Etiqueta actual: {self.current_label}  ({self.counts[self.current_label]} fotos existentes)")

            # Modo manual con tecla F.
            if not self.rafaga_activa and key in (ord('f'), ord('F')):
                # Si no se detectó mano, no se guarda la imagen.
                if not hand_detected:
                    print("No se guardó: no hay mano detectada.")
                    continue

                # Obtiene el tiempo actual.
                now = time.time()

                # Evita guardar muchas imágenes demasiado rápido.
                if now - self.last_save_time < SAVE_DELAY:
                    continue

                # Guarda una imagen manual y muestra confirmación.
                if self.guardar_imagen(frame):
                    print(f"Guardada manual: {self.current_label} #{self.counts[self.current_label]}")

            # Activar ráfaga con SPACE.
            if not self.rafaga_activa and key == 32:
                # Si no hay mano detectada, no inicia la ráfaga.
                if not hand_detected:
                    print("No se inició la ráfaga: no hay mano detectada.")
                else:
                    # Activa el modo ráfaga.
                    self.rafaga_activa    = True

                    # Reinicia el contador de imágenes guardadas en esta ráfaga.
                    self.rafaga_guardadas = 0

                    # Mensaje de inicio de ráfaga.
                    print(f"\nRáfaga iniciada para '{self.current_label}' — tomando {RAFAGA_TOTAL} fotos...")

            # Lógica ráfaga automática.
            if self.rafaga_activa:
                # Si se pierde la mano, la ráfaga se pausa hasta volver a detectarla.
                if not hand_detected:
                    print("Ráfaga pausada: perdí la mano...")
                    continue

                # Obtiene el tiempo actual.
                now = time.time()

                # Guarda una imagen únicamente si ya pasó el delay definido.
                if now - self.last_save_time >= RAFAGA_DELAY:
                    # Guarda la imagen y aumenta el contador de la ráfaga.
                    if self.guardar_imagen(frame):
                        self.rafaga_guardadas += 1
                        print(f"  Rafaga {self.current_label}: {self.rafaga_guardadas}/{RAFAGA_TOTAL}")

                    # Si se alcanza el total de imágenes, se desactiva la ráfaga.
                    if self.rafaga_guardadas >= RAFAGA_TOTAL:
                        self.rafaga_activa = False
                        print(f"\nRáfaga completada: {RAFAGA_TOTAL} fotos de '{self.current_label}' guardadas.\n")

        # Libera la cámara al terminar.
        self.cap.release()

        # Cierra todas las ventanas de OpenCV.
        cv2.destroyAllWindows()


# Punto de entrada del programa.
# Ejecuta la captura solamente si este archivo se corre directamente.
if __name__ == "__main__":
    HandDatasetCollector().iniciar()
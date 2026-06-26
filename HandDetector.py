from predictor_dinamicas import PredictorDinamicas
from PIL import Image, ImageDraw, ImageFont
import cv2
import mediapipe as mp
import numpy as np
import joblib
import os
import time
import json

# ==========================================================
# CONFIGURACIÓN GENERAL
# ==========================================================

# Lista de letras válidas que el sistema puede reconocer mediante señas estáticas.
# También se usan más adelante para validar si una predicción puede agregarse al texto.
VALID_LETTERS = [
    "A", "B", "C", "D", "E",
    "F", "G", "H", "I", "J", "k", "L",
    "M", "N", "O", "P", "R",
    "S", "T", "U", "V", "W", "Y", "Z", "Ñ"
]


# En lugar de tener una lista de palabras escrita directamente en el código,
# esta función carga las palabras desde el archivo words.json.
def cargar_palabras(path="words.json"):
    """
    Carga todas las palabras disponibles desde un archivo JSON.

    Parámetros:
        path:
            Ruta del archivo JSON que contiene las palabras.

    Proceso:
        1. Abre el archivo JSON.
        2. Lee el contenido del archivo.
        3. Recorre las listas de palabras agrupadas por letra.
        4. Une todas las palabras en una sola lista.

    Retorna:
        Una lista con todas las palabras disponibles para sugerencias.
    """

    # Abre el archivo JSON usando codificación UTF-8 para soportar acentos y la letra Ñ.
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Lista donde se almacenarán todas las palabras del archivo.
    todas = []

    # Recorre cada letra y su lista de palabras correspondiente.
    for letra, lista in data["palabras"].items():
        # Agrega todas las palabras de esa letra a la lista general.
        todas.extend(lista)

    # Devuelve la lista completa de palabras.
    return todas


# Carga las palabras al iniciar el programa.
WORDS = cargar_palabras()


class HandDetector:
    """
    Clase principal encargada de:
        - Abrir la cámara.
        - Detectar la mano con MediaPipe.
        - Extraer landmarks.
        - Usar el modelo estático para predecir letras.
        - Usar el predictor dinámico para letras con movimiento.
        - Construir texto con las letras reconocidas.
        - Mostrar una interfaz visual con OpenCV.
        - Mostrar sugerencias de palabras.
    """

    def __init__(self):
        """
        Inicializa la cámara, los modelos y las variables necesarias para la interfaz.

        Proceso:
            1. Abre la cámara principal.
            2. Inicializa MediaPipe Hands.
            3. Carga el modelo de señas estáticas.
            4. Crea el predictor de señas dinámicas.
            5. Inicializa variables para el scroll de sugerencias.
        """

        # Abre la cámara principal del equipo.
        self.cap = cv2.VideoCapture(0)

        # Inicializa los módulos de MediaPipe para detección de manos.
        self.mp_hands = mp.solutions.hands

        # Configura MediaPipe Hands para detectar como máximo una mano.
        self.hands = self.mp_hands.Hands(max_num_hands=1)

        # Herramienta de MediaPipe para dibujar landmarks y conexiones de la mano.
        self.mp_drawing = mp.solutions.drawing_utils

        # Ruta del modelo entrenado para reconocimiento de señas estáticas.
        model_path = "vowels_model.joblib"

        # Verifica si el modelo existe antes de intentar cargarlo.
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"No encontré {model_path}. Primero ejecuta:\n"
                "  python vowels.py\n"
                "  python model.py\n"
            )

        # Carga el modelo entrenado desde el archivo .joblib.
        self.model = joblib.load(model_path)

        # Umbral mínimo de confianza para aceptar una predicción estática.
        self.threshold = 0.82

        # Modelo nuevo para señas dinamicas.
        self.predictor_dinamicas = PredictorDinamicas()

        # Índice usado para controlar el desplazamiento vertical de sugerencias.
        self.scroll_index = 0

        # Cantidad máxima de sugerencias secundarias visibles al mismo tiempo.
        self.max_visible_suggestions = 4

        # Coordenadas del área donde se detectará el scroll del mouse.
        self.sug_x1 = 230
        self.sug_y1 = 700
        self.sug_x2 = 1085
        self.sug_y2 = 700

        # Total de sugerencias secundarias disponibles.
        self.total_secondary_suggestions = 0

    def _features(self, hand_landmarks, handedness_label: str):
        """
        Extrae y normaliza los landmarks de la mano detectada.

        Parámetros:
            hand_landmarks:
                Landmarks de la mano detectada por MediaPipe.

            handedness_label:
                Etiqueta que indica si la mano detectada es derecha o izquierda.

        Proceso:
            1. Convierte los landmarks en un arreglo NumPy.
            2. Toma la muñeca como punto de origen.
            3. Normaliza la escala usando el landmark 9.
            4. Invierte el eje X si la mano detectada es derecha.
            5. Devuelve los landmarks aplanados en forma compatible con el modelo.

        Retorna:
            Un arreglo con forma (1, 63), listo para ser usado por el modelo.
        """

        # Convierte los 21 landmarks de MediaPipe en una matriz con coordenadas x, y, z.
        pts = np.array(
            [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark],
            dtype=np.float32
        )

        # Centra la mano usando el landmark 0, que corresponde a la muñeca.
        pts = pts - pts[0]

        # Calcula la escala tomando como referencia el landmark 9.
        # Se suma 1e-6 para evitar una división entre cero.
        scale = np.linalg.norm(pts[9]) + 1e-6

        # Normaliza los puntos para reducir variaciones de tamaño.
        pts = pts / scale

        # Si la mano detectada es derecha, se invierte el eje X.
        # Esto ayuda a unificar la orientación de la mano.
        if handedness_label.lower() == "right":
            pts[:, 0] *= -1

        # Aplana los landmarks y les da forma de una sola muestra para el modelo.
        return pts.flatten().reshape(1, -1)

    def mouse_callback(self, event, x, y, flags, param):
        """
        Controla el desplazamiento con la rueda del mouse dentro del área de sugerencias.

        Parámetros:
            event:
                Tipo de evento del mouse detectado por OpenCV.

            x, y:
                Posición actual del mouse.

            flags:
                Información adicional del evento, usada aquí para saber la dirección del scroll.

            param:
                Parámetro adicional requerido por OpenCV.

        Función:
            Permite subir o bajar en la lista de sugerencias secundarias.
        """

        # Verifica si el mouse está dentro del área de sugerencias.
        dentro_area = (
            self.sug_x1 <= x <= self.sug_x2 and
            self.sug_y1 <= y <= self.sug_y2
        )

        # Solo aplica desplazamiento si el evento es la rueda del mouse y está dentro del área.
        if event == cv2.EVENT_MOUSEWHEEL and dentro_area:
            # Scroll hacia arriba.
            if flags > 0:
                if self.scroll_index > 0:
                    self.scroll_index -= 1

            # Scroll hacia abajo.
            elif flags < 0:
                if self.scroll_index + self.max_visible_suggestions < self.total_secondary_suggestions:
                    self.scroll_index += 1

    def draw_text_pil(self, img, text, position, font_size=32, color=(0, 0, 0)):
        """
        Dibuja texto sobre una imagen usando PIL.

        Esta función se usa principalmente para poder mostrar caracteres especiales,
        como acentos o la letra Ñ, que OpenCV no siempre maneja bien con cv2.putText.

        Parámetros:
            img:
                Imagen en formato OpenCV BGR.

            text:
                Texto que se desea dibujar.

            position:
                Posición donde se dibujará el texto.

            font_size:
                Tamaño de fuente.

            color:
                Color del texto en formato RGB.

        Retorna:
            Imagen nuevamente en formato OpenCV BGR.
        """

        # Convierte la imagen de BGR a RGB para trabajar con PIL.
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Convierte la imagen NumPy en imagen PIL.
        pil_img = Image.fromarray(img_rgb)

        # Crea el objeto para dibujar sobre la imagen.
        draw = ImageDraw.Draw(pil_img)

        # Variable donde se guardará la fuente encontrada.
        font = None

        # Lista de posibles fuentes disponibles en Windows o en el entorno actual.
        font_candidates = [
            "C:/Windows/Fonts/arial.ttf",
            "C:/Windows/Fonts/calibri.ttf",
            "arial.ttf",
            "calibri.ttf",
        ]

        # Intenta cargar una fuente TrueType de la lista.
        for font_path in font_candidates:
            try:
                font = ImageFont.truetype(font_path, font_size)
                break
            except:
                pass

        # Si no se encontró ninguna fuente, usa la fuente por defecto de PIL.
        if font is None:
            font = ImageFont.load_default()

        # Dibuja el texto en la posición indicada.
        draw.text(position, text, font=font, fill=color)

        # Convierte la imagen de regreso a formato OpenCV BGR.
        img_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

        return img_bgr

    def wrap_text_lines(self, text, max_width, font, font_scale, thickness):
        """
        Divide un texto en varias líneas para que no exceda un ancho máximo.

        Parámetros:
            text:
                Texto que se desea mostrar.

            max_width:
                Ancho máximo permitido por línea.

            font:
                Fuente de OpenCV usada para calcular el tamaño.

            font_scale:
                Escala de la fuente.

            thickness:
                Grosor del texto.

        Proceso:
            1. Divide el texto en palabras.
            2. Intenta agregar palabras a una línea mientras quepan.
            3. Si una palabra no cabe completa, la divide por caracteres.
            4. Devuelve una lista de líneas listas para dibujarse.

        Retorna:
            Lista de líneas de texto.
        """

        # Si el texto está vacío, devuelve una línea vacía.
        if not text.strip():
            return [""]

        # Divide el texto por espacios.
        words = text.split(" ")

        # Lista donde se guardarán las líneas finales.
        lines = []

        # Línea que se está construyendo actualmente.
        current_line = ""

        # Recorre palabra por palabra.
        for word in words:
            # Prueba cómo quedaría la línea si se agrega la palabra actual.
            test_line = word if current_line == "" else current_line + " " + word

            # Calcula el ancho en pixeles de la línea de prueba.
            (w, _), _ = cv2.getTextSize(test_line, font, font_scale, thickness)

            # Si la línea cabe dentro del ancho máximo, se conserva.
            if w <= max_width:
                current_line = test_line
            else:
                # Si la línea actual ya tiene contenido, se agrega a la lista.
                if current_line:
                    lines.append(current_line)
                    current_line = ""

                # Calcula el ancho de la palabra actual.
                (word_w, _), _ = cv2.getTextSize(word, font, font_scale, thickness)

                # Si la palabra cabe sola, empieza una nueva línea con esa palabra.
                if word_w <= max_width:
                    current_line = word
                else:
                    # Si la palabra es demasiado larga, se divide carácter por carácter.
                    chunk = ""

                    for ch in word:
                        test_chunk = chunk + ch
                        (chunk_w, _), _ = cv2.getTextSize(
                            test_chunk, font, font_scale, thickness
                        )

                        # Si el fragmento cabe, se conserva.
                        if chunk_w <= max_width:
                            chunk = test_chunk
                        else:
                            # Si ya no cabe, se guarda el fragmento anterior.
                            if chunk:
                                lines.append(chunk)

                            # Inicia un nuevo fragmento con el carácter actual.
                            chunk = ch

                    # Si queda un fragmento pendiente, se vuelve la línea actual.
                    if chunk:
                        current_line = chunk

        # Agrega la última línea si quedó contenido pendiente.
        if current_line:
            lines.append(current_line)

        return lines

    def draw_ui(self, frame_camera, letra_actual, texto_concatenado, sugerencias, sistema_activo=True):
        """
        Dibuja toda la interfaz visual del sistema.

        Parámetros:
            frame_camera:
                Frame capturado desde la cámara.

            letra_actual:
                Letra o acción confirmada actualmente.

            texto_concatenado:
                Texto formado por las letras reconocidas.

            sugerencias:
                Lista de palabras sugeridas según el texto actual.

            sistema_activo:
                Indica si el sistema debe mostrarse como activo o inactivo.

        Retorna:
            Un canvas completo con la interfaz dibujada.
        """

        # Tamaño del lienzo principal de la interfaz.
        ui_h, ui_w = 900, 1600

        # Crea un canvas gris claro.
        canvas = np.full((ui_h, ui_w, 3), 242, dtype=np.uint8)

        # Colores usados en la interfaz.
        COLOR_TITULO = (35, 35, 35)
        COLOR_SUB = (120, 120, 120)
        COLOR_BORDE = (225, 225, 225)
        COLOR_CAJA = (255, 255, 255)
        COLOR_SOMBRA = (220, 220, 220)
        COLOR_AZUL = (255, 120, 0)
        COLOR_VERDE = (120, 210, 130)
        COLOR_GRIS_PALIDO = (190, 190, 190)
        COLOR_LINEA = (235, 235, 235)
        COLOR_SCROLL_BG = (230, 230, 230)
        COLOR_SCROLL_THUMB = (160, 160, 160)

        # Título principal y subtítulo.
        cv2.putText(canvas, "Deteccion de mano", (70, 55), cv2.FONT_HERSHEY_SIMPLEX, 1.2, COLOR_TITULO, 3)
        cv2.putText(canvas, "Prototipo v1", (72, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_SUB, 1)

        # Indicador visual del estado del sistema.
        estado_color = COLOR_VERDE if sistema_activo else (170, 170, 170)
        cv2.circle(canvas, (1110, 50), 7, estado_color, -1)
        cv2.putText(
            canvas,
            "Sistema Activo" if sistema_activo else "Sistema Inactivo",
            (1125, 55),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            COLOR_SUB,
            1,
        )

        # Encabezado de la sección de detección.
        cv2.putText(canvas, "DETECCION", (75, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)

        # Coordenadas de la caja donde se muestra la letra actual.
        x1, y1, x2, y2 = 70, 145, 180, 255

        # Dibuja sombra, caja blanca y borde de la letra actual.
        cv2.rectangle(canvas, (x1 + 6, y1 + 6), (x2 + 6, y2 + 6), COLOR_SOMBRA, -1)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), COLOR_CAJA, -1)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), COLOR_BORDE, 1)

        # Define cómo se mostrará visualmente la predicción actual.
        if letra_actual == "BORRAR":
            letra_show = "⌫"
        elif letra_actual == "ESPACIO":
            letra_show = "_"
        else:
            letra_show = letra_actual if letra_actual and len(letra_actual) == 1 else "-"

        # Calcula el tamaño del texto para centrarlo dentro de la caja.
        (lw, lh), _ = cv2.getTextSize(letra_show, cv2.FONT_HERSHEY_SIMPLEX, 2.1, 4)
        lx = x1 + ((x2 - x1) - lw) // 2
        ly = y1 + ((y2 - y1) + lh) // 2

        # Dibuja la letra actual.
        cv2.putText(canvas, letra_show, (lx, ly), cv2.FONT_HERSHEY_SIMPLEX, 2.1, COLOR_AZUL, 4)

        # Etiqueta inferior de la caja de detección.
        cv2.rectangle(canvas, (82, 268), (150, 290), (232, 232, 232), -1)
        cv2.putText(canvas, "ACTUAL", (92, 284), cv2.FONT_HERSHEY_SIMPLEX, 0.38, COLOR_SUB, 1)

        # Coordenadas del panel donde se muestra la cámara.
        cam_x1, cam_y1 = 235, 120
        cam_x2, cam_y2 = 1135, 850

        # Dibuja sombra y fondo oscuro del panel de cámara.
        cv2.rectangle(canvas, (cam_x1 + 8, cam_y1 + 8), (cam_x2 + 8, cam_y2 + 8), (215, 215, 215), -1)
        cv2.rectangle(canvas, (cam_x1, cam_y1), (cam_x2, cam_y2), (15, 15, 15), -1)

        # Calcula el tamaño disponible para ajustar el frame de la cámara.
        panel_w = cam_x2 - cam_x1
        panel_h = cam_y2 - cam_y1
        fh, fw = frame_camera.shape[:2]

        # Escala el frame para que quepa dentro del panel sin deformarse.
        scale = min(panel_w / fw, panel_h / fh)
        new_w = int(fw * scale)
        new_h = int(fh * scale)

        # Redimensiona el frame capturado.
        frame_resized = cv2.resize(frame_camera, (new_w, new_h))

        # Calcula los desplazamientos para centrar el frame dentro del panel.
        offset_x = cam_x1 + (panel_w - new_w) // 2
        offset_y = cam_y1 + (panel_h - new_h) // 2

        # Inserta el frame redimensionado dentro del canvas.
        canvas[offset_y:offset_y + new_h, offset_x:offset_x + new_w] = frame_resized

        # Dibuja el borde del panel de cámara.
        cv2.rectangle(canvas, (cam_x1, cam_y1), (cam_x2, cam_y2), COLOR_BORDE, 1)

        # Coordenadas del panel derecho donde se muestra el texto escrito.
        search_x1, search_y1 = 1145, 120
        search_x2 = 1590

        # Parámetros visuales del cuadro de texto.
        padding_x = 25
        padding_y = 22
        line_gap = 42
        font_text = cv2.FONT_HERSHEY_SIMPLEX
        font_scale_text = 1.0
        thickness_text = 2

        # Texto que se mostrará en el cuadro.
        texto_show = texto_concatenado if texto_concatenado else ""

        # Crea un cursor parpadeante usando el tiempo actual.
        mostrar_cursor = int(time.time() * 2) % 2 == 0
        cursor = "|" if mostrar_cursor else ""

        # Texto final a dibujar, incluyendo el cursor.
        texto_dibujar = texto_show + cursor

        # Ancho máximo del texto dentro del panel.
        max_text_width = (search_x2 - search_x1) - (padding_x * 2)

        # Divide el texto en líneas para que no se salga del cuadro.
        lineas = self.wrap_text_lines(
            texto_dibujar,
            max_text_width,
            font_text,
            font_scale_text,
            thickness_text,
        )

        # Sugerencias fijas desde y=480 — el cuadro de texto nunca las tapa
        SUG_Y_FIJO = 480

        # Calcula cuántas líneas máximo caben antes del área de sugerencias.
        max_lineas = max(1, (SUG_Y_FIJO - search_y1 - padding_y * 2 - 20) // line_gap)

        # Limita las líneas visibles para que no invadan el área de sugerencias.
        lineas = lineas[:max_lineas]

        # Calcula la altura del cuadro de texto.
        min_box_height = 195
        contenido_h = padding_y * 2 + len(lineas) * line_gap
        box_h = max(min_box_height, contenido_h)
        search_y2 = min(search_y1 + box_h, SUG_Y_FIJO - 20)

        # Dibuja el cuadro de texto.
        cv2.rectangle(canvas, (search_x1, search_y1), (search_x2, search_y2), COLOR_CAJA, -1)
        cv2.rectangle(canvas, (search_x1, search_y1), (search_x2, search_y2), COLOR_BORDE, 1)

        # Dibuja cada línea de texto usando PIL para soportar caracteres especiales.
        for i, linea in enumerate(lineas):
            y_text = search_y1 + padding_y + i * line_gap
            canvas = self.draw_text_pil(
                canvas,
                linea,
                (search_x1 + padding_x, y_text),
                font_size=36,
                color=(0, 120, 255),
            )

        # Si hay texto y sugerencias, muestra una sugerencia principal como autocompletado.
        if texto_show and sugerencias:
            sugerencia_principal = sugerencias[0]

            # Comparar solo contra la ultima palabra, no todo el texto
            ultimo_fragmento_display = texto_show.split(" ")[-1]

            # Calcula el resto de la palabra sugerida.
            if sugerencia_principal.startswith(ultimo_fragmento_display) and ultimo_fragmento_display:
                resto = sugerencia_principal[len(ultimo_fragmento_display):]
            else:
                resto = ""

            # Si existe un resto sugerido, lo muestra en gris.
            if resto and lineas:
                ultima_linea = lineas[-1].replace(cursor, "")
                (w_last, _), _ = cv2.getTextSize(
                    ultima_linea,
                    font_text,
                    font_scale_text,
                    thickness_text,
                )

                # Posición donde inicia el texto de autocompletado.
                x_resto = search_x1 + padding_x + w_last + 10
                y_resto = search_y1 + padding_y + 28 + (len(lineas) - 1) * line_gap

                # Solo dibuja el resto si cabe dentro del cuadro.
                if x_resto < search_x2 - 40:
                    canvas = self.draw_text_pil(
                        canvas,
                        resto,
                        (x_resto, y_resto - 30),
                        font_size=32,
                        color=(190, 190, 190),
                    )

        # Coordenadas del panel de sugerencias secundarias.
        sug_x1 = search_x1
        sug_y1 = SUG_Y_FIJO  # Posición fija — no depende del tamaño del cuadro de texto
        sug_w = search_x2 - search_x1
        sug_item_h = 34

        # Obtiene todas las sugerencias excepto la principal.
        sugerencias_secundarias = sugerencias[1:] if len(sugerencias) > 1 else []

        # Guarda cuántas sugerencias secundarias existen.
        self.total_secondary_suggestions = len(sugerencias_secundarias)

        # Obtiene solo las sugerencias visibles según el scroll actual.
        sugerencias_visibles = sugerencias_secundarias[
            self.scroll_index:self.scroll_index + self.max_visible_suggestions
        ]

        # Si hay sugerencias visibles, dibuja el panel.
        if sugerencias_visibles:
            total_h = len(sugerencias_visibles) * sug_item_h

            # Actualiza las coordenadas reales del área de sugerencias para el mouse.
            self.sug_x1 = sug_x1
            self.sug_y1 = sug_y1
            self.sug_x2 = sug_x1 + sug_w
            self.sug_y2 = sug_y1 + total_h

            # Dibuja fondo y borde del panel de sugerencias.
            cv2.rectangle(canvas, (sug_x1, sug_y1), (sug_x1 + sug_w, sug_y1 + total_h), COLOR_CAJA, -1)
            cv2.rectangle(canvas, (sug_x1, sug_y1), (sug_x1 + sug_w, sug_y1 + total_h), COLOR_BORDE, 1)

            # Dibuja cada sugerencia visible.
            for i, s in enumerate(sugerencias_visibles):
                yy = sug_y1 + (i * sug_item_h)

                # Dibuja líneas separadoras entre sugerencias.
                if i > 0:
                    cv2.line(canvas, (sug_x1, yy), (sug_x1 + sug_w, yy), COLOR_LINEA, 1)

                # Escribe la sugerencia.
                cv2.putText(canvas, s, (sug_x1 + 20, yy + 23), cv2.FONT_HERSHEY_SIMPLEX, 0.65, COLOR_TITULO, 2)

            # Si hay más sugerencias que las visibles, dibuja una barra de scroll.
            if len(sugerencias_secundarias) > self.max_visible_suggestions:
                bar_x1 = sug_x1 + sug_w - 12
                bar_x2 = sug_x1 + sug_w - 6
                bar_y1 = sug_y1 + 4
                bar_y2 = sug_y1 + total_h - 4

                # Fondo de la barra de desplazamiento.
                cv2.rectangle(canvas, (bar_x1, bar_y1), (bar_x2, bar_y2), COLOR_SCROLL_BG, -1)

                # Datos para calcular el tamaño del indicador de scroll.
                total_items = len(sugerencias_secundarias)
                visible_items = self.max_visible_suggestions

                # Calcula la altura del indicador proporcional a los elementos visibles.
                thumb_h = max(18, int((visible_items / total_items) * (bar_y2 - bar_y1)))

                # Calcula el máximo desplazamiento posible.
                max_scroll = max(1, total_items - visible_items)

                # Calcula la posición vertical del indicador.
                thumb_y = bar_y1 + int(
                    (self.scroll_index / max_scroll) * ((bar_y2 - bar_y1) - thumb_h)
                )

                # Dibuja el indicador de scroll.
                cv2.rectangle(canvas, (bar_x1, thumb_y), (bar_x2, thumb_y + thumb_h), COLOR_SCROLL_THUMB, -1)
        else:
            # Si no hay sugerencias visibles, actualiza el área a valores vacíos.
            self.sug_x1 = sug_x1
            self.sug_y1 = sug_y1
            self.sug_x2 = sug_x1 + sug_w
            self.sug_y2 = sug_y1
            self.total_secondary_suggestions = 0
            self.scroll_index = 0

        # Devuelve el canvas completo con toda la interfaz dibujada.
        return canvas

    def iniciar(self):
        """
        Inicia el ciclo principal de detección.

        Proceso:
            1. Lee frames de la cámara.
            2. Detecta landmarks de la mano.
            3. Obtiene predicciones estáticas y dinámicas.
            4. Confirma letras después de varios frames.
            5. Agrega letras, espacios o borra texto según la predicción.
            6. Genera sugerencias de palabras.
            7. Dibuja la interfaz completa.
            8. Permite salir con ESC o borrar con Backspace.
        """

        # Letra detectada en el frame actual.
        letra_actual = ""

        # Letra confirmada después de mantenerse estable varios frames.
        letra_confirmada = ""

        # Contador de frames consecutivos con la misma predicción.
        contador_frames = 0

        # Cantidad de frames necesarios para confirmar una letra.
        frames_requeridos = 5

        # Texto acumulado por el usuario.
        texto = ""

        # Última letra o acción agregada al texto.
        ultima_letra_agregada = ""

        # Tiempo mínimo entre letras agregadas para evitar repeticiones rápidas.
        cooldown_seg = 0.5

        # Momento en que se agregó la última letra o acción.
        ultimo_add_time = 0.0

        # Nombre de la ventana principal.
        window_name = "Deteccion de mano - Prototipo v1"

        # Crea la ventana de OpenCV.
        cv2.namedWindow(window_name)

        # Asocia el callback del mouse para controlar el scroll de sugerencias.
        cv2.setMouseCallback(window_name, self.mouse_callback)

        # Ciclo principal de la aplicación.
        while True:
            # Captura un frame de la cámara.
            ret, frame = self.cap.read()

            # Si no se puede leer el frame, termina el ciclo.
            if not ret:
                break

            # Convierte el frame de BGR a RGB para MediaPipe.
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Procesa el frame para detectar la mano.
            results = self.hands.process(image)

            # Texto de predicción estática del frame actual.
            pred_text = ""

            # Si MediaPipe detectó landmarks de una mano.
            if results.multi_hand_landmarks:
                # Toma la primera mano detectada.
                hand_lms = results.multi_hand_landmarks[0]

                # Valor por defecto de lateralidad de mano.
                handedness_label = "unknown"

                # Si MediaPipe detectó si es mano izquierda o derecha, obtiene esa etiqueta.
                if results.multi_handedness:
                    handedness_label = results.multi_handedness[0].classification[0].label

                # Dibuja los landmarks sobre el frame original.
                self.mp_drawing.draw_landmarks(frame, hand_lms, self.mp_hands.HAND_CONNECTIONS)

                # funcion de señas dinamicas
                letra_dinamica = self.predictor_dinamicas.actualizar(hand_lms, handedness_label)

                # Si el predictor dinámico devuelve una letra, se agrega al texto.
                if letra_dinamica:
                    texto += letra_dinamica
                    ultima_letra_agregada = letra_dinamica
                    ultimo_add_time = time.time()

                # Extrae las características normalizadas para el modelo estático.
                X = self._features(hand_lms, handedness_label)

                # Si el modelo permite obtener probabilidades, se usa la confianza de predicción.
                if hasattr(self.model, "predict_proba"):
                    # Obtiene las probabilidades de cada clase.
                    probs = self.model.predict_proba(X)[0]

                    # Obtiene el índice de la clase con mayor probabilidad.
                    idx = int(np.argmax(probs))

                    # Confianza de la predicción.
                    conf = float(probs[idx])

                    # Etiqueta predicha por el modelo.
                    label = self.model.classes_[idx]

                    # Acepta letras válidas, Borrar y Espacio — siempre que supere el threshold
                    etiquetas_validas_estatico = set(VALID_LETTERS) | {"Borrar", "Espacio"}

                    # Solo acepta la predicción si supera el umbral y pertenece a las etiquetas permitidas.
                    pred_text = label if (conf >= self.threshold and label in etiquetas_validas_estatico) else ""
                else:
                    # Si el modelo no tiene predict_proba, usa directamente predict.
                    pred_text = str(self.model.predict(X)[0])
            else:
                # Si no hay mano detectada, limpia el estado interno del predictor dinámico.
                self.predictor_dinamicas.limpiar()

            # Verifica si la predicción actual se mantiene igual en frames consecutivos.
            if pred_text == letra_actual:
                contador_frames += 1
            else:
                # Si la predicción cambia, reinicia el contador.
                contador_frames = 0
                letra_actual = pred_text

            # Si la predicción se mantuvo suficientes frames, se confirma.
            if contador_frames >= frames_requeridos:
                letra_confirmada = letra_actual

            # Obtiene el tiempo actual para controlar cooldowns.
            ahora = time.time()

            # Si la letra confirmada es una letra válida, intenta agregarla al texto.
            if letra_confirmada in VALID_LETTERS:
                # Evita repetir la misma letra inmediatamente y respeta el cooldown.
                if letra_confirmada != ultima_letra_agregada and (ahora - ultimo_add_time) >= cooldown_seg:
                    texto += letra_confirmada
                    ultima_letra_agregada = letra_confirmada
                    ultimo_add_time = ahora

            # Si la acción confirmada es borrar.
            elif letra_confirmada == "Borrar":
                # Primera vez borrando: 0.8s de espera; si sigue borrando: 1.2s entre cada letra
                cooldown_borrar = 0.8 if ultima_letra_agregada != "Borrar" else 1.2

                # Borra un carácter si se cumple el tiempo de espera.
                if (ahora - ultimo_add_time) >= cooldown_borrar:
                    texto = texto[:-1]
                    ultima_letra_agregada = letra_confirmada
                    ultimo_add_time = ahora

            # Si la acción confirmada es espacio.
            elif letra_confirmada == "Espacio":
                # Agrega espacio solo si no se acaba de agregar y se cumple el cooldown.
                if letra_confirmada != ultima_letra_agregada and (ahora - ultimo_add_time) >= 0.5:
                    # Evita agregar espacios dobles consecutivos.
                    if texto and texto[-1] != " ":
                        texto += " "
                        ultima_letra_agregada = letra_confirmada
                        ultimo_add_time = ahora
            else:
                # Si no hay una letra o acción válida, permite que la siguiente detección pueda agregarse.
                ultima_letra_agregada = ""

            # Voltea el frame horizontalmente para mostrarlo como espejo en la interfaz.
            frame = cv2.flip(frame, 1)

            # Convierte el texto a mayúsculas para buscar sugerencias.
            pref = texto.upper()

            # Lista de sugerencias generadas.
            sugerencias = []

            # Solo busca sugerencias si hay texto escrito.
            if pref.strip():
                # Obtiene la última palabra o fragmento escrito.
                ultimo_fragmento = pref.split(" ")[-1]

                # Si existe un fragmento, busca palabras que inicien con ese fragmento.
                if ultimo_fragmento:
                    sugerencias = sorted(
                        [w for w in WORDS if w.startswith(ultimo_fragmento)],
                        key=len,
                    )[:20]

            # Calcula cuántas sugerencias secundarias existen.
            total_secundarias = max(0, len(sugerencias) - 1)

            # Calcula el máximo valor permitido para el scroll.
            max_scroll = max(0, total_secundarias - self.max_visible_suggestions)

            # Corrige el índice del scroll si se pasa del máximo permitido.
            if self.scroll_index > max_scroll:
                self.scroll_index = max_scroll

            # Dibuja toda la interfaz.
            ui = self.draw_ui(frame, letra_confirmada, texto, sugerencias, sistema_activo=True)

            # Muestra la interfaz en pantalla.
            cv2.imshow(window_name, ui)

            # Lee la tecla presionada.
            key = cv2.waitKey(1) & 0xFF

            # ESC: salir del programa.
            if key == 27:
                break

            # Backspace: borrar último carácter del texto.
            elif key == 8:
                texto = texto[:-1]
                self.scroll_index = 0

        # Libera la cámara.
        self.cap.release()

        # Cierra todas las ventanas de OpenCV.
        cv2.destroyAllWindows()


# Punto de entrada del programa.
# Solo se ejecuta si este archivo se corre directamente.
if __name__ == "__main__":
    # Crea una instancia del detector de manos.
    detector = HandDetector()

    # Inicia el sistema de detección.
    detector.iniciar()
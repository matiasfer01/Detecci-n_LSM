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

VALID_LETTERS = [
    "A", "B", "C", "D", "E",
    "F", "G", "H", "I", "J", "k", "L",
    "M", "N", "O", "P", "R",
    "S", "T", "U", "V", "W", "Y", "Z", "Ñ"
]

# En lugar de la lista WORDS hardcodeada, carga así:
def cargar_palabras(path="words.json"):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # Aplana todas las listas de cada letra en una sola lista
    todas = []
    for letra, lista in data["palabras"].items():
        todas.extend(lista)
    return todas

WORDS = cargar_palabras()



class HandDetector:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(max_num_hands=1)
        self.mp_drawing = mp.solutions.drawing_utils

        model_path = "vowels_model.joblib"
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"No encontré {model_path}. Primero ejecuta:\n"
                "  python vowels.py\n"
                "  python model.py\n"
            )

        self.model = joblib.load(model_path)
        self.threshold = 0.82
        
        # Modelo nuevo para señas dinamicas.
        self.predictor_dinamicas = PredictorDinamicas()

        self.scroll_index = 0
        self.max_visible_suggestions = 4

        self.sug_x1 = 230
        self.sug_y1 = 700
        self.sug_x2 = 1085
        self.sug_y2 = 700
        self.total_secondary_suggestions = 0


    def _features(self, hand_landmarks, handedness_label: str):
        pts = np.array(
            [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark],
            dtype=np.float32
        )

        pts = pts - pts[0]
        scale = np.linalg.norm(pts[9]) + 1e-6
        pts = pts / scale

        if handedness_label.lower() == "right":
            pts[:, 0] *= -1

        return pts.flatten().reshape(1, -1)

    def mouse_callback(self, event, x, y, flags, param):
        dentro_area = (
            self.sug_x1 <= x <= self.sug_x2 and
            self.sug_y1 <= y <= self.sug_y2
        )

        if event == cv2.EVENT_MOUSEWHEEL and dentro_area:
            if flags > 0:
                if self.scroll_index > 0:
                    self.scroll_index -= 1
            elif flags < 0:
                if self.scroll_index + self.max_visible_suggestions < self.total_secondary_suggestions:
                    self.scroll_index += 1

    def draw_text_pil(self, img, text, position, font_size=32, color=(0, 0, 0)):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        draw = ImageDraw.Draw(pil_img)

        font = None
        font_candidates = [
            "C:/Windows/Fonts/arial.ttf",
            "C:/Windows/Fonts/calibri.ttf",
            "arial.ttf",
            "calibri.ttf",
        ]

        for font_path in font_candidates:
            try:
                font = ImageFont.truetype(font_path, font_size)
                break
            except:
                pass

        if font is None:
            font = ImageFont.load_default()

        draw.text(position, text, font=font, fill=color)

        img_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        return img_bgr

    def wrap_text_lines(self, text, max_width, font, font_scale, thickness):
        if not text.strip():
            return [""]

        words = text.split(" ")
        lines = []
        current_line = ""

        for word in words:
            test_line = word if current_line == "" else current_line + " " + word
            (w, _), _ = cv2.getTextSize(test_line, font, font_scale, thickness)

            if w <= max_width:
                current_line = test_line
            else:
                if current_line:
                    lines.append(current_line)
                    current_line = ""

                (word_w, _), _ = cv2.getTextSize(word, font, font_scale, thickness)

                if word_w <= max_width:
                    current_line = word
                else:
                    chunk = ""
                    for ch in word:
                        test_chunk = chunk + ch
                        (chunk_w, _), _ = cv2.getTextSize(
                            test_chunk, font, font_scale, thickness
                        )

                        if chunk_w <= max_width:
                            chunk = test_chunk
                        else:
                            if chunk:
                                lines.append(chunk)
                            chunk = ch

                    if chunk:
                        current_line = chunk

        if current_line:
            lines.append(current_line)

        return lines

    def draw_ui(self, frame_camera, letra_actual, texto_concatenado, sugerencias, sistema_activo=True):
        ui_h, ui_w = 900, 1600
        canvas = np.full((ui_h, ui_w, 3), 242, dtype=np.uint8)

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

        cv2.putText(canvas, "Deteccion de mano", (70, 55), cv2.FONT_HERSHEY_SIMPLEX, 1.2, COLOR_TITULO, 3)
        cv2.putText(canvas, "Prototipo v1", (72, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_SUB, 1)

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

        cv2.putText(canvas, "DETECCION", (75, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)

        x1, y1, x2, y2 = 70, 145, 180, 255
        cv2.rectangle(canvas, (x1 + 6, y1 + 6), (x2 + 6, y2 + 6), COLOR_SOMBRA, -1)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), COLOR_CAJA, -1)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), COLOR_BORDE, 1)

        if letra_actual == "BORRAR":
            letra_show = "⌫"
        elif letra_actual == "ESPACIO":
            letra_show = "_"
        else:
            letra_show = letra_actual if letra_actual and len(letra_actual) == 1 else "-"

        (lw, lh), _ = cv2.getTextSize(letra_show, cv2.FONT_HERSHEY_SIMPLEX, 2.1, 4)
        lx = x1 + ((x2 - x1) - lw) // 2
        ly = y1 + ((y2 - y1) + lh) // 2

        cv2.putText(canvas, letra_show, (lx, ly), cv2.FONT_HERSHEY_SIMPLEX, 2.1, COLOR_AZUL, 4)

        cv2.rectangle(canvas, (82, 268), (150, 290), (232, 232, 232), -1)
        cv2.putText(canvas, "ACTUAL", (92, 284), cv2.FONT_HERSHEY_SIMPLEX, 0.38, COLOR_SUB, 1)

        cam_x1, cam_y1 = 235, 120
        cam_x2, cam_y2 = 1135, 850

        cv2.rectangle(canvas, (cam_x1 + 8, cam_y1 + 8), (cam_x2 + 8, cam_y2 + 8), (215, 215, 215), -1)
        cv2.rectangle(canvas, (cam_x1, cam_y1), (cam_x2, cam_y2), (15, 15, 15), -1)

        panel_w = cam_x2 - cam_x1
        panel_h = cam_y2 - cam_y1
        fh, fw = frame_camera.shape[:2]

        scale = min(panel_w / fw, panel_h / fh)
        new_w = int(fw * scale)
        new_h = int(fh * scale)

        frame_resized = cv2.resize(frame_camera, (new_w, new_h))
        offset_x = cam_x1 + (panel_w - new_w) // 2
        offset_y = cam_y1 + (panel_h - new_h) // 2

        canvas[offset_y:offset_y + new_h, offset_x:offset_x + new_w] = frame_resized
        cv2.rectangle(canvas, (cam_x1, cam_y1), (cam_x2, cam_y2), COLOR_BORDE, 1)

        search_x1, search_y1 = 1145, 120
        search_x2 = 1590

        padding_x = 25
        padding_y = 22
        line_gap = 42
        font_text = cv2.FONT_HERSHEY_SIMPLEX
        font_scale_text = 1.0
        thickness_text = 2

        texto_show = texto_concatenado if texto_concatenado else ""

        mostrar_cursor = int(time.time() * 2) % 2 == 0
        cursor = "|" if mostrar_cursor else ""

        texto_dibujar = texto_show + cursor
        max_text_width = (search_x2 - search_x1) - (padding_x * 2)

        lineas = self.wrap_text_lines(
            texto_dibujar,
            max_text_width,
            font_text,
            font_scale_text,
            thickness_text,
        )

        # Sugerencias fijas desde y=480 — el cuadro de texto nunca las tapa
        SUG_Y_FIJO = 480
        max_lineas = max(1, (SUG_Y_FIJO - search_y1 - padding_y * 2 - 20) // line_gap)
        lineas = lineas[:max_lineas]

        min_box_height = 195
        contenido_h = padding_y * 2 + len(lineas) * line_gap
        box_h = max(min_box_height, contenido_h)
        search_y2 = min(search_y1 + box_h, SUG_Y_FIJO - 20)

        cv2.rectangle(canvas, (search_x1, search_y1), (search_x2, search_y2), COLOR_CAJA, -1)
        cv2.rectangle(canvas, (search_x1, search_y1), (search_x2, search_y2), COLOR_BORDE, 1)

        for i, linea in enumerate(lineas):
            y_text = search_y1 + padding_y + i * line_gap
            canvas = self.draw_text_pil(
                canvas,
                linea,
                (search_x1 + padding_x, y_text),
                font_size=36,
                color=(0, 120, 255),
            )

        if texto_show and sugerencias:
            sugerencia_principal = sugerencias[0]
            # Comparar solo contra la ultima palabra, no todo el texto
            ultimo_fragmento_display = texto_show.split(" ")[-1]
            if sugerencia_principal.startswith(ultimo_fragmento_display) and ultimo_fragmento_display:
                resto = sugerencia_principal[len(ultimo_fragmento_display):]
            else:
                resto = ""

            if resto and lineas:
                ultima_linea = lineas[-1].replace(cursor, "")
                (w_last, _), _ = cv2.getTextSize(
                    ultima_linea,
                    font_text,
                    font_scale_text,
                    thickness_text,
                )
                x_resto = search_x1 + padding_x + w_last + 10
                y_resto = search_y1 + padding_y + 28 + (len(lineas) - 1) * line_gap

                if x_resto < search_x2 - 40:
                    canvas = self.draw_text_pil(
                        canvas,
                        resto,
                        (x_resto, y_resto - 30),
                        font_size=32,
                        color=(190, 190, 190),
                    )

        sug_x1 = search_x1
        sug_y1 = SUG_Y_FIJO  # Posición fija — no depende del tamaño del cuadro de texto
        sug_w = search_x2 - search_x1
        sug_item_h = 34

        sugerencias_secundarias = sugerencias[1:] if len(sugerencias) > 1 else []
        self.total_secondary_suggestions = len(sugerencias_secundarias)

        sugerencias_visibles = sugerencias_secundarias[
            self.scroll_index:self.scroll_index + self.max_visible_suggestions
        ]

        if sugerencias_visibles:
            total_h = len(sugerencias_visibles) * sug_item_h

            self.sug_x1 = sug_x1
            self.sug_y1 = sug_y1
            self.sug_x2 = sug_x1 + sug_w
            self.sug_y2 = sug_y1 + total_h

            cv2.rectangle(canvas, (sug_x1, sug_y1), (sug_x1 + sug_w, sug_y1 + total_h), COLOR_CAJA, -1)
            cv2.rectangle(canvas, (sug_x1, sug_y1), (sug_x1 + sug_w, sug_y1 + total_h), COLOR_BORDE, 1)

            for i, s in enumerate(sugerencias_visibles):
                yy = sug_y1 + (i * sug_item_h)

                if i > 0:
                    cv2.line(canvas, (sug_x1, yy), (sug_x1 + sug_w, yy), COLOR_LINEA, 1)

                cv2.putText(canvas, s, (sug_x1 + 20, yy + 23), cv2.FONT_HERSHEY_SIMPLEX, 0.65, COLOR_TITULO, 2)

            if len(sugerencias_secundarias) > self.max_visible_suggestions:
                bar_x1 = sug_x1 + sug_w - 12
                bar_x2 = sug_x1 + sug_w - 6
                bar_y1 = sug_y1 + 4
                bar_y2 = sug_y1 + total_h - 4

                cv2.rectangle(canvas, (bar_x1, bar_y1), (bar_x2, bar_y2), COLOR_SCROLL_BG, -1)

                total_items = len(sugerencias_secundarias)
                visible_items = self.max_visible_suggestions

                thumb_h = max(18, int((visible_items / total_items) * (bar_y2 - bar_y1)))
                max_scroll = max(1, total_items - visible_items)
                thumb_y = bar_y1 + int(
                    (self.scroll_index / max_scroll) * ((bar_y2 - bar_y1) - thumb_h)
                )

                cv2.rectangle(canvas, (bar_x1, thumb_y), (bar_x2, thumb_y + thumb_h), COLOR_SCROLL_THUMB, -1)
        else:
            self.sug_x1 = sug_x1
            self.sug_y1 = sug_y1
            self.sug_x2 = sug_x1 + sug_w
            self.sug_y2 = sug_y1
            self.total_secondary_suggestions = 0
            self.scroll_index = 0

        return canvas

    def iniciar(self):
        letra_actual = ""
        letra_confirmada = ""
        contador_frames = 0
        frames_requeridos = 5

        texto = ""
        ultima_letra_agregada = ""
        cooldown_seg = 0.5
        ultimo_add_time = 0.0

        window_name = "Deteccion de mano - Prototipo v1"
        cv2.namedWindow(window_name)
        cv2.setMouseCallback(window_name, self.mouse_callback)

        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(image)

            pred_text = ""

            if results.multi_hand_landmarks:
                hand_lms = results.multi_hand_landmarks[0]

                handedness_label = "unknown"
                if results.multi_handedness:
                    handedness_label = results.multi_handedness[0].classification[0].label

                self.mp_drawing.draw_landmarks(frame, hand_lms, self.mp_hands.HAND_CONNECTIONS)

                                # funcion de señas dinamicas
                letra_dinamica = self.predictor_dinamicas.actualizar(hand_lms, handedness_label)

                if letra_dinamica:
                    texto += letra_dinamica
                    ultima_letra_agregada = letra_dinamica
                    ultimo_add_time = time.time()


                X = self._features(hand_lms, handedness_label)
                

                if hasattr(self.model, "predict_proba"):
                    probs = self.model.predict_proba(X)[0]
                    idx = int(np.argmax(probs))
                    conf = float(probs[idx])
                    label = self.model.classes_[idx]
                    # Acepta letras válidas, Borrar y Espacio — siempre que supere el threshold
                    etiquetas_validas_estatico = set(VALID_LETTERS) | {"Borrar", "Espacio"}
                    pred_text = label if (conf >= self.threshold and label in etiquetas_validas_estatico) else ""
                else:
                    pred_text = str(self.model.predict(X)[0])
            else:
                self.predictor_dinamicas.limpiar()

            if pred_text == letra_actual:
                contador_frames += 1
            else:
                contador_frames = 0
                letra_actual = pred_text

            if contador_frames >= frames_requeridos:
                letra_confirmada = letra_actual

            ahora = time.time()
            if letra_confirmada in VALID_LETTERS:
                if letra_confirmada != ultima_letra_agregada and (ahora - ultimo_add_time) >= cooldown_seg:
                    texto += letra_confirmada
                    ultima_letra_agregada = letra_confirmada
                    ultimo_add_time = ahora

            elif letra_confirmada == "Borrar":
                # Primera vez borrando: 0.8s de espera; si sigue borrando: 1.2s entre cada letra
                cooldown_borrar = 0.8 if ultima_letra_agregada != "Borrar" else 1.2
                if (ahora - ultimo_add_time) >= cooldown_borrar:
                    texto = texto[:-1]
                    ultima_letra_agregada = letra_confirmada
                    ultimo_add_time = ahora

            elif letra_confirmada == "Espacio":
                if letra_confirmada != ultima_letra_agregada and (ahora - ultimo_add_time) >= 0.5:
                    if texto and texto[-1] != " ":
                        texto += " "
                        ultima_letra_agregada = letra_confirmada
                        ultimo_add_time = ahora
            else:
                ultima_letra_agregada = ""

            frame = cv2.flip(frame, 1)

            pref = texto.upper()
            sugerencias = []
            if pref.strip():
                ultimo_fragmento = pref.split(" ")[-1]
                if ultimo_fragmento:
                    sugerencias = sorted(
                        [w for w in WORDS if w.startswith(ultimo_fragmento)],
                        key=len,
                    )[:20]

            total_secundarias = max(0, len(sugerencias) - 1)
            max_scroll = max(0, total_secundarias - self.max_visible_suggestions)
            if self.scroll_index > max_scroll:
                self.scroll_index = max_scroll

            ui = self.draw_ui(frame, letra_confirmada, texto, sugerencias, sistema_activo=True)
            cv2.imshow(window_name, ui)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break
            elif key == 8:
                texto = texto[:-1]
                self.scroll_index = 0

        self.cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    detector = HandDetector()
    detector.iniciar()
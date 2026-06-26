import cv2
import mediapipe as mp
import numpy as np
import os
import time

# ================= CONFIG =================
DATASET_DIR = os.path.join("dataset", "MyDataset")
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
    ord("0"): "Borrar",
    ord("1"): "Espacio"
}
SAVE_DELAY   = 0.20   # delay modo manual (SPACE)
RAFAGA_DELAY = 0.15   # delay entre fotos en modo ráfaga (segundos)
RAFAGA_TOTAL = 200    # cuántas fotos toma la ráfaga automática
IMG_EXT = ".jpg"
# =========================================

class HandDatasetCollector:

    def __init__(self):
        self.cap = cv2.VideoCapture(0)

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            max_num_hands=1,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6
        )
        self.mp_drawing = mp.solutions.drawing_utils

        self.current_label  = "A"
        self.last_save_time = 0.0
        self.counts         = {}

        # Modo ráfaga
        self.rafaga_activa    = False
        self.rafaga_guardadas = 0

        os.makedirs(DATASET_DIR, exist_ok=True)

        all_labels = sorted(set(LABEL_KEYS.values()))
        for lbl in all_labels:
            folder = os.path.join(DATASET_DIR, lbl)
            os.makedirs(folder, exist_ok=True)
            self.counts[lbl] = len([
                f for f in os.listdir(folder)
                if f.lower().endswith((".jpg", ".jpeg", ".png"))
            ])

    def guardar_imagen(self, frame):
        folder   = os.path.join(DATASET_DIR, self.current_label)
        idx      = self.counts[self.current_label] + 1
        filename = f"{self.current_label}_{idx:05d}{IMG_EXT}"
        path     = os.path.join(folder, filename)
        ok       = cv2.imwrite(path, frame)
        if ok:
            self.counts[self.current_label] += 1
            self.last_save_time = time.time()
            return True
        return False

    def iniciar(self):
        print("=== Captura de Dataset ===")
        print("Teclas letra : a b c d e f g h i l m n o p r s t u v w y  |  0=Borrar  1=Espacio")
        print("SPACE        : iniciar rafaga automatica de 200 fotos")
        print("F            : guardar 1 imagen manual")
        print("ESC          : salir")
        print(f"Guardando en : {DATASET_DIR}\n")

        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results   = self.hands.process(image_rgb)

            hand_detected = False
            display_frame = frame.copy()

            if results.multi_hand_landmarks:
                hand_detected = True
                self.mp_drawing.draw_landmarks(
                    display_frame,
                    results.multi_hand_landmarks[0],
                    self.mp_hands.HAND_CONNECTIONS
                )

            display_frame = cv2.flip(display_frame, 1)

            # ---- HUD minimo: solo la letra actual ----
            cv2.putText(display_frame, f"{self.current_label}",
                        (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (255, 255, 255), 3)

            cv2.imshow("Captura Dataset - Prototipo v1", display_frame)

            key = cv2.waitKey(1) & 0xFF

            # Salir
            if key == 27:
                break

            # Cambiar etiqueta (solo si no hay ráfaga activa)
            if not self.rafaga_activa and key in LABEL_KEYS:
                self.current_label = LABEL_KEYS[key]
                print(f"Etiqueta actual: {self.current_label}  ({self.counts[self.current_label]} fotos existentes)")

            # Modo manual F
            if not self.rafaga_activa and key in (ord('f'), ord('F')):
                if not hand_detected:
                    print("No se guardó: no hay mano detectada.")
                    continue
                now = time.time()
                if now - self.last_save_time < SAVE_DELAY:
                    continue
                if self.guardar_imagen(frame):
                    print(f"Guardada manual: {self.current_label} #{self.counts[self.current_label]}")

            # Activar ráfaga con SPACE
            if not self.rafaga_activa and key == 32:
                if not hand_detected:
                    print("No se inició la ráfaga: no hay mano detectada.")
                else:
                    self.rafaga_activa    = True
                    self.rafaga_guardadas = 0
                    print(f"\nRáfaga iniciada para '{self.current_label}' — tomando {RAFAGA_TOTAL} fotos...")

            # Lógica ráfaga automática
            if self.rafaga_activa:
                if not hand_detected:
                    print("Ráfaga pausada: perdí la mano...")
                    continue

                now = time.time()
                if now - self.last_save_time >= RAFAGA_DELAY:
                    if self.guardar_imagen(frame):
                        self.rafaga_guardadas += 1
                        print(f"  Rafaga {self.current_label}: {self.rafaga_guardadas}/{RAFAGA_TOTAL}")

                    if self.rafaga_guardadas >= RAFAGA_TOTAL:
                        self.rafaga_activa = False
                        print(f"\nRáfaga completada: {RAFAGA_TOTAL} fotos de '{self.current_label}' guardadas.\n")

        self.cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    HandDatasetCollector().iniciar()
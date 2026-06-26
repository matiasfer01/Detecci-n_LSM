from collections import deque
import numpy as np
import joblib
import os
import time


class PredictorDinamicas:
    def __init__(self, model_path="dynamic_signs_model.joblib"):
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"No encontré {model_path}. Primero ejecuta entrenar_dinamicas.py"
            )

        self.model = joblib.load(model_path)

        self.frames_por_secuencia = 60
        self.secuencia = deque(maxlen=self.frames_por_secuencia)

        self.threshold = 0.82
        self.cooldown = 1.2
        self.last_time = 0.0

    def extraer_landmarks(self, hand_lms, handedness_label):
        pts = np.array(
            [[lm.x, lm.y, lm.z] for lm in hand_lms.landmark],
            dtype=np.float32
        )

        pts = pts - pts[0]

        scale = np.linalg.norm(pts[9]) + 1e-6
        pts = pts / scale

        if handedness_label.lower() == "right":
            pts[:, 0] *= -1

        return pts.flatten()

    def actualizar(self, hand_lms, handedness_label):
        features = self.extraer_landmarks(hand_lms, handedness_label)
        self.secuencia.append(features)

        if len(self.secuencia) < self.frames_por_secuencia:
            return ""

        X = np.array(self.secuencia, dtype=np.float32).flatten().reshape(1, -1)

        ahora = time.time()
        if ahora - self.last_time < self.cooldown:
            return ""

        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(X)[0]
            idx = int(np.argmax(probs))
            conf = float(probs[idx])
            label = self.model.classes_[idx]

            if conf >= self.threshold and label != "Nada":
                self.last_time = ahora
                self.secuencia.clear()
                return label
        else:
            label = self.model.predict(X)[0]
            self.last_time = ahora
            self.secuencia.clear()
            return label

        return ""

    def limpiar(self):
        self.secuencia.clear()
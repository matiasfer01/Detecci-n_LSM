import cv2
import mediapipe as mp
import numpy as np
import os
import time
import json

# ==============================
# CONFIGURACIÓN
# ==============================

DATASET_DIR = "dataset_dinamicas"
CLASES = ["J", "Ñ", "Z", "X", "K", "Q", "Nada"]  # "Nada" = mano quieta sin seña
FRAMES_POR_SECUENCIA = 60
EJEMPLOS_POR_CLASE = 600

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils


def extraer_landmarks(hand_lms, handedness_label):
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


def guardar_secuencia(clase, secuencia, contador):
    carpeta = os.path.join(DATASET_DIR, clase)
    os.makedirs(carpeta, exist_ok=True)

    ruta = os.path.join(carpeta, f"{clase}_{contador}.json")

    with open(ruta, "w", encoding="utf-8") as f:
        json.dump(secuencia, f)


def contar_existentes(clase):
    carpeta = os.path.join(DATASET_DIR, clase)

    if not os.path.exists(carpeta):
        return 0

    archivos = [
        f for f in os.listdir(carpeta)
        if f.endswith(".json") and f.startswith(f"{clase}_")
    ]

    return len(archivos)


def elegir_clase():
    print("\n=== Clases disponibles ===")
    for i, c in enumerate(CLASES):
        existentes = contar_existentes(c)
        estado = f"{existentes}/{EJEMPLOS_POR_CLASE}"
        completa = " ✅" if existentes >= EJEMPLOS_POR_CLASE else ""
        print(f"  {i + 1}. {c} — {estado}{completa}")

    print("\nEscribe el número o el nombre de la letra que quieres grabar: ", end="")
    entrada = input().strip()

    if entrada.isdigit():
        idx = int(entrada) - 1
        if 0 <= idx < len(CLASES):
            return CLASES[idx]

    for c in CLASES:
        if c.upper() == entrada.upper():
            return c

    print(f"'{entrada}' no es una opción válida.")
    return None


def capturar_clase(cap, hands, clase):
    print(f"\n=== Capturando clase: {clase} ===")

    if clase == "Nada":
        print("Mantén la mano quieta frente a la cámara SIN hacer ninguna seña.")
        print("Puedes variar un poco la posición pero sin movimiento intencional.")
    else:
        print(f"Realiza la seña dinámica de la letra '{clase}'.")

    print("Presiona ESPACIO para grabar cada ejemplo.")
    print("Presiona N para elegir otra letra.")
    print("Presiona ESC para salir del programa.")

    contador = contar_existentes(clase)

    if contador >= EJEMPLOS_POR_CLASE:
        print(f"La clase '{clase}' ya está completa ({contador}/{EJEMPLOS_POR_CLASE}).")
        return True

    while contador < EJEMPLOS_POR_CLASE:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        texto = f"Clase: {clase} | Ejemplo: {contador}/{EJEMPLOS_POR_CLASE}"
        cv2.putText(frame, texto, (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        instruccion = "Mano quieta (sin seña)" if clase == "Nada" else f"Senia: {clase}"
        cv2.putText(frame, instruccion, (30, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 200, 0), 2)

        cv2.putText(frame, "ESPACIO=grabar | N=cambiar letra | ESC=salir", (30, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        if results.multi_hand_landmarks:
            hand_lms = results.multi_hand_landmarks[0]
            mp_drawing.draw_landmarks(frame, hand_lms, mp_hands.HAND_CONNECTIONS)

        cv2.imshow("Captura de senias dinamicas", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == 27:
            return False

        if key in (ord('n'), ord('N')):
            return True

        if key == 32:
            secuencia = []
            print(f"Grabando '{clase}' ejemplo {contador + 1}...")

            for i in range(FRAMES_POR_SECUENCIA):
                ret, frame = cap.read()
                if not ret:
                    break

                frame = cv2.flip(frame, 1)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(rgb)

                if results.multi_hand_landmarks:
                    hand_lms = results.multi_hand_landmarks[0]

                    handedness_label = "unknown"
                    if results.multi_handedness:
                        handedness_label = results.multi_handedness[0].classification[0].label

                    features = extraer_landmarks(hand_lms, handedness_label)
                    secuencia.append(features.tolist())
                else:
                    secuencia.append([0.0] * 63)

                cv2.putText(frame, f"Grabando {clase}: {i + 1}/{FRAMES_POR_SECUENCIA}",
                            (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

                cv2.imshow("Captura de senias dinamicas", frame)
                cv2.waitKey(1)

            if len(secuencia) == FRAMES_POR_SECUENCIA:
                guardar_secuencia(clase, secuencia, contador)
                contador += 1
                print("Guardado correctamente.")

            time.sleep(0.5)

    print(f"\n✅ Clase '{clase}' completada con {contador} ejemplos.")
    return True


def main():
    cap = cv2.VideoCapture(0)

    with mp_hands.Hands(max_num_hands=1) as hands:
        while True:
            clase = elegir_clase()

            if clase is None:
                continue

            seguir = capturar_clase(cap, hands, clase)

            if not seguir:
                break

    cap.release()
    cv2.destroyAllWindows()
    print("\nSesión terminada.")


if __name__ == "__main__":
    main()
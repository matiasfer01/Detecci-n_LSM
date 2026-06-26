import cv2
import mediapipe as mp
import numpy as np
import os
import time
import json

# ==============================
# CONFIGURACIÓN GENERAL
# ==============================

# Carpeta principal donde se guardará el dataset generado.
DATASET_DIR = "dataset_dinamicas"

# Lista de clases que se pueden capturar.
# "Nada" representa una mano quieta sin realizar ninguna seña.
CLASES = ["J", "Ñ", "Z", "X", "K", "Q", "Nada"]

# Cantidad de frames que tendrá cada secuencia grabada.
# Cada ejemplo de una clase se compone de 60 frames.
FRAMES_POR_SECUENCIA = 60

# Número total de ejemplos que se desea capturar por cada clase.
EJEMPLOS_POR_CLASE = 600

# Inicialización de módulos de MediaPipe para detección y dibujo de manos.
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils


def extraer_landmarks(hand_lms, handedness_label):
    """
    Extrae y normaliza los landmarks de una mano detectada.

    Parámetros:
        hand_lms:
            Landmarks de la mano detectada por MediaPipe.
        handedness_label:
            Indica si la mano detectada es izquierda o derecha.

    Proceso:
        1. Convierte los landmarks de MediaPipe en un arreglo NumPy.
        2. Usa la muñeca como punto de referencia inicial.
        3. Normaliza la escala usando el landmark 9.
        4. Si la mano detectada es derecha, invierte el eje X para unificar la orientación.
        5. Devuelve todos los puntos en un arreglo plano.

    Retorna:
        Un vector de 63 valores:
        21 landmarks * 3 coordenadas (x, y, z).
    """

    # Convierte los 21 landmarks de la mano en un arreglo con coordenadas x, y, z.
    pts = np.array(
        [[lm.x, lm.y, lm.z] for lm in hand_lms.landmark],
        dtype=np.float32
    )

    # Centra todos los puntos tomando como origen el landmark 0, que corresponde a la muñeca.
    pts = pts - pts[0]

    # Calcula una escala usando la distancia del landmark 9.
    # Se suma 1e-6 para evitar división entre cero.
    scale = np.linalg.norm(pts[9]) + 1e-6

    # Normaliza los puntos para que las manos tengan una escala comparable.
    pts = pts / scale

    # Si la mano detectada es derecha, se invierte el eje X.
    # Esto permite tratar manos izquierda y derecha de forma más uniforme.
    if handedness_label.lower() == "right":
        pts[:, 0] *= -1

    # Convierte la matriz de landmarks en un vector plano.
    return pts.flatten()


def guardar_secuencia(clase, secuencia, contador):
    """
    Guarda una secuencia capturada en formato JSON.

    Parámetros:
        clase:
            Clase o letra que se está grabando.
        secuencia:
            Lista con los landmarks de todos los frames capturados.
        contador:
            Número consecutivo del ejemplo dentro de la clase.

    Resultado:
        Crea una carpeta por clase y guarda el archivo JSON correspondiente.
    """

    # Construye la ruta de la carpeta donde se guardarán los ejemplos de la clase.
    carpeta = os.path.join(DATASET_DIR, clase)

    # Crea la carpeta si no existe.
    os.makedirs(carpeta, exist_ok=True)

    # Define el nombre del archivo JSON.
    ruta = os.path.join(carpeta, f"{clase}_{contador}.json")

    # Guarda la secuencia en un archivo JSON.
    with open(ruta, "w", encoding="utf-8") as f:
        json.dump(secuencia, f)


def contar_existentes(clase):
    """
    Cuenta cuántos ejemplos JSON existen para una clase determinada.

    Parámetros:
        clase:
            Clase o letra que se desea revisar.

    Retorna:
        Número de archivos JSON existentes para esa clase.
    """

    # Ruta de la carpeta correspondiente a la clase.
    carpeta = os.path.join(DATASET_DIR, clase)

    # Si la carpeta aún no existe, significa que no hay ejemplos guardados.
    if not os.path.exists(carpeta):
        return 0

    # Lista únicamente los archivos JSON que correspondan a la clase indicada.
    archivos = [
        f for f in os.listdir(carpeta)
        if f.endswith(".json") and f.startswith(f"{clase}_")
    ]

    # Devuelve la cantidad de archivos encontrados.
    return len(archivos)


def elegir_clase():
    """
    Muestra las clases disponibles y permite seleccionar una clase para grabar.

    El usuario puede escribir:
        - El número de la opción.
        - El nombre de la letra o clase.

    Retorna:
        La clase seleccionada si es válida.
        None si la entrada no corresponde a ninguna clase.
    """

    print("\n=== Clases disponibles ===")

    # Muestra cada clase junto con la cantidad de ejemplos ya capturados.
    for i, c in enumerate(CLASES):
        existentes = contar_existentes(c)
        estado = f"{existentes}/{EJEMPLOS_POR_CLASE}"
        completa = " ✅" if existentes >= EJEMPLOS_POR_CLASE else ""
        print(f"  {i + 1}. {c} — {estado}{completa}")

    # Solicita al usuario seleccionar una clase.
    print("\nEscribe el número o el nombre de la letra que quieres grabar: ", end="")
    entrada = input().strip()

    # Si el usuario escribe un número, se intenta usar como índice de la lista de clases.
    if entrada.isdigit():
        idx = int(entrada) - 1
        if 0 <= idx < len(CLASES):
            return CLASES[idx]

    # Si el usuario escribe una letra o nombre, se compara con las clases disponibles.
    for c in CLASES:
        if c.upper() == entrada.upper():
            return c

    # Mensaje de error cuando la opción no es válida.
    print(f"'{entrada}' no es una opción válida.")
    return None


def capturar_clase(cap, hands, clase):
    """
    Captura ejemplos de una clase usando la cámara y MediaPipe Hands.

    Parámetros:
        cap:
            Objeto de captura de video de OpenCV.
        hands:
            Objeto de MediaPipe Hands para detectar landmarks.
        clase:
            Clase que se desea grabar.

    Controles:
        ESPACIO:
            Graba un ejemplo de la clase actual.
        N:
            Regresa al menú para seleccionar otra clase.
        ESC:
            Sale del programa.

    Retorna:
        True si se debe continuar el programa.
        False si el usuario decide salir.
    """

    print(f"\n=== Capturando clase: {clase} ===")

    # Instrucciones específicas según la clase seleccionada.
    if clase == "Nada":
        print("Mantén la mano quieta frente a la cámara SIN hacer ninguna seña.")
        print("Puedes variar un poco la posición pero sin movimiento intencional.")
    else:
        print(f"Realiza la seña dinámica de la letra '{clase}'.")

    print("Presiona ESPACIO para grabar cada ejemplo.")
    print("Presiona N para elegir otra letra.")
    print("Presiona ESC para salir del programa.")

    # Cuenta cuántos ejemplos ya existen para continuar desde ese número.
    contador = contar_existentes(clase)

    # Si la clase ya tiene todos los ejemplos requeridos, no se captura más.
    if contador >= EJEMPLOS_POR_CLASE:
        print(f"La clase '{clase}' ya está completa ({contador}/{EJEMPLOS_POR_CLASE}).")
        return True

    # Ciclo principal de captura mientras la clase no esté completa.
    while contador < EJEMPLOS_POR_CLASE:
        ret, frame = cap.read()

        # Si no se puede leer la cámara, se rompe el ciclo.
        if not ret:
            break

        # Voltea el frame horizontalmente para una vista tipo espejo.
        frame = cv2.flip(frame, 1)

        # Convierte el frame de BGR a RGB porque MediaPipe trabaja con RGB.
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Procesa el frame para detectar manos.
        results = hands.process(rgb)

        # Texto informativo sobre la clase y el avance.
        texto = f"Clase: {clase} | Ejemplo: {contador}/{EJEMPLOS_POR_CLASE}"
        cv2.putText(frame, texto, (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # Muestra la instrucción correspondiente en pantalla.
        instruccion = "Mano quieta (sin seña)" if clase == "Nada" else f"Senia: {clase}"
        cv2.putText(frame, instruccion, (30, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 200, 0), 2)

        # Muestra los controles disponibles.
        cv2.putText(frame, "ESPACIO=grabar | N=cambiar letra | ESC=salir", (30, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # Si se detecta una mano, dibuja los landmarks en pantalla.
        if results.multi_hand_landmarks:
            hand_lms = results.multi_hand_landmarks[0]
            mp_drawing.draw_landmarks(frame, hand_lms, mp_hands.HAND_CONNECTIONS)

        # Muestra la ventana de captura.
        cv2.imshow("Captura de senias dinamicas", frame)

        # Lee la tecla presionada.
        key = cv2.waitKey(1) & 0xFF

        # ESC: termina el programa.
        if key == 27:
            return False

        # N: permite cambiar de clase.
        if key in (ord('n'), ord('N')):
            return True

        # ESPACIO: inicia la grabación de una secuencia.
        if key == 32:
            secuencia = []
            print(f"Grabando '{clase}' ejemplo {contador + 1}...")

            # Captura los frames que formarán una secuencia completa.
            for i in range(FRAMES_POR_SECUENCIA):
                ret, frame = cap.read()

                # Si falla la lectura de la cámara, se detiene la captura de la secuencia.
                if not ret:
                    break

                # Voltea el frame para mantener la vista tipo espejo.
                frame = cv2.flip(frame, 1)

                # Convierte el frame a RGB para MediaPipe.
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Procesa la imagen para detectar landmarks de la mano.
                results = hands.process(rgb)

                # Si se detecta una mano, se extraen sus landmarks normalizados.
                if results.multi_hand_landmarks:
                    hand_lms = results.multi_hand_landmarks[0]

                    # Valor por defecto si MediaPipe no determina la lateralidad de la mano.
                    handedness_label = "unknown"

                    # Si existe información de lateralidad, se obtiene si es mano izquierda o derecha.
                    if results.multi_handedness:
                        handedness_label = results.multi_handedness[0].classification[0].label

                    # Extrae las características normalizadas de la mano.
                    features = extraer_landmarks(hand_lms, handedness_label)

                    # Agrega los landmarks del frame actual a la secuencia.
                    secuencia.append(features.tolist())
                else:
                    # Si no se detecta mano, se guarda un vector de ceros.
                    # Esto mantiene la secuencia con el mismo tamaño.
                    secuencia.append([0.0] * 63)

                # Muestra el progreso de grabación en pantalla.
                cv2.putText(frame, f"Grabando {clase}: {i + 1}/{FRAMES_POR_SECUENCIA}",
                            (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

                # Muestra el frame actual durante la grabación.
                cv2.imshow("Captura de senias dinamicas", frame)
                cv2.waitKey(1)

            # Solo guarda la secuencia si tiene exactamente la cantidad esperada de frames.
            if len(secuencia) == FRAMES_POR_SECUENCIA:
                guardar_secuencia(clase, secuencia, contador)
                contador += 1
                print("Guardado correctamente.")

            # Pausa breve antes de permitir capturar otro ejemplo.
            time.sleep(0.5)

    print(f"\n✅ Clase '{clase}' completada con {contador} ejemplos.")
    return True


def main():
    """
    Función principal del programa.

    Proceso general:
        1. Abre la cámara.
        2. Inicializa MediaPipe Hands.
        3. Permite elegir una clase.
        4. Captura ejemplos de esa clase.
        5. Libera la cámara y cierra las ventanas al finalizar.
    """

    # Abre la cámara principal del equipo.
    cap = cv2.VideoCapture(0)

    # Inicializa MediaPipe Hands permitiendo detectar una sola mano.
    with mp_hands.Hands(max_num_hands=1) as hands:
        while True:
            # Permite al usuario elegir qué clase desea capturar.
            clase = elegir_clase()

            # Si la clase no es válida, vuelve a pedir una opción.
            if clase is None:
                continue

            # Captura ejemplos de la clase seleccionada.
            seguir = capturar_clase(cap, hands, clase)

            # Si el usuario presiona ESC, se termina el programa.
            if not seguir:
                break

    # Libera la cámara.
    cap.release()

    # Cierra todas las ventanas creadas por OpenCV.
    cv2.destroyAllWindows()

    print("\nSesión terminada.")


# Punto de entrada del programa.
# Solo ejecuta main() si este archivo se ejecuta directamente.
if __name__ == "__main__":
    main()
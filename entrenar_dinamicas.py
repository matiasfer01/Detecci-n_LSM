import os
import json
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from collections import Counter


DATASET_DIR = "dataset_dinamicas"
MODEL_PATH = "dynamic_signs_model.joblib"

FRAMES_POR_SECUENCIA = 60
LANDMARKS_POR_FRAME = 63


def cargar_dataset():
    X = []
    y = []

    if not os.path.exists(DATASET_DIR):
        print(f"No existe la carpeta: {DATASET_DIR}")
        return np.array(X), np.array(y)

    for clase in os.listdir(DATASET_DIR):
        carpeta = os.path.join(DATASET_DIR, clase)

        if not os.path.isdir(carpeta):
            continue

        for archivo in os.listdir(carpeta):
            if archivo.endswith(".json"):
                ruta = os.path.join(carpeta, archivo)

                try:
                    with open(ruta, "r", encoding="utf-8") as f:
                        secuencia = json.load(f)

                    secuencia = np.array(secuencia, dtype=np.float32)

                    if secuencia.shape != (FRAMES_POR_SECUENCIA, LANDMARKS_POR_FRAME):
                        print(f"Archivo ignorado por forma incorrecta: {ruta}")
                        print(f"Forma encontrada: {secuencia.shape}")
                        continue

                    # Aplanar: 60 frames * 63 landmarks = 3780 datos
                    X.append(secuencia.flatten())
                    y.append(clase)

                except Exception as e:
                    print(f"Error leyendo {ruta}: {e}")

    return np.array(X), np.array(y)


def main():
    X, y = cargar_dataset()

    print("\nTotal de ejemplos válidos:", len(X))
    print("Clases encontradas:", sorted(set(y)))

    conteo = Counter(y)

    print("\nEjemplos por clase:")
    for clase, cantidad in sorted(conteo.items()):
        print(f"  {clase}: {cantidad}")

    if len(X) == 0:
        print("\nNo hay datos para entrenar.")
        return

    if len(set(y)) < 2:
        print("\nNecesitas al menos 2 clases para entrenar.")
        return

    clase_menor = min(conteo.values())

    if clase_menor < 2:
        print("\nError: alguna clase tiene menos de 2 ejemplos.")
        print("Graba más ejemplos de esa clase antes de entrenar.")
        return

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        class_weight="balanced",
        max_depth=None
    )

    print("\nEntrenando modelo...")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print("\nAccuracy:", accuracy_score(y_test, y_pred))
    print("\nReporte:")
    print(classification_report(y_test, y_pred))

    joblib.dump(model, MODEL_PATH)
    print(f"\nModelo guardado como: {MODEL_PATH}")


if __name__ == "__main__":
    main()
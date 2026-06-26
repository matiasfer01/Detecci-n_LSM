import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# ==============================
# CONFIGURACIÓN GENERAL
# ==============================

# Archivo CSV donde están guardados los landmarks extraídos de las señas estáticas.
CSV_FILE = "vowels_landmarks.csv"

# Nombre del archivo donde se guardará el modelo entrenado.
MODEL_FILE = "vowels_model.joblib"


def main():
    """
    Función principal del programa.

    Proceso general:
        1. Carga el archivo CSV con los landmarks de las señas.
        2. Separa las etiquetas de las características.
        3. Divide los datos en entrenamiento y prueba.
        4. Crea y entrena un modelo Random Forest.
        5. Evalúa el modelo usando un reporte de clasificación.
        6. Muestra la matriz de confusión.
        7. Guarda el modelo entrenado en un archivo .joblib.
    """

    # Lee el archivo CSV que contiene los datos de entrenamiento.
    df = pd.read_csv(CSV_FILE)

    # Extrae la columna "label", que contiene la clase o seña correspondiente.
    # Esta variable representa las salidas esperadas del modelo.
    y = df["label"].values

    # Elimina la columna "label" para quedarse únicamente con las características.
    # Estas características corresponden a los landmarks de la mano.
    X = df.drop(columns=["label"]).values.astype(np.float32)

    # Divide el dataset en entrenamiento y prueba.
    # test_size=0.2 indica que el 20% de los datos será usado para prueba.
    # random_state=42 permite obtener la misma división cada vez que se ejecute.
    # stratify=y mantiene la proporción de clases en entrenamiento y prueba.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Crea el modelo Random Forest para clasificación.
    model = RandomForestClassifier(
        # Cantidad de árboles que tendrá el bosque.
        n_estimators=300,

        # Semilla para que el entrenamiento sea reproducible.
        random_state=42,

        # Ajusta el peso de las clases para ayudar cuando hay desbalance entre ellas.
        class_weight="balanced"
    )

    # Entrena el modelo usando los datos de entrenamiento.
    model.fit(X_train, y_train)

    # Realiza predicciones usando los datos de prueba.
    preds = model.predict(X_test)

    # Muestra el reporte de clasificación.
    # Incluye precision, recall, f1-score y support para cada clase.
    print("\n=== Reporte ===")
    print(classification_report(y_test, preds))

    # Muestra la matriz de confusión.
    # Sirve para observar qué clases fueron clasificadas correctamente
    # y cuáles se confundieron con otras.
    print("=== Matriz de confusión ===")
    print(confusion_matrix(y_test, preds))

    # Guarda el modelo entrenado en un archivo .joblib.
    joblib.dump(model, MODEL_FILE)

    # Mensaje de confirmación indicando dónde se guardó el modelo.
    print(f"\n✅ Modelo guardado en: {MODEL_FILE}")


# Punto de entrada del programa.
# Ejecuta main() solo si este archivo se ejecuta directamente.
if __name__ == "__main__":
    main()
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib

CSV_FILE = "vowels_landmarks.csv"
MODEL_FILE = "vowels_model.joblib"

def main():
    df = pd.read_csv(CSV_FILE)

    y = df["label"].values
    X = df.drop(columns=["label"]).values.astype(np.float32)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        class_weight="balanced"
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    print("\n=== Reporte ===")
    print(classification_report(y_test, preds))
    print("=== Matriz de confusión ===")
    print(confusion_matrix(y_test, preds))

    joblib.dump(model, MODEL_FILE)
    print(f"\n✅ Modelo guardado en: {MODEL_FILE}")

if __name__ == "__main__":
    main()

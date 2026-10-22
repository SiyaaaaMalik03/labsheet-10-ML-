
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

BASE = Path(__file__).resolve().parent
DATA_FILE = BASE / "disease_data.csv"
MODEL_FILE = BASE / "disease_model.joblib"

SYMPTOMS = [
    "Fever",
    "Cough",
    "Headache",
    "Fatigue",
    "Sore_Throat",
    "Body_Pain"
]

def create_dataset():
    # Synthetic data for academic demonstration only.
    rng = np.random.default_rng(42)
    patterns = {
        "Flu_Like_Illness": [1, 1, 1, 1, 0, 1],
        "Common_Cold_Like": [0, 1, 0, 0, 1, 0],
        "Migraine_Like": [0, 0, 1, 0, 0, 0],
        "Allergy_Like": [0, 1, 0, 0, 1, 0],
        "General_Fatigue": [0, 0, 0, 1, 0, 0]
    }

    rows = []
    for disease, pattern in patterns.items():
        for _ in range(100):
            symptoms = [
                int(1 - value if rng.random() < 0.08 else value)
                for value in pattern
            ]
            rows.append(dict(zip(SYMPTOMS, symptoms)) | {
                "Disease": disease
            })

    df = pd.DataFrame(rows)
    df.to_csv(DATA_FILE, index=False)
    return df

def main():
    if DATA_FILE.exists():
        df = pd.read_csv(DATA_FILE)
    else:
        print("Creating illustrative synthetic symptom dataset.")
        df = create_dataset()

    required = SYMPTOMS + ["Disease"]
    missing = [col for col in required if col not in df.columns]

    if missing:
        raise ValueError(f"Dataset is missing columns: {missing}")

    df = df[required].copy()
    for col in SYMPTOMS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["Disease"] = df["Disease"].fillna("Unknown")
    df = df.dropna(subset=["Disease"])
    df = df.drop_duplicates()

    X = df[SYMPTOMS]
    y = df["Disease"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(max_depth=8, random_state=42),
        "Random Forest": RandomForestClassifier(
            n_estimators=150, random_state=42
        )
    }

    best_model = None
    best_accuracy = -1

    print("\nDISEASE CLASSIFICATION MODEL EVALUATION")
    print("Dataset shape:", df.shape)
    print("\nMissing values:")
    print(df.isnull().sum())

    for name, classifier in models.items():
        pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("classifier", classifier)
        ])

        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)

        print(f"\n{name}: {accuracy:.2%} accuracy")
        print(classification_report(
            y_test, predictions, zero_division=0
        ))

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = pipeline

    joblib.dump(best_model, MODEL_FILE)
    print(f"\nBest model saved to: {MODEL_FILE}")
    print(f"Best test accuracy: {best_accuracy:.2%}")

    print("\nENTER SYMPTOMS")
    print("Enter 1 for Yes and 0 for No.")

    answers = {}
    try:
        for symptom in SYMPTOMS:
            value = int(input(f"{symptom.replace('_', ' ')}? (1/0): "))
            if value not in (0, 1):
                raise ValueError("Only 0 or 1 is allowed.")
            answers[symptom] = value

        patient = pd.DataFrame([answers], columns=SYMPTOMS)
        prediction = best_model.predict(patient)[0]

        print("\nModel's illustrative classification:", prediction)
        print(
            "This output is not a medical diagnosis. "
            "Consult a qualified healthcare professional for health concerns."
        )

    except ValueError as error:
        print("Invalid input:", error)

if __name__ == "__main__":
    main()

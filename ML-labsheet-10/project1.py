
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

BASE = Path(__file__).resolve().parent
DATA_FILE = BASE / "loan_data.csv"
MODEL_FILE = BASE / "loan_model.joblib"

def create_dataset():
    rng = np.random.default_rng(42)
    n = 500

    df = pd.DataFrame({
        "Income": rng.integers(15000, 150001, n),
        "Education": rng.choice(["Graduate", "Not Graduate"], n),
        "Self_Employed": rng.choice(["Yes", "No"], n),
        "Married": rng.choice(["Yes", "No"], n),
        "LoanAmount": rng.integers(20, 501, n),
        "Credit_History": rng.choice([0, 1], n, p=[0.25, 0.75]),
        "Dependents": rng.integers(0, 5, n)
    })

    score = (
        (df["Income"] > 50000).astype(int)
        + df["Credit_History"]
        + (df["LoanAmount"] < 250).astype(int)
        + (df["Education"] == "Graduate").astype(int)
    )
    df["Loan_Status"] = np.where(score >= 2, "Approved", "Rejected")

    df.to_csv(DATA_FILE, index=False)
    return df

def main():
    if DATA_FILE.exists():
        df = pd.read_csv(DATA_FILE)
    else:
        print("Creating illustrative sample loan dataset.")
        df = create_dataset()

    target = "Loan_Status"
    X = df.drop(columns=[target])
    y = df[target]

    numeric = [
        "Income", "LoanAmount", "Credit_History", "Dependents"
    ]
    categorical = ["Education", "Self_Employed", "Married"]

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessing = ColumnTransformer([
        ("numeric", numeric_pipeline, numeric),
        ("categorical", categorical_pipeline, categorical)
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(max_depth=6, random_state=42),
        "Random Forest": RandomForestClassifier(
            n_estimators=150, random_state=42
        )
    }

    best_model = None
    best_accuracy = -1

    print("\nLOAN APPROVAL MODEL EVALUATION")
    for name, classifier in models.items():
        pipeline = Pipeline([
            ("preprocessing", preprocessing),
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

    print("\nENTER APPLICANT DETAILS")
    try:
        applicant = pd.DataFrame([{
            "Income": float(input("Monthly income (₹): ")),
            "Education": input("Education (Graduate/Not Graduate): ").strip(),
            "Self_Employed": input("Self employed? (Yes/No): ").strip(),
            "Married": input("Married? (Yes/No): ").strip(),
            "LoanAmount": float(input("Loan amount (dataset units): ")),
            "Credit_History": int(input("Credit history (0 or 1): ")),
            "Dependents": int(input("Number of dependents: "))
        }])

        prediction = best_model.predict(applicant)[0]
        print("\nModel prediction:", prediction)
        print("This is an educational demo, not a real lending decision.")

    except ValueError:
        print("Invalid input. Enter numeric values in numeric fields.")

if __name__ == "__main__":
    main()

"""
ML Model: Random Forest Classifier on Iris Dataset
Trains the model, evaluates accuracy, and saves results.
"""

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import json
import os


def load_data():
    """Load the Iris dataset and split into train/test sets."""
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        iris.data, iris.target, test_size=0.2, random_state=42
    )
    return X_train, X_test, y_train, y_test, iris.target_names


def train_model(X_train, y_train):
    """Train a Random Forest classifier."""
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    """Make predictions and compute accuracy."""
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    return predictions, accuracy


def save_results(accuracy, model_name="Random Forest", dataset_name="Iris"):
    """Save model results to a JSON file for the deployment page."""
    results = {
        "model_name": model_name,
        "dataset": dataset_name,
        "accuracy": round(accuracy * 100, 2),
    }
    with open("results.json", "w") as f:
        json.dump(results, f, indent=2)
    return results


def main():
    print("=" * 50)
    print("  ML CI/CD Pipeline - Model Training")
    print("=" * 50)

    # Load data
    X_train, X_test, y_train, y_test, target_names = load_data()
    print(f"\n[DATA] Dataset: Iris")
    print(f"   Training samples: {len(X_train)}")
    print(f"   Testing samples:  {len(X_test)}")
    print(f"   Classes: {list(target_names)}")

    # Train
    print("\n[TRAIN] Training Random Forest Classifier...")
    model = train_model(X_train, y_train)
    print("   Model trained successfully!")

    # Evaluate
    predictions, accuracy = evaluate_model(model, X_test, y_test)
    print(f"\n[RESULT] Model Accuracy: {accuracy * 100:.2f}%")

    # Save results
    results = save_results(accuracy)
    print(f"\n[SAVE] Results saved to results.json")
    print(f"   {results}")

    print("\n" + "=" * 50)
    return accuracy


if __name__ == "__main__":
    main()

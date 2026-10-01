"""
Automated Test Cases for the ML Model using pytest.
"""

import pytest
import os
import json
from model import load_data, train_model, evaluate_model, save_results


class TestDataLoading:
    """Tests for data loading functionality."""

    def test_load_data_returns_correct_splits(self):
        X_train, X_test, y_train, y_test, target_names = load_data()
        assert len(X_train) > 0, "Training set should not be empty"
        assert len(X_test) > 0, "Test set should not be empty"

    def test_load_data_correct_features(self):
        X_train, X_test, y_train, y_test, _ = load_data()
        assert X_train.shape[1] == 4, "Iris dataset should have 4 features"
        assert X_test.shape[1] == 4, "Test set should have 4 features"

    def test_load_data_correct_split_ratio(self):
        X_train, X_test, y_train, y_test, _ = load_data()
        total = len(X_train) + len(X_test)
        assert total == 150, "Iris dataset should have 150 samples total"

    def test_target_names_present(self):
        _, _, _, _, target_names = load_data()
        assert len(target_names) == 3, "Iris has 3 target classes"


class TestModelTraining:
    """Tests for model training functionality."""

    def test_model_trains_successfully(self):
        X_train, _, y_train, _, _ = load_data()
        model = train_model(X_train, y_train)
        assert model is not None, "Model should not be None after training"

    def test_model_can_predict(self):
        X_train, X_test, y_train, _, _ = load_data()
        model = train_model(X_train, y_train)
        predictions = model.predict(X_test)
        assert len(predictions) > 0, "Model should produce predictions"


class TestModelEvaluation:
    """Tests for model evaluation functionality."""

    def test_accuracy_above_threshold(self):
        X_train, X_test, y_train, y_test, _ = load_data()
        model = train_model(X_train, y_train)
        _, accuracy = evaluate_model(model, X_test, y_test)
        assert accuracy >= 0.85, f"Accuracy {accuracy:.2f} is below 85% threshold"

    def test_predictions_shape_matches_test_set(self):
        X_train, X_test, y_train, y_test, _ = load_data()
        model = train_model(X_train, y_train)
        predictions, _ = evaluate_model(model, X_test, y_test)
        assert len(predictions) == len(y_test), "Predictions count should match test set"

    def test_accuracy_is_valid_range(self):
        X_train, X_test, y_train, y_test, _ = load_data()
        model = train_model(X_train, y_train)
        _, accuracy = evaluate_model(model, X_test, y_test)
        assert 0.0 <= accuracy <= 1.0, "Accuracy should be between 0 and 1"


class TestResultsSaving:
    """Tests for saving results to JSON."""

    def test_save_results_creates_file(self):
        save_results(0.95)
        assert os.path.exists("results.json"), "results.json should be created"

    def test_save_results_correct_content(self):
        save_results(0.9667, "Random Forest", "Iris")
        with open("results.json", "r") as f:
            data = json.load(f)
        assert data["model_name"] == "Random Forest"
        assert data["dataset"] == "Iris"
        assert data["accuracy"] == 96.67

    def test_save_results_accuracy_rounded(self):
        save_results(0.93333333)
        with open("results.json", "r") as f:
            data = json.load(f)
        assert data["accuracy"] == 93.33

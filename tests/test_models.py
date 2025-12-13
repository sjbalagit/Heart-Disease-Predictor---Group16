# VSCode Copilot and ChatGPT was used to assit in writing this test file.
import pytest
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from sklearn.dummy import DummyClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from utils.models import get_models

def test_get_models():
    """
    Test the get_models function to ensure it returns the correct models.
    """
    models = get_models(random_state=42)

    assert isinstance(models, dict)

    expected_keys = {"Dummy Classifier", "Decision Tree", "Logistic Regression", "SVM RBF"}
    assert set(models.keys()) == expected_keys

    assert isinstance(models["Dummy Classifier"], DummyClassifier)
    assert isinstance(models["Decision Tree"], DecisionTreeClassifier)
    assert isinstance(models["Logistic Regression"], LogisticRegression)
    assert isinstance(models["SVM RBF"], SVC)

    assert models["Decision Tree"].random_state == 42
    assert models["Logistic Regression"].random_state == 42


def test_get_models_different_random_state():
    """
    Test get_models with a different random_state value.
    """
    models = get_models(random_state=999)

    assert models["Decision Tree"].random_state == 999
    assert models["Logistic Regression"].random_state == 999


def test_get_models_no_random_state():
    """
    Test get_models with random_state=None.
    """
    models = get_models(random_state=None)

    assert models["Decision Tree"].random_state is None
    assert models["Logistic Regression"].random_state is None


def test_get_models_default_random_state():
    """
    Test get_models with default random_state value.
    """
    models = get_models()

    assert models["Decision Tree"].random_state == 123
    assert models["Logistic Regression"].random_state == 123


def test_get_models_return_type():
    """
    Test that get_models returns a dictionary.
    """
    models = get_models(random_state=0)
    assert isinstance(models, dict)

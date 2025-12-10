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

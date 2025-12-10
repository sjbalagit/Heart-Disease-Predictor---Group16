from sklearn.dummy import DummyClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

def get_models(random_state=123):
    """
    Returns a dictionary of classification models with specified random_state where applicable.
    
    Parameters
    ----------
    random_state : int, optional
        Random state for reproducibility, by default 123
    Returns
    ----------
    dict
        Dictionary with model names as keys and model instances as values
    """
    return {
        "Dummy Classifier": DummyClassifier(strategy='most_frequent'),
        "Decision Tree": DecisionTreeClassifier(random_state=random_state),
        "Logistic Regression": LogisticRegression(random_state=random_state),
        "SVM RBF": SVC(random_state=random_state)
    }
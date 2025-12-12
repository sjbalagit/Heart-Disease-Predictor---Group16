from sklearn.dummy import DummyClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import numpy as np

def get_models(random_state=123):
    
    return {
        "Dummy Classifier": DummyClassifier(strategy='most_frequent'),
        "Decision Tree": DecisionTreeClassifier(random_state=random_state),
        "Logistic Regression": LogisticRegression(random_state=random_state),
        "SVM RBF": SVC(random_state=random_state)
    }
def get_param_dist():

    return {
    "Decision Tree": {'decisiontreeclassifier__max_depth': np.arange(1, 11)},
    "Logistic Regression": {"logisticregression__C" : 10.0 ** np.arange(-3, 2, 1), "logisticregression__max_iter" : [80, 100, 500, 1000, 1500, 2000]},
    "SVM RBF": {"svc__C": 10.0 ** np.arange(-3, 2, 1), "svc__gamma": 10.0 ** np.arange(-3, 2, 1)}
    }
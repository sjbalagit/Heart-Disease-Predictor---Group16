# VSCode Copilot and ChatGPT was used to assit in writing this test file.
import pandas as pd
import pytest
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import make_scorer, fbeta_score
from sklearn.utils._param_validation import InvalidParameterError

from utils.mean_std_cv_scores import mean_std_cross_val_scores

def test_mean_std_cv_scores():
    """
    Test mean_std_cross_val_scores with fbeta_score as scoring metric.
    """
    X = pd.DataFrame({
        "age": [20, 30, 40, 50, 60],
        "chol": [180, 200, 190, 210, 220]
    })
    y = pd.Series(["No Heart Disease", "Heart Disease", "No Heart Disease", "Heart Disease", "No Heart Disease"])

    scoring = make_scorer(fbeta_score, beta=2, pos_label="Heart Disease")

    model = make_pipeline(StandardScaler(), LogisticRegression())

    scores = mean_std_cross_val_scores(
        model=model, 
        X_train=X,
        y_train=y,
        scoring=scoring,
        cv=3,
        return_train_score=True
    )


    assert isinstance(scores, pd.Series)
    assert isinstance(scores["train_score"], str)
    assert "train_score" in scores.index
    assert "test_score" in scores.index
    assert "(" in scores["train_score"]
    assert "+/-" in scores["train_score"]

def test_mean_std_cv_scores_no_train_score():
    """
    Test mean_std_cross_val_scores with return_train_score=False
    """
    X = pd.DataFrame({
        "age": [20, 30, 40, 50, 60],
        "chol": [180, 200, 190, 210, 220]
    })
    y = pd.Series(["No Heart Disease", "Heart Disease", "No Heart Disease", "Heart Disease", "No Heart Disease"])

    scoring = make_scorer(fbeta_score, beta=2, pos_label="Heart Disease")
    
    model = make_pipeline(StandardScaler(), LogisticRegression())

    scores = mean_std_cross_val_scores(
        model=model,
        X_train=X,
        y_train=y,
        scoring=scoring,
        cv=2,
        return_train_score=False
    )

    assert "train_score" not in scores.index
    assert "test_score" in scores.index


def test_mean_std_cv_scores_invalid_cv():
    """
    Test mean_std_cross_val_scores with invalid cv value.
    """
    X = pd.DataFrame({"age":[20,30]})
    y = pd.Series(["No Heart Disease", "Heart Disease"])

    scoring = make_scorer(fbeta_score, beta=2, pos_label="Heart Disease")
    model = LogisticRegression()

    with pytest.raises(ValueError):
        mean_std_cross_val_scores(model, X, y, scoring, cv=5)


def test_mean_std_cv_scores_invalid_model():
    """
    Test mean_std_cross_val_scores with invalid model type.
    """
    X = pd.DataFrame({"age":[20,30,40]})
    y = pd.Series(["No Heart Disease","Heart Disease","No Heart Disease"])

    scoring = make_scorer(fbeta_score, beta=2, pos_label="Heart Disease")
    
    with pytest.raises(InvalidParameterError):
        mean_std_cross_val_scores("not_a_model", X, y, scoring, cv=2)


def test_mean_std_cv_scores_mismatched_lengths():
    """
    Test mean_std_cross_val_scores with mismatched lengths of X and y.
    """
    X = pd.DataFrame({"age":[20,30,40]})
    y = pd.Series(["No Heart Disease", "Heart Disease"])

    scoring = make_scorer(fbeta_score, beta=2, pos_label="Heart Disease")
    model = LogisticRegression()

    with pytest.raises(ValueError):
        mean_std_cross_val_scores(model, X, y, scoring)
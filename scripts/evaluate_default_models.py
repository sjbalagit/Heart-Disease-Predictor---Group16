import click
import numpy as np
import pandas as pd
from pathlib import Path
import pickle

from sklearn.metrics import make_scorer, fbeta_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import make_column_transformer

from sklearn.dummy import DummyClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

# this one goes to utils.py
def mean_std_cross_val_scores(model, X_train, y_train, **kwargs):
    """
    Returns mean and std of cross validation

    Parameters
    ----------
    model :
        scikit-learn model
    X_train : numpy array or pandas DataFrame
        X in the training data
    y_train :
        y in the training data

    Returns
    ----------
        pandas Series with mean scores from cross_validation
    """
    scores = cross_validate(model, X_train, y_train, **kwargs)
    mean_scores = pd.DataFrame(scores).mean()
    std_scores = pd.DataFrame(scores).std()
    out_col = []
    for i in range(len(mean_scores)):
        out_col.append((f"%0.3f (+/- %0.3f)" % (mean_scores.iloc[i], std_scores.iloc[i])))
    return pd.Series(data=out_col, index=mean_scores.index)
    
@click.command()
@click.option('--X_train', required=True, help='Path to X_train data CSV')
@click.option('--y_train', required=True, help='Path to y_train data CSV')
@click.option('--preprocessor', required=True, help='Path to preprocessor')
@click.option('--beta', default=2.0, help='Beta parameter for fbeta_score')
@click.option('--random-state', default=123, help='Random state for classifiers')
@click.option('--results', required=True, help='Path to save results table')

def main(X_train, y_train, preprocessor, beta, random_state, results):
    Path(results).mkdir(parents=True, exist_ok=True)

    X_train = pd.read_csv(X_train)
    y_train = pd.read_csv(y_train).squeeze("columns")

    with open(preprocessor, "rb") as f:
        preprocessor = pickle.load(f)

    models = {
        "dummy clf": DummyClassifier(strategy='most_frequent'),
        "decision tree": DecisionTreeClassifier(random_state=random_state),
        "Logistic Regression": LogisticRegression(random_state=random_state),
        "RBF SVM": SVC(random_state=random_state)
    }

    scorer = make_scorer(fbeta_score, pos_label='Heart Disease', beta=beta)

    results_dict = {}
    for name, model in models.items():
        pipe = make_pipeline(preprocessor, model)
        results_dict[name] = mean_std_cross_val_scores(
            pipe, X_train, y_train, cv=5, return_train_score=True, scoring=scorer
        )
    results_df = pd.DataFrame(results_dict).T
    
    results_df.to_csv(Path(results) / "CV_scores_default_parameters.csv", index=True)


if __name__ == "__main__":
    main()
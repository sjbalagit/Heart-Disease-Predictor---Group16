# hyperparameter_tuning.py
# author: Mantram Sharma
# date: 2025-12-06

import click
import os
import numpy as np
import pandas as pd
import pickle
from sklearn import set_config
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import fbeta_score, make_scorer
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from joblib import dump

@click.command()
@click.option('--X_train_path', required=True, help='Path to X_train data CSV')
@click.option('--y_train_path', required=True, help='Path to y_train data CSV')
@click.option('--pos_label', default='Heart Disease', help='Positive class label for fbeta_score')
@click.option('--results_to', type=str, help="Path to directory where the final model will be written to")
@click.option('--beta', default=2.0, help='Beta parameter for fbeta_score')
@click.option('--X_test_path', required=True, help="Path to X_test data CSV")
@click.option('--y_test_path', required=True, help="Path to y_test data CSV")
@click.option('--preprocessor_path', required=True, help='Path to preprocessor')
@click.option('--seed', type=int, help="Random seed", default=123)

def main(X_train_path, y_train_path, X_test_path, y_test_path, preprocessor_path, pos_label, beta, seed, results_to):
    '''
    Perform hyperparameter tuning on three classifiers: Decision Tree, Logistic Regression, and SVM.
    Also save the best classifier model and scores.
    '''
    np.random.seed(seed)
    set_config(transform_output="pandas")

    # Reading the training data and loading the preprocessor
    X_train = pd.read_csv(X_train_path)
    y_train = pd.read_csv(y_train_path).squeeze("columns")
    X_test = pd.read_csv(X_test_path)
    y_test = pd.read_csv(y_test_path).squeeze("columns")

    with open(preprocessor_path, "rb") as f:
        preprocessor = pickle.load(f)

    # Running the hyperparameter tuning for Decision Tree
    tree_param_dist = {
    'decisiontreeclassifier__max_depth': np.arange(1, 11)
    }
    tree_model = make_pipeline(preprocessor, DecisionTreeClassifier(random_state=seed))
    search_tree = RandomizedSearchCV(tree_model, tree_param_dist, return_train_score=True,
                                    n_jobs=-1, scoring=make_scorer(fbeta_score, pos_label=pos_label, beta=beta))
    search_tree.fit(X_train, y_train)

    # Running the hyperparameter tuning for Logistic Regression
    logistic_param_dist = {
    "logisticregression__C" : 10.0 ** np.arange(-3, 2, 1),
    "logisticregression__max_iter" : [80, 100, 500, 1000, 1500, 2000]
    }
    log_model = make_pipeline(preprocessor, LogisticRegression(random_state=seed))
    search_log = RandomizedSearchCV(log_model, logistic_param_dist, return_train_score=True,
                                    n_jobs=-1, scoring=make_scorer(fbeta_score, pos_label=pos_label, beta=beta))
    search_log.fit(X_train, y_train)

    # Running the hyperparameter tuning for SVM
    SVM_param_dist = {
    "svc__C": 10.0 ** np.arange(-3, 2, 1),
    "svc__gamma": 10.0 ** np.arange(-3, 2, 1)
    }
    svm_model = make_pipeline(preprocessor, SVC(random_state=seed))
    search_svm = RandomizedSearchCV(svm_model, SVM_param_dist, return_train_score=True,
                                    n_jobs=-1, scoring=make_scorer(fbeta_score, pos_label=pos_label, beta=beta))
    search_svm.fit(X_train, y_train)

    # Finding the best model from the best scores
    best_score = 0
    best_model = None
    model_summary = {
        'Decision Tree': [search_tree, search_tree.best_score_, search_tree.best_params_],
        'Logistic Regression': [search_log, search_log.best_score_, search_log.best_params_],
        'RBF SVM': [search_svm, search_svm.best_score_, search_svm.best_params_]
    }
    for model_name, summary in model_summary.items():
        print(f"The best F2 score for {model_name} is {summary[1]} with parameters {summary[2]}")
        best_score = max(best_score, summary[1])
        if best_score == summary[1]:
            best_model = model_name
        else:
            continue
    
    # Build Final Models with Best Parameters
    final_model = make_pipeline(preprocessor, model_summary[best_model][0])
    final_model.fit(X_train, y_train)
    result = final_model.score(X_test, y_test)
    return f'(The score on the test set using {best_model} is {result})'

    # Save final model in pickle file
    with open(os.path.join(results_to, "heart_final_model.pickle"), 'wb') as f:
        pickle.dump(final_model, f)

    cm = ConfusionMatrixDisplay.from_estimator(
    final_model,
    X_test,
    y_test
    )
    cm.to_png(os.path.join(results_to, "confusion_matrix.png"))


if __name__ == '__main__':
    main()
    
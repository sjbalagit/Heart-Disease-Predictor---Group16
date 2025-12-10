# hyperparameter_tuning.py
# author: Mantram Sharma
# date: 2025-12-06

import click
import os
import numpy as np
import pandas as pd
import pickle
from sklearn import set_config
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.optimal_hyperparameters import tune_hyperparameters
from utils.models import get_models
from utils.optimal_hyperparameters import get_best_model

@click.command()
@click.option('--train-data', required=True, help='Path to train data CSV')
@click.option('--target-col', required=True, help='Name of the target column')
@click.option('--preprocessor-path', required=True, help='Path to preprocessor')
@click.option('--pos-label', default='Heart Disease', help='Positive class label for fbeta_score')
@click.option('--beta', default=2.0, help='Beta parameter for fbeta_score')
@click.option('--seed', type=int, help="Random seed", default=123)
@click.option('--results-to', type=str, help="Path to directory where the final model will be written to")

def main(train_data, target_col, preprocessor_path, pos_label, beta, seed, results_to):
    '''
    Perform hyperparameter tuning on three classifiers: Decision Tree, Logistic Regression, and SVM.
    Also save the best classifier model and scores.
    '''
    set_config(transform_output="pandas")

    # Reading the training data and loading the preprocessor
    train_df = pd.read_csv(train_data)

    X_train = train_df.drop(columns=[target_col])
    y_train = train_df[target_col]

    with open(preprocessor_path, "rb") as f:
        preprocessor = pickle.load(f)

    # Running the hyperparameter tuning for Decision Tree
    tree_param_dist = {
    'decisiontreeclassifier__max_depth': np.arange(1, 11)
    }
    search_tree = tune_hyperparameters(X_train, y_train, get_models(random_state=seed)["decision tree"], preprocessor, tree_param_dist, pos_label, beta, seed)

    # Running the hyperparameter tuning for Logistic Regression
    logistic_param_dist = {
        "logisticregression__C" : 10.0 ** np.arange(-3, 2, 1),
        "logisticregression__max_iter" : [80, 100, 500, 1000, 1500, 2000]
    }
    search_log = tune_hyperparameters(X_train, y_train, get_models(random_state=seed)["Logistic Regression"], preprocessor, logistic_param_dist, pos_label, beta, seed)

    # Running the hyperparameter tuning for SVM
    SVM_param_dist = {
        "svc__C": 10.0 ** np.arange(-3, 2, 1),
        "svc__gamma": 10.0 ** np.arange(-3, 2, 1)
    }
    search_svm = tune_hyperparameters(X_train, y_train, get_models(random_state=seed)["RBF SVM"], preprocessor, SVM_param_dist, pos_label, beta, seed)

    # Finding the best model from the best scores and creating final_model
    model_summary = {
        'Decision Tree': [search_tree, search_tree.best_score_, search_tree.best_params_],
        'Logistic Regression': [search_log, search_log.best_score_, search_log.best_params_],
        'RBF SVM': [search_svm, search_svm.best_score_, search_svm.best_params_]
    }
    final_model, results_dict = get_best_model(model_summary)
    
    os.makedirs(results_to, exist_ok=True)
    with open(os.path.join(results_to, "final_model.pickle"), 'wb') as f:
        pickle.dump(final_model, f)

    # Save models and results
    results_df = pd.DataFrame(results_dict).T
    results_df.columns = ['F2 Score', 'Best Model Parameters']
    results_df.to_csv(os.path.join(results_to, "hyperparameter_model_results.csv"), index=True)

if __name__ == '__main__':
    main()  
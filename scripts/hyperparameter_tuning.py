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
from utils.models import get_models, get_param_dist

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

    # Running the hyperparameter tuning for all models
    model_summary = dict()
    for model_name, model_info in get_models(random_state=seed).items():
        if model_name == "Dummy Classifier":
            continue
        model_summary[model_name] = [tune_hyperparameters(X_train, y_train, model_info, preprocessor, get_param_dist()[model_name], pos_label, beta, seed), 
                                     tune_hyperparameters(X_train, y_train, model_info, preprocessor, get_param_dist()[model_name], pos_label, beta, seed).best_score_,
                                     tune_hyperparameters(X_train, y_train, model_info, preprocessor, get_param_dist()[model_name], pos_label, beta, seed).best_params_
                                    ]

    # Finding the best model from the best scores and creating final_model
    results_dict = dict()
    best_score = 0
    best_model = None
    for model_name, summary in model_summary.items():
        results_dict[model_name] = [summary[1], summary[2]]
        print(f"The best F2 score for {model_name} is {summary[1]} with parameters {summary[2]}")
        best_score = max(best_score, summary[1])
        if best_score == summary[1]:
            best_model = model_name
        else:
            continue
    final_model = model_summary[best_model][0].best_estimator_
    
    os.makedirs(results_to, exist_ok=True)
    with open(os.path.join(results_to, "final_model.pickle"), 'wb') as f:
        pickle.dump(final_model, f)

    # Save models and results
    results_df = pd.DataFrame(results_dict).T
    results_df.columns = ['F2 Score', 'Best Model Parameters']
    results_df.to_csv(os.path.join(results_to, "hyperparameter_model_results.csv"), index=True)

if __name__ == '__main__':
    main()  
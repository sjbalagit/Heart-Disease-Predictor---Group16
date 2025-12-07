# evaluate_scores.py
# author: Mantram Sharma
# date: 2025-12-06

import click
import os
import numpy as np
import pandas as pd
import pickle
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import fbeta_score
from sklearn import set_config

@click.command()
@click.option('--test-data', required=True, help='Path to test data CSV')
@click.option('--target-col', required=True, help='Name of the target column')
@click.option('--final-model-path', required=True, help='Path to the final model')
@click.option('--pos-label', default='Heart Disease', help='Positive class label for fbeta_score')
@click.option('--beta', default=2.0, help='Beta parameter for fbeta_score')
@click.option('--results-to', type=str, help="Path to directory where the final model will be written to")

def main(test_data, target_col, final_model_path, pos_label, beta, results_to):
    '''
    Evaluate the final model on the test data and save the results.
    '''
    set_config(transform_output="pandas")

    # Reading the test data
    test_df = pd.read_csv(test_data)

    X_test = test_df.drop(columns=[target_col])
    y_test = test_df[target_col]

    # Load the final model
    with open(final_model_path, "rb") as f:
        final_model = pickle.load(f)

    y_pred = final_model.predict(X_test)
    result = fbeta_score(y_test, y_pred, beta=beta, pos_label=pos_label)

    result_df = pd.DataFrame({'Best Model': ['RBF SVM'], 
                              'Test F2 Score': result})
    result_df.columns = ['Best Model', 'Test F2 Score']

    os.makedirs(results_to, exist_ok=True)
    
    result_df.to_csv(os.path.join(results_to, "evaluate_model_results.csv"), index=False)

    # Save the confusion matrix plot
    cm = ConfusionMatrixDisplay.from_estimator(final_model, X_test, y_test, labels = ['No Heart Disease','Heart Disease'])
    fig = cm.figure_
    fig.set_figwidth(8)
    fig.set_figheight(6)

    fig.tight_layout()
    fig.savefig(os.path.join(results_to, "confusion_matrix.png"))

    cm_df = pd.crosstab(
        y_test, y_pred
    )
    cm_df.columns = ["Predicted No Heart Disease", "Predicted Heart Disease"]
    cm_df.index   = ["Actual No Heart Disease", "Actual Heart Disease"]
    cm_df.to_csv(os.path.join(results_to, "confusion_matrix.csv"), index=True)

if __name__ == '__main__':
    main()  
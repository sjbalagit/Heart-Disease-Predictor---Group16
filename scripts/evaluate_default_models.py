import click
import pandas as pd
from pathlib import Path
import pickle

from sklearn.pipeline import make_pipeline
from sklearn.metrics import make_scorer, fbeta_score
from utils import mean_std_cross_val_scores
from models import get_models

    
@click.command()
@click.option('--X_train_path', required=True, help='Path to X_train data CSV')
@click.option('--y_train_path', required=True, help='Path to y_train data CSV')
@click.option('--preprocessor_path', required=True, help='Path to preprocessor')
@click.option('--pos_label', default='Heart Disease', help='Positive class label for fbeta_score')
@click.option('--beta', default=2.0, help='Beta parameter for fbeta_score')
@click.option('--random_state', default=123, help='Random state for classifiers')
@click.option('--results', required=True, help='File path to save results table, include name of the CSV file e.g., results/CV_scores_default_parameters.csv')

def main(X_train_path, y_train_path, preprocessor_path, pos_label, beta, random_state, results):

    X_train = pd.read_csv(X_train_path)
    y_train = pd.read_csv(y_train_path).squeeze("columns")

    with open(preprocessor_path, "rb") as f:
        preprocessor = pickle.load(f)

    models = get_models(random_state=random_state)
    scorer = make_scorer(fbeta_score, pos_label=pos_label, beta=beta)

    results_dict = {}
    for name, model in models.items():
        pipe = make_pipeline(preprocessor, model)
        results_dict[name] = mean_std_cross_val_scores(
            pipe, X_train, y_train, cv=5, return_train_score=True, scoring=scorer
        )
    results_df = pd.DataFrame(results_dict).T

    results_path = Path(results)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    
    results_df.to_csv(results_path, index=True)


if __name__ == "__main__":
    main()

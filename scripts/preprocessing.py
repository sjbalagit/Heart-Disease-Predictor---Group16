# preprocessing.py
# author: Sarisha Das
# date: 2023-12-01

import click
import os
import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn import set_config
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import make_column_transformer

@click.command()
@click.option('--raw-data', type=str, help="Path to raw data")
@click.option('--data-to', type=str, help="Path to directory where processed data will be written to")
@click.option('--preprocessor-to', type=str, help="Path to directory where the preprocessor object will be written to")
@click.option('--seed', type=int, help="Random seed", default=123)
@click.option('--split', type=float,     
              show_default=True,
              help="Proportion of the dataset to allocate to the test split.", 
              default=0.2)

def main(raw_data, data_to, preprocessor_to, seed, split):
    '''This script splits the raw data into train and test sets, 
    and then preprocesses the data to be used in exploratory data analysis.
    It also saves the preprocessor to be used in the model training script.'''
    np.random.seed(seed)
    set_config(transform_output="pandas")

    heart = pd.read_csv(raw_data)

    # Change values of 1 and 0 to 'Heart Disease' and 'No Heart Disease' in target
    heart['target'] = heart['target'].replace({
        1 : 'Heart Disease',
        0 : 'No Heart Disease'
    })

    # Create train test split
    train_heart, test_heart = train_test_split(
            heart, test_size = split, random_state=seed
        )

    #makedir in case the directory doesn't exist
    os.makedirs(data_to, exist_ok=True)
    os.makedirs(preprocessor_to, exist_ok=True)

    # Save raw split data
    train_heart.to_csv(os.path.join(data_to, "train_heart.csv"), index=False)
    test_heart.to_csv(os.path.join(data_to, "test_heart.csv"), index=False)

    train_targets = train_heart['target']
    train_heart = train_heart.drop(columns = ['target'])

    test_targets = test_heart['target']
    test_heart = test_heart.drop(columns = ['target'])

    # Column definitions
    binary = ["gender", "fasting_blood_sugar", "exercise_angina"]
    target = ["target"]
    ohe = ["chest_pain", "resting_electro"]
    numerical = [
        "age",
        "resting_bp",
        "serum_cholesterol",
        "max_heart_rate",
        "old_peak",
        "num_major_vessels",
    ]
    ordinal = ["slope"]
    drop = ["patient_id"]

    heart_preprocessor = make_column_transformer(
            (StandardScaler(), numerical),
            (OneHotEncoder(sparse_output=False), ohe),
            (OrdinalEncoder(), ordinal),
            ('passthrough', binary),
            ('drop', drop)
        )
    
    # Save preprocessor
    with open(
        os.path.join(preprocessor_to, "heart_preprocessor.pickle"), "wb"
    ) as f:
        pickle.dump(heart_preprocessor, f)

    heart_train_preprocessed = heart_preprocessor.fit_transform(train_heart)
    heart_test_preprocessed = heart_preprocessor.transform(test_heart)

    # Add train target back
    heart_train_preprocessed["target"] = train_targets.values

    # Add test target back
    heart_test_preprocessed["target"] = test_targets.values

    # Save processed data with target included
    heart_train_preprocessed.to_csv(
        os.path.join(data_to, "heart_train_preprocessed.csv"), index=False
    )
    heart_test_preprocessed.to_csv(
        os.path.join(data_to, "heart_test_preprocessed.csv"), index=False
    )

if __name__ == '__main__':
    main()
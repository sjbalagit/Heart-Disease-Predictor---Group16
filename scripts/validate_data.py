#!/usr/bin/env python3

import click
import pandas as pd
import pandera.pandas as pa
from pandera import Check, Column, DataFrameSchema
import os

# validate data
schema = pa.DataFrameSchema(
    {
        'patient_id': pa.Column(int, pa.Check.greater_than(0)),
        'age': pa.Column(int, pa.Check.between(0, 90), nullable=True),
        'gender': pa.Column(int, pa.Check.between(0, 1), nullable=True),
        'chest_pain': pa.Column(int, pa.Check.between(0, 3), nullable=True),
        'resting_bp': pa.Column(int, pa.Check.between(94, 200), nullable=True),
        'serum_cholesterol': pa.Column(
            int, 
            checks=[
                pa.Check(lambda s: s >= 126 and s <= 564,
                        element_wise=True,
                        # Attributed to pandera documentation:
                        # https://pandera.readthedocs.io/en/stable/checks.html#raise-warning-instead-of-error-on-check-failure
                        raise_warning=True,
                        error="There are outliers in the data values"),
            ], 
            nullable=True),
        'fasting_blood_sugar': pa.Column(int, pa.Check.between(0, 1), nullable=True),
        'resting_electro': pa.Column(int, pa.Check.between(0, 2), nullable=True),
        'max_heart_rate': pa.Column(int, pa.Check.between(71, 202), nullable=True),
        'exercise_angina': pa.Column(int, pa.Check.between(0, 1), nullable=True),
        'old_peak': pa.Column(float, pa.Check.between(0.0, 6.2), nullable=True),
        'slope': pa.Column(
            int, 
            checks=[
                pa.Check(lambda s: s >= 1 and s <= 3,
                        element_wise=True,
                        raise_warning=True,
                        error="Certain slope values are out of range"),
            ], 
            nullable=True),
        'num_major_vessels': pa.Column(int, pa.Check.between(0, 3), nullable=True),
        'target': pa.Column(int, pa.Check.isin([0, 1]))
    },
    checks=[
        pa.Check(lambda df: ~df.duplicated().any(), error="Duplicate rows found."),
        pa.Check(lambda df: ~(df.isna().all(axis=1)).any(), error="Empty rows found.")
    ]
)

@click.command()
@click.option(
    "--raw-data",
    type=str,
    required=True,
    help="Path to the raw input CSV file."
)

@click.option(
    "--data-to",
    type=str,
    help="Output path to save the validated dataset."
)
def main(raw_data, data_to):
    """Validate heart disease dataset using Pandera schema."""

    colnames = [
        'patient_id', 
        'age', 
        'gender', 
        'chest_pain', 
        'resting_bp',
        'serum_cholesterol', 
        'fasting_blood_sugar', 
        'resting_electro',
        'max_heart_rate', 
        'exercise_angina', 
        'old_peak', 
        'slope', 
        'num_major_vessels',
        'target'
    ]

    heart = pd.read_csv(raw_data, names=colnames, header=0)

    validated_df = schema.validate(heart, lazy=True)

    if not os.path.exists(data_to):
        os.makedirs(data_to)
    validated_df.to_csv(os.path.join(
        data_to, "heart_validated.csv"), index=False)


if __name__ == "__main__":
    main()

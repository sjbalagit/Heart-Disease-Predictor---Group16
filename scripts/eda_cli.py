# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.7
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# +
import os
import click
import pandas as pd
from eda_functions import (
    load_data,
    compute_summary_statistics,
    plot_target_distribution,
    plot_numerical_distributions,
    plot_boxplots,
    plot_categorical_vs_target,
)


@click.command()
@click.option("--data", type=str, required=True,
              help="Path to processed training data CSV.")
@click.option("--output-dir", type=str, required=True,
              help="Directory where all EDA plots and summary files will be saved.")
def main(data, output_dir):
    """
    Run exploratory data analysis on the heart disease dataset.

    Parameters
    ----------
    data : str
        Path to the processed heart disease dataset CSV.

    output_dir : str
        Directory where plots and summary statistics will be saved.
    """
    df = load_data(data)

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    #Summary statistics
    summary = compute_summary_statistics(df)
    summary.to_csv(os.path.join(output_dir, "summary_statistics.csv"))

    #Bar chart: Heart disease distribution 
    bar_chart = plot_target_distribution(df)
    bar_chart.save(os.path.join(output_dir, "heart_disease_counts.png"), scale_factor=2)

    #Numerical distributions 
    num_cols = ["age", "resting_bp", "serum_cholesterol", "max_heart_rate", "old_peak"]
    num_dist = plot_numerical_distributions(df, num_cols)
    num_dist.save(os.path.join(output_dir, "numerical_feature_distributions.png"), scale_factor=2)

    #Boxplots
    boxplots = plot_boxplots(df, num_cols)
    boxplots.save(os.path.join(output_dir, "boxplots_vs_target.png"), scale_factor=2)

    #Categorical vs target
    axis_titles = {
        "gender": "Gender (0 = Female, 1 = Male)",
        "chest_pain": "Chest Pain Type",
        "fasting_blood_sugar": "Fasting Blood Sugar",
        "resting_electro": "Resting ECG",
        "exercise_angia": "Exercise-Induced Angina",
        "slope": "Slope of ST Segment",
        "num_major_vessels": "Number of Major Vessels",
    }

    cat_plot = plot_categorical_vs_target(df, axis_titles)
    cat_plot.save(os.path.join(output_dir, "categorical_vs_target.png"), scale_factor=2)


if __name__ == "__main__":
    main()

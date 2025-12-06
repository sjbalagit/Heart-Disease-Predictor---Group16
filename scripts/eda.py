# eda.py
# author: Omowunmi Obadero
# date: 2025-12-05

import os, sys
import click
import pandas as pd
sys.path.append(os.path.join(os.path.dirname(__file__),'..'))
from utils.eda_helper import (
    load_data,
    compute_summary_statistics,
    plot_target_distribution,
    plot_numerical_distributions,
    plot_boxplots,
    plot_categorical_vs_target,
    plot_correlation_heatmap,  
)


@click.command()
@click.option("--data", type=str, required=True,
              help="Path to processed training data CSV.")
@click.option("--output-dir", type=str, required=True,
              help="Directory where all EDA plots and summary files will be saved.")
@click.option("--target-col", type=str, required=True,
              help="Name of the target column.")
@click.option("--num-cols", type=str, default="",
              help="Comma-separated list of numerical columns.")
@click.option("--cat-cols", type=str, default="",
              help="Comma-separated list of categorical columns.")
@click.option("--axis-titles", type=str, default="",
              help="Optional comma-separated list of column:title for categorical plots, e.g. gender:Gender")
def main(data, output_dir, target_col, num_cols, cat_cols, axis_titles):
    """
    Run exploratory data analysis on any dataset with numerical, categorical, and target columns.

    Parameters
    ----------
    data : str
        Path to CSV dataset.
    output_dir : str
        Directory where plots and summary statistics will be saved.
    target_col : str
        Name of the target column.
    num_cols : str
        Comma-separated list of numerical column names.
    cat_cols : str
        Comma-separated list of categorical column names.
    axis_titles : str
        Optional comma-separated mapping of categorical columns to axis titles.
    """
    df = load_data(data)
    os.makedirs(output_dir, exist_ok=True)

    # Parse numerical and categorical columns
    num_cols = [col.strip() for col in num_cols.split(",") if col.strip()]
    cat_cols = [col.strip() for col in cat_cols.split(",") if col.strip()]

    # Parse axis titles mapping
    axis_titles_dict = {}
    if axis_titles:
        for item in axis_titles.split(","):
            col, title = item.split(":")
            axis_titles_dict[col.strip()] = title.strip()

    # Summary statistics
    summary = compute_summary_statistics(df)
    summary.to_csv(os.path.join(output_dir, "summary_statistics.csv"))

    # Target distribution
    bar_chart = plot_target_distribution(df, target_col=target_col)
    bar_chart.save(os.path.join(output_dir, f"{target_col}_distribution.png"), scale_factor=2)

    # Numerical distributions
    if num_cols:
        num_dist = plot_numerical_distributions(df, num_cols)
        num_dist.save(os.path.join(output_dir, "numerical_feature_distributions.png"), scale_factor=2)

        # Boxplots vs target
        boxplots = plot_boxplots(df, num_cols, target_col)
        boxplots.save(os.path.join(output_dir, "boxplots_vs_target.png"), scale_factor=2)

    # Categorical vs target
    if cat_cols:
        cat_plot = plot_categorical_vs_target(df, cat_cols, target_col, axis_titles=axis_titles_dict)
        cat_plot.save(os.path.join(output_dir, "categorical_vs_target.png"), scale_factor=2)

    # Correlation heatmap
    corr_chart = plot_correlation_heatmap(df, num_cols=num_cols, cat_cols=cat_cols, target_col=target_col)
    corr_chart.save(os.path.join(output_dir, "correlation_heatmap.png"), scale_factor=2)


if __name__ == "__main__":
    main()

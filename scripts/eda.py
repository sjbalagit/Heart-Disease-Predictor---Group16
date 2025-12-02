# +
import pandas as pd
import altair as alt


def load_data(path):
    """
    Load the processed heart disease dataset.

    Parameters
    ----------
    path : str
        Path to the CSV file.

    Returns
    -------
    pandas.DataFrame
        Loaded dataset.
    """
    return pd.read_csv(path)


def compute_summary_statistics(df):
    """
    Compute summary statistics for the dataset.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataset.

    Returns
    -------
    pandas.DataFrame
        Summary statistics including numerical and categorical features.
    """
    return df.describe(include="all")



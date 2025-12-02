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



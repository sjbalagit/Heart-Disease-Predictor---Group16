import pytest
import sys
import os
import pandas as pd
import altair as alt

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.eda_helper import plot_numerical_distributions


@pytest.fixture
def simple_dataframe():
    return pd.DataFrame({
        "class": ["Benign", "Malignant", "Benign"],
        "mean_radius": [6.1, 7.8, 6.5],
        "mean_area": [145.2, 156.1, 148.3]
    })


def test_plot_numerical_distributions_basic(simple_dataframe):
    df = simple_dataframe
    chart = plot_numerical_distributions(df, ["mean_radius"])
    hist = chart.vconcat[0].hconcat[0]

    assert isinstance(chart, alt.VConcatChart), "should return an Altair VConcatChart"
    assert hist.encoding.x.shorthand == "mean_radius:Q", "x-axis should map to the correct column"
    assert hist.mark == "bar", "mark should be a bar"


def test_plot_numerical_distributions_edge_cases(simple_dataframe):
    df = simple_dataframe

    df_empty = df.iloc[0:0]
    chart_empty = plot_numerical_distributions(df_empty, ["mean_radius"])
    assert isinstance(chart_empty, alt.VConcatChart), "should return a chart for empty DataFrame"

    df_single = df.iloc[0:1]
    chart_single = plot_numerical_distributions(df_single, ["mean_radius"])
    hist_single = chart_single.vconcat[0].hconcat[0]
    assert isinstance(chart_single, alt.VConcatChart), "should return a chart for single-row DataFrame"
    assert hist_single.encoding.x.shorthand == "mean_radius:Q", "x-axis should map correctly for single row"


def test_plot_numerical_distributions_with_nans(simple_dataframe):
    df = simple_dataframe.copy()
    df.loc[1, "mean_radius"] = None  # Introduce NaN
    chart_nans = plot_numerical_distributions(df, ["mean_radius"])
    hist_nans = chart_nans.vconcat[0].hconcat[0]

    assert isinstance(chart_nans, alt.VConcatChart), "should return a chart even with NaNs"
    assert hist_nans.encoding.x.shorthand == "mean_radius:Q", "x-axis should remain correct with NaNs"

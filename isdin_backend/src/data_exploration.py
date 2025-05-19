from google.cloud import bigquery, storage
import google.auth
import os
from jinja2 import Template
from google.oauth2 import service_account
import json
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
import requests
from meteostat import Point, Daily
import statsmodels.api as sm
from statsmodels.api import add_constant
from dython import nominal
from statsmodels.stats.outliers_influence import variance_inflation_factor
import matplotlib.pyplot as plt
import seaborn as sns


def data_exploration(df, list_bucket):
    """
    Performs exploratory data analysis on the given DataFrame and list of columns.

    Args:
    df (pd.DataFrame): The input DataFrame.
    list_bucket (list): List of column names to include in the analysis.

    Returns:
    None
    """ 

    # Display basic information about the DataFrame
    print(df.info())

    # Display the number of unique values for each column
    print(df.nunique())

    # Display descriptive statistics for each column
    print(df.describe().T)

    # Seleccionar todas las columnas numéricas
    numericas = df.select_dtypes(include='number')

    # Filtrar columnas que solo contienen los valores 1 y 0
    filtro = numericas.apply(lambda col: set(col) != {0, 1} and set(col) != {0})
    columnas_seleccionadas = numericas.columns[filtro].tolist()


    # Generate and display associations between categorical variables
    nominal.associations(df[columnas_seleccionadas], figsize=(20, 8), mark_columns=True, cramers_v_bias_correction=False)

    # Calculate Variance Inflation Factor (VIF) for each feature
    vif_data = pd.DataFrame()
    vif_data["feature"] = df[list_bucket + ['units']].columns
    vif_data["VIF"] = [variance_inflation_factor(df[list_bucket + ['units']].values, i)
                       for i in range(len(df[list_bucket + ['units']].columns))]

    # Display VIF data
    print(vif_data)

    # Create a new column 'year_week' by concatenating 'year' and 'week' columns
    df['year_week'] = df.apply(lambda row: str(row['year']) + str(row['week']), axis=1)

    # Create and display a scatter plot for 'year_week' vs 'units'
    plt.scatter(df['year_week'], df['units'])
    plt.xlabel('Year Week')
    plt.ylabel('Units')
    plt.title('Scatter Plot of Units over Year Week')
    plt.show()

    # Create and display distribution plots for 'units' and 'pvpr'
    sns.displot(df["units"])
    sns.displot(df["pvpr"])

    # Create and configure the investment evolution plot
    plt.figure(figsize=(10, 6))
    colors = plt.cm.viridis(np.linspace(0, 1, len(list_bucket)))

    # Plot each variable in 'list_bucket' with different colors
    for col, color in zip(list_bucket, colors):
        plt.plot(df['year_week'], df[col], label=col, color=color)

    # Add titles and labels
    plt.title('Investment Evolution')
    plt.xlabel('Year Week')
    plt.ylabel('Investment')
    plt.legend()

    # Display the plot
    plt.show()
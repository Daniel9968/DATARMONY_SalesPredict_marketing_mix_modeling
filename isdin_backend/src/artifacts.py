from typing import List
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import pandas as pd
import statsmodels.api as sm
from sklearn.ensemble import RandomForestRegressor

def create_predict_table(X_train, model, df, updated_list_bucket, features, df_markets_week):
    """
    Create a table with model prediction results compared to actual results using the same investments.

    Parameters:
    -----------
    X_train (pandas.DataFrame): model training features.
    X_test (pandas.DataFrame): model test features
    y_train (pandas.Series): model training target variable.
    y_test (pandas.Series): model test target variable.
    model (fitted model object): A machine learning model object that has been fitted to the training data.
    df (pandas.DataFrame): Dataframe with investment values   without adstock.
    updated_list_bucket: Name of investment channels without adstock.

    Returns:
    --------
    predict_table (pandas.DataFrame):
        A DataFrame containing the combined input data, actual target values,
        and predicted values from the model.
    """

    filtered_df = df[df['future_week'] == 0].copy()
    if not isinstance(model, RandomForestRegressor):
        X_train = X_train.drop(columns=['const']) 

    X = filtered_df[X_train.columns]

    if not isinstance(model, RandomForestRegressor):
        X = sm.add_constant(X, has_constant='add')
    
    if isinstance(model, RandomForestRegressor):
        feature_order = model.feature_names_in_
        X = X[feature_order]

    y_pred = model.predict(X)
    
    predict_table = pd.concat([X, pd.Series(y_pred, name='Predict_sales')], axis=1)
    if not isinstance(model, RandomForestRegressor):
        predict_table = predict_table.drop(columns=['const'])

    # Adding investments without adstoc: 
    list_bucket_adstock = [col + '-adstock' for col in updated_list_bucket]
    columns_to_merge = filtered_df[updated_list_bucket + list_bucket_adstock + features + ['year', 'week', 'units', 'sales_without_investment']]
    
    # Merge X_total with the selected columns from filtered_df based on the keys 'week_columns' and 'year'
    predict_table = pd.merge(columns_to_merge, predict_table, on=list_bucket_adstock + features + ['year'], how='left')
    
    # Sort the predict table by 'year' and 'week' in ascending order
    predict_table = predict_table.sort_values(by=['year', 'week'], ascending=True)
    
    # Rename units column to Real
    predict_table.rename(columns={'units': 'Real_sales'}, inplace=True)

    # sort columns
    predict_table = predict_table[['year', 'week'] + sorted(predict_table.columns.difference(['year', 'week','sales_without_investment', 'Real_sales', 'Predict_sales']).tolist()) + ['sales_without_investment','Real_sales', 'Predict_sales']]
    
    if df_markets_week['diff_markets'].sum() > 0:
        predict_table = predict_table.merge(df_markets_week[['year', 'week', 'diff_markets']], on=['year', 'week'], how='left')
        
        predict_table['diff_markets'] = predict_table['diff_markets'].fillna(0)
        predict_table['Real_markets'] = predict_table['Real_sales'] * predict_table['diff_markets']
        predict_table['Predict_markets'] = predict_table['Predict_sales'] * predict_table['diff_markets']
        predict_table['markets_without_investment'] = predict_table['sales_without_investment'] * predict_table['diff_markets']
    
    # Delete columns from predict_table only if those columns exist
    columns = [col for col in df.columns if col.startswith('week_') or col.startswith('month_')] # Identify columns starting with 'week_'
    # Check if the columns to be removed are present in predict_table
    columns_to_drop = [col for col in columns if col in predict_table.columns]
    # Remove columns only if they exist in predict_table
    if 'diff_markets' in predict_table.columns:
        columns_to_drop.append('diff_markets')
    if columns_to_drop:
        predict_table = predict_table.drop(columns=columns_to_drop)

    return predict_table

def generating_saturation_points(df, list_bucket: List[str]) -> pd.DataFrame:
    """
    Elects the best regression type that fits the investment data for each channel, establishes a historical maximum value and classifies them as saturation or not.

    Args:
        df (pd.DataFrame): training df with adstock.
        list_bucket_adstock (list): list of channels with the suffix -adstock.

    Returns:
        pd.DataFrame: DataFrame containing the best model, R2 for each channel, 
                      the maximum historical investment value, and the classification of whether it has already been saturated or not.
    """
    
    data = []

    for channel in list_bucket:
        df_channel = df[df[channel] != 0].copy()
        if df_channel[channel].sum() != 0:
            X = df_channel[channel].to_numpy().astype(float).reshape(-1, 1)
            y = df_channel['units'].to_numpy()

            linear_model = LinearRegression().fit(X, y)
            poly_model = LinearRegression().fit(PolynomialFeatures(degree=2).fit_transform(X), y)
            log_model_inc = LinearRegression().fit(np.log(X), y)
            log_model_dec = LinearRegression().fit(-np.log(X), y)

            linear_r2 = linear_model.score(X, y)
            poly_r2 = poly_model.score(PolynomialFeatures(degree=2).fit_transform(X), y)
            log_r2 = log_model_inc.score(np.log(X), y)
            log_dec_r2 = log_model_dec.score(-np.log(X), y)

            r2_scores = {
                'Linear': linear_r2, 
                'Polynomial': poly_r2, 
                'Logarithmic_inc': log_r2, 
                'Logarithmic_dec': log_dec_r2
            }
        else:
            r2_scores = {
                'Linear': 0, 
                'Polynomial': 0, 
                'Logarithmic_inc': 0, 
                'Logarithmic_dec': 0
            }

        best_model = max(r2_scores, key=r2_scores.get)
        max_historical_spend = round(df[channel].max(), 2)

        if (best_model == 'Polynomial' and r2_scores[best_model] >= 0.65) or (best_model == 'Logarithmic_inc' and r2_scores[best_model] >= 0.65):
            saturation_status = 'saturado'
        else:
            saturation_status = 'aún no saturado'

        data.append({
            'Channel': channel,
            'Model': best_model,
            'R2': round(r2_scores[best_model], 2),
            'Max_Historical_Spend': max_historical_spend,
            'Saturation': saturation_status
        })

    saturation_df = pd.DataFrame(data)
    return saturation_df

def get_channel_importances_and_coefficients(df_importancia, df_coeficientes, list_bucket_adstock):
    """
    Returns a DataFrame with the coefficients and importance percentages for selected channels 
    based on a Random Forest model and a linear regression model.

    Parameters:
    - df_importancia (pd.DataFrame): DataFrame containing feature importances from a Random Forest model. 
      Should include columns ['feature', 'feature_importances'].
    - df_coeficientes (pd.DataFrame): DataFrame containing coefficients from a linear regression model.
      Should include columns ['channel', 'coefficient'].
    - list_bucket_adstock (list): List of channels to filter and include in the resulting DataFrame.

    Returns:
    - pd.DataFrame: A DataFrame with columns ['channel', 'feature_importances', 'coefficient'], 
      where 'feature_importances' are represented as percentages summing to 100%.

    Example:
    df_result = get_channel_importances_and_coefficients(df_importancia, df_coeficientes, list_bucket_adstock)
    """
    
    # Filter the Random Forest importances for channels in the specified list
    df_importances_coef = df_importancia[df_importancia['feature'].isin(list_bucket_adstock)].copy()
    
    # Rename 'feature' to 'channel' and select relevant columns
    df_importances_coef = df_importances_coef.rename(columns={'feature': 'channel'})[['channel', 'feature_importances']]
    
    # Merge with the regression coefficients DataFrame on 'channel'
    df_importances_coef = df_importances_coef.merge(df_coeficientes[['channel', 'coefficient']], on='channel', how='left')
    
    # Calculate total importance and convert to percentages
    total_importance = df_importances_coef['feature_importances'].sum()
    df_importances_coef['importance_%'] = ((df_importances_coef['feature_importances'] / total_importance) * 100).round(2)

    # Drop the original 'feature_importances' column
    df_importances_coef = df_importances_coef.drop(columns='feature_importances')

    
    return df_importances_coef
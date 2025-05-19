import numpy as np
import pandas as pd
from src.data_transformation import generate_data, adding_rows_of_future_weeks_to_forecast, transform_data, adstock_data, get_weather_data_api, calculate_sales_with_and_without_investment, transform_df_aov, transform_df_market, create_table_predict_units
from src.model_utilities import train_model, evaluation_model, generating_model_coefficients, selection_of_model_to_be_served, model_residuals, cross_validation_evaluation, train_model_random_forest, evaluation_model_random_forest, cross_validation_evaluation_random_forest, analyze_best_tree
from src.artifacts import create_predict_table, generating_saturation_points, get_channel_importances_and_coefficients
from src.cloud import cloudToLocal, filesToCloud, send_email, MyLogger
from src.data_exploration import data_exploration
from google.cloud import bigquery, storage
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import  r2_score
from datetime import datetime, timedelta
from io import BytesIO
from meteostat import Point, Daily
from dotenv import load_dotenv
from google.oauth2 import service_account
import logging
import google.cloud.logging
from google.cloud.logging_v2.handlers import CloudLoggingHandler
import json
import requests
import pandas as pd
import numpy as np
import google.auth
import os
import pickle
import statsmodels.api as sm
import smtplib
from joblib import dump, load

def main_foto(request=None):

    # Get the current date
    current_date = datetime.now()

    # Format the date as a string in the desired format (YYYYMMDD)
    current_date_format = current_date.strftime("%Y%m%d")
    logger.info("Foto: The current date has been generated")

    # Get the full path of the current file
    ruta_fichero = os.path.abspath(__file__)

    # Get only the directory path
    ruta_directorio = os.path.dirname(ruta_fichero)
    # print(f"La ruta completa del fichero es: {ruta_directorio}")

    # Generation of the paths to the SQL files
    query_investment = os.path.join(ruta_directorio, 'SQL', 'query_investment.sql')
    query_channels = os.path.join(ruta_directorio, 'SQL', 'query_channels.sql')
    query_weeks = os.path.join(ruta_directorio, 'SQL', 'query_weeks.sql')
    query_aov = os.path.join(ruta_directorio, 'SQL', 'query_aov.sql')
    query_markets = os.path.join(ruta_directorio, 'SQL', 'query_markets.sql')
    logger.info("Foto: The paths to the SQL files have been generated")

    # Opening fixed files saved in Google Storage
    cloudToLocal(bucket_name, 'SRC/tab-ccaa.xlsx', './tab-ccaa.xlsx')
    # Save the file in a variable
    tab_ccaa = pd.read_excel('./tab-ccaa.xlsx') # autonomous communities in Spain
    # Delete the file
    os.remove('./tab-ccaa.xlsx')

    cloudToLocal(bucket_name, 'SRC/tab-prov.xlsx', './tab-prov.xlsx')
    # Save the file in a variable
    tab_prov = pd.read_excel('./tab-prov.xlsx') # provinces in Spain
    # Delete the file
    os.remove('./tab-prov.xlsx')

    cloudToLocal(bucket_name, 'SRC/tab-share-ccaa.xlsx', './tab-share-ccaa.xlsx')
    # Save the file in a variable
    tab_share_ccaa = pd.read_excel('./tab-share-ccaa.xlsx') # Share by autonomous community
    # Delete the file
    os.remove('./tab-share-ccaa.xlsx')


    logger.info("Foto: The cloud SRC files has been downloaded, saved in a variable, and deleted")
    
    # Features used in the code:
    # Lists with the names of all investment channels
    list_bucket = np.append(generate_data(query_channels)['campaign_bucket_0'].tolist(), ['TV', 'CTV'])
    list_bucket_without_TV = generate_data(query_channels)['campaign_bucket_0'].tolist()
    # Other features
    weeks = generate_data(query_weeks)['week'].tolist()
    months = [f'month_{i}' for i in range(2, 13)]
    weather = ['tavg_weighted_mean', 'prcp_weighted_mean']
    months_weather = months + ['prcp_weighted_mean']
    # months_licon = months + ['redeem_ilrs']
    # week_promo = weeks + ['Other_Diff_AVG_anual', 'ISDIN_Diff_AVG_anual']

    df_raw = generate_data(query_investment, bu = '"Foto"', index_bu='"2200|1520"', country = '"ES"', list_bucket = tuple(list_bucket_without_TV)) # creating df with raw data for the BU of interest
    # df_raw.to_csv('df_raw.csv')

    # dfs for prediction beyond Sales (model is trained to calculate units sold in sales): marketis and in euros by average order value
    df_aov = transform_df_aov(query_aov, bu = '"Foto"')
    df_markets_week = transform_df_market(query_markets, df_raw, bu = '"Foto"') 

    # We add 4 lines corresponding to the 4 future weeks (we do this because the interface offers these production 
    # possibilities to the user, so we already pre-calculate things like Adstock)
    df_raw_fut_weeks = adding_rows_of_future_weeks_to_forecast(df_raw, list_bucket, country_id = 'ES', currency = 'EUR', bu = 'Foto')
    df_transf, updated_list_bucket = transform_data(df_raw_fut_weeks, list_bucket) # data cleaning and transformation
    df_adstock, list_bucket_adstock,list_just_adstock = adstock_data(df = df_transf, theta = 0.6, updated_list_bucket = updated_list_bucket) # adstock
    # data_exploration(df_transf[df_transf['future_week'] == 0], updated_list_bucket)

    # adding weather features
    df_train, weather_data_ccaa = get_weather_data_api(df = df_adstock[df_adstock['future_week'] == 0],
                                    api_key = 'AIzaSyARAcLMJsP-4R-nM8QAw0QJKWWizM5lkdI', 
                                    df_cities = tab_prov, 
                                    df_provincias = tab_ccaa, 
                                    df_share_ccaa = tab_share_ccaa,
                                    bu='Foto')                                    
    
    # Training and tuning of week models (separated by weeks, hence Week) and climate models
    model_months, X_months_train, X_months_test, y_months_train, y_months_test = train_model(df_train, list_bucket_adstock, 'units', months)
    model_months_weather, X_month_weather_train, X_month_weather_test, y_month_weather_train, y_month_weather_test = train_model(df_train, list_bucket_adstock, 'units', months_weather)
    # model_weather, X_weather_train, X_weather_test, y_weather_train, y_weather_test = train_model(df_train, list_bucket_adstock, 'units', weather)
    # model_bucket, X_bucket_train, X_bucket_test, y_bucket_train, y_bucket_test = train_model(df_train, list_bucket_adstock, 'units')
    # model_week_promo, X_week_promo_train, X_week_promo_test, y_week_promo_train, y_week_promo_test = train_model(df_train, list_bucket_adstock, 'units', week_promo)
    # model_licon, X_licon_train, X_licon_test, y_licon_train, y_licon_test = train_model(df_train, list_bucket_adstock, 'units', months_licon)

    df_evaluation_months = evaluation_model(model_months, X_months_test, y_months_test)
    df_coef_model_months = generating_model_coefficients(model_months, list_bucket_adstock)
    df_cross_validation_months = cross_validation_evaluation(df_train, list_bucket_adstock, 'units', months)

    df_evaluation_months_weather = evaluation_model(model_months_weather, X_month_weather_test, y_month_weather_test)
    df_coef_model_months_weather = generating_model_coefficients(model_months_weather, list_bucket_adstock)
    df_cross_validation_months_weather = cross_validation_evaluation(df_train, list_bucket_adstock, 'units', months_weather)

    df_saturation = generating_saturation_points(df_train, updated_list_bucket)

    fig_residuals_diagnostic_months, model_residual_tests_months = model_residuals(model_months, X_months_train, y_months_train, "foto_fig_residuals_diagnostic_months.png")

    # Since we don't have weather data for future weeks (the interface doesn't allow for future predictions), we set it to 0.
    df_adstock_future = df_adstock[df_adstock['future_week'] == 1].copy()
    df_adstock_future.loc[:, 'tavg_weighted_mean'] = 0
    df_adstock_future.loc[:, 'prcp_weighted_mean'] = 0
    df_all_dates_and_weather = pd.concat([df_train, df_adstock_future], ignore_index=True)

    # final df used as input to the interface: it contains all features, all possible dates (including future ones), actual sales, predicted by the model corresponding 
    # to real investments and predicted sales for investments equal to zero (or organic sales).
    df_final = calculate_sales_with_and_without_investment(df_all_dates_and_weather, list_just_adstock, months, model_months)
    df_final_weather = calculate_sales_with_and_without_investment(df_all_dates_and_weather, list_just_adstock, months_weather, model_months_weather)
    
    cloudToLocal(bucket_name, 'Current_model/foto_current_model.pkl', './foto_current_model.pkl')

    # Save the file in a variable as pickle format
    with open('./foto_current_model.pkl', 'rb') as f:
        current_model = pickle.load(f)
    
    # Delete the file
    os.remove('./foto_current_model.pkl')
    logger.info("Foto: The file foto_current_model.pkl has been downloaded, saved in a variable, and deleted")


    cloudToLocal(bucket_name, 'Current_model/foto_models_in_production.csv', './foto_models_in_production.csv')

    # Save the file in a variable
    with open('./foto_models_in_production.csv', 'r', encoding='utf-8') as f:
        models_in_production = pd.read_csv(f)
    
    # Delete the file
    os.remove('./foto_models_in_production.csv')
    logger.info("Foto: The file foto_models_in_production.csv has been downloaded, saved in a variable, and deleted")

    date_current_model = models_in_production["model_name"].iloc[-1].split("_")[-1]


    cloudToLocal(bucket_name, f'{date_current_model}/foto_X_months_train.pkl', './foto_X_months_train.pkl')

    # Save the file in a variable as pickle format
    with open('./foto_X_months_train.pkl', 'rb') as f:
        X_current_train = pickle.load(f)
    
    # Delete the file
    os.remove('./foto_X_months_train.pkl')
    logger.info("Foto: The file foto_X_months_train.pkl has been downloaded, saved in a variable, and deleted")



    cloudToLocal(bucket_name, f'{date_current_model}/foto_y_months_train.pkl', './foto_y_months_train.pkl')

    # Save the file in a variable as pickle format
    with open('./foto_y_months_train.pkl', 'rb') as f:
        y_current_train = pickle.load(f)
    
    # Delete the file
    os.remove('./foto_y_months_train.pkl')
    logger.info("Foto: The file foto_y_months_train.pkl has been downloaded, saved in a variable, and deleted")

    if set(X_current_train.columns) != set(X_months_train.columns):
        current_model = model_months
        X_current_train = X_months_train
        y_current_train = y_months_train

    ### best_models.pkl
    # To select the model in production we have to compare the r2 of this week's model (trained in the previous lines) 
    # with the model that is currently in production (tested with the same data as the current week's model)

    # We must delete the constants from X, as future functions add
    # Updating X_test (This week's model testing dataset X): 
    X_months_test_without_const = X_months_test.loc[:, X_months_test.columns != 'const'] #deleting constant
    # X test dataset of the current production model:
    X_current_train = X_current_train.loc[:, X_current_train.columns != 'const'] #deleting constant

    # Compare and remove lines from X_months_test that are also in X_current_train, as we have to ensure that the model is not being tested with training data.
    X_months_test_unique = pd.merge(X_months_test_without_const, X_current_train, how='left', indicator=True)
    X_months_test_unique = X_months_test_unique[X_months_test_unique['_merge'] == 'left_only'].drop_duplicates(subset=X_months_test.columns).drop(columns='_merge')
    # Getting the indices of the unique data
    indices_unique = X_months_test_unique.index
    # Filtering y_test with the same indices
    y_months_test_unique = y_months_test[indices_unique]

    # We re-evaluated the current production model, using the most current test X:
    df_evaluation_current_old_model = evaluation_model(current_model, X_months_test_unique, y_months_test_unique)

    # Function to select the new model in production (the one that presents the highest r2 with current data)
    current_model, current_evaluation, model_name,start_date, end_date, X_current_train, y_current_train = selection_of_model_to_be_served(model_months, current_model, df_evaluation_months, df_evaluation_current_old_model, models_in_production, X_months_train, y_months_train, X_current_train, y_current_train)

    # Add information about the model in production to a table
    new_row = pd.DataFrame({'start_date': [start_date ], 'end_date': [end_date], 'model_name': [model_name]})
    if models_in_production['end_date'].iloc[-1] != new_row['end_date'].iloc[-1]:
        models_in_production = pd.concat([models_in_production, new_row], ignore_index=True)

    # Creating the table shown in the interface: channel investments, sales_without_investment, Real, Predict (for sales and market)
    predict_table_months = create_predict_table(X_months_train, current_model, df_final, updated_list_bucket, months, df_markets_week)
    predict_table_weather = create_predict_table(X_month_weather_train, model_months_weather, df_final_weather, updated_list_bucket, months_weather, df_markets_week)

    # Generate current files (with NEW CURRENT MODEL - WITH THE WINNING MODEL)
    df_final_current = calculate_sales_with_and_without_investment(df_all_dates_and_weather, list_just_adstock, months, current_model)
    predict_table_current = create_predict_table(X_current_train, current_model, df_final_current, updated_list_bucket, months, df_markets_week)
    fig_residuals_diagnostic_current, model_residual_tests_current = model_residuals(current_model, X_current_train, y_current_train, "foto_fig_residuals_diagnostic_current.png")
    df_coef_model_current = generating_model_coefficients(current_model, list_bucket_adstock)
    foto_df_year = create_table_predict_units(df_final_current, df_coef_model_current, updated_list_bucket, 'year')
    foto_df_month = create_table_predict_units(df_final_current, df_coef_model_current, updated_list_bucket, 'month')
    foto_df_week = create_table_predict_units(df_final_current, df_coef_model_current, updated_list_bucket, 'week')


    # Upload files to cloud storage
    filesToCloud('foto_model_months.pkl', pickle.dumps(model_months), 'application/octet-stream', bucket_name, current_date_format)
    filesToCloud('foto_model_weather.pkl', pickle.dumps(model_months_weather), 'application/octet-stream', bucket_name, current_date_format)
    filesToCloud('foto_evaluation_months.csv', df_evaluation_months.to_csv(index=False), 'text/csv', bucket_name, current_date_format)
    filesToCloud('foto_evaluation_weather.csv', df_evaluation_months_weather.to_csv(index=False), 'text/csv', bucket_name, current_date_format)
    filesToCloud('foto_coefficients_month.csv', df_coef_model_months.to_csv(index=False), 'text/csv', bucket_name, current_date_format)
    filesToCloud('foto_coefficients_weather.csv', df_coef_model_months_weather.to_csv(index=False), 'text/csv', bucket_name, current_date_format)
    filesToCloud('foto_saturation.csv', df_saturation.to_csv(index=False), 'text/csv', bucket_name, current_date_format)
    
    filesToCloud('foto_df.csv', df_final.to_csv(index=False), 'text/csv', bucket_name, current_date_format)
    filesToCloud('foto_df_final_weather.csv', df_final_weather.to_csv(index=False), 'text/csv', bucket_name, current_date_format)
    
    filesToCloud('foto_current_model.pkl', pickle.dumps(current_model), 'application/octet-stream', bucket_name, 'Current_model')
    filesToCloud('foto_evaluation_current_model.csv', current_evaluation.to_csv(index=False), 'text/csv', bucket_name, 'Current_model')
    filesToCloud('foto_models_in_production.csv', models_in_production.to_csv(index=False), 'text/csv', bucket_name, 'Current_model')
    
    filesToCloud('foto_predict_table_months.csv', predict_table_months.to_csv(index=False), 'text/csv', bucket_name, current_date_format)
    filesToCloud('foto_predict_table_weather.csv', predict_table_weather.to_csv(index=False), 'text/csv', bucket_name, current_date_format)

    filesToCloud('foto_model_residual_tests_months.pkl', pickle.dumps(model_residual_tests_months), 'application/octet-stream', bucket_name, current_date_format)
    filesToCloud('foto_weather_data_ccaa.csv', weather_data_ccaa.to_csv(index=False), 'text/csv', bucket_name, current_date_format)
    
    filesToCloud('foto_fig_residuals_diagnostic_months.png', './foto_fig_residuals_diagnostic_months.png', 'image/png', bucket_name, current_date_format)
    os.remove('./foto_fig_residuals_diagnostic_months.png')
    filesToCloud('foto_fig_residuals_diagnostic_current.png', './foto_fig_residuals_diagnostic_current.png', 'image/png', bucket_name, 'Current_model')
    os.remove('./foto_fig_residuals_diagnostic_current.png')

    filesToCloud('foto_X_months_train.pkl', pickle.dumps(X_months_train), 'application/octet-stream', bucket_name, current_date_format)
    filesToCloud('foto_y_months_train.pkl', pickle.dumps(y_months_train), 'application/octet-stream', bucket_name, current_date_format)
    
    filesToCloud('foto_df_final_current.csv', df_final_current.to_csv(index=False), 'text/csv', bucket_name, 'Current_model')
    filesToCloud('foto_predict_table_current.csv', predict_table_current.to_csv(index=False), 'text/csv', bucket_name, 'Current_model')
    
    filesToCloud('foto_model_residual_tests_current.pkl', pickle.dumps(model_residual_tests_current), 'application/octet-stream', bucket_name, 'Current_model')

    filesToCloud('foto_df_coef_model_current.csv', df_coef_model_current.to_csv(index=False), 'text/csv', bucket_name, 'Current_model')

    filesToCloud('foto_df_aov.csv', df_aov.to_csv(index=False), 'text/csv', bucket_name, current_date_format)
    filesToCloud('foto_df_markets_week.csv', df_markets_week.to_csv(index=False), 'text/csv', bucket_name, current_date_format)

    filesToCloud('foto_df_year.csv', foto_df_year.to_csv(index=False), 'text/csv', bucket_name, 'Current_model')
    filesToCloud('foto_df_month.csv', foto_df_month.to_csv(index=False), 'text/csv', bucket_name, 'Current_model')
    filesToCloud('foto_df_week.csv', foto_df_week.to_csv(index=False), 'text/csv', bucket_name, 'Current_model')

    logger.info("Foto: The files have been uploaded to Cloud Storage")

    logger.info("Foto: End of process")

def main_ceutics(request=None):

    # Get the current date
    current_date = datetime.now()

    # Format the date as a string in the desired format (YYYYMMDD)
    current_date_format = current_date.strftime("%Y%m%d")
    logger.info("Ceutics: The current date has been generated")

    # Get the full path of the current file
    ruta_fichero = os.path.abspath(__file__)

    # Get only the directory path
    ruta_directorio = os.path.dirname(ruta_fichero)
    # print(f"La ruta completa del fichero es: {ruta_directorio}")

    # Generation of the paths to the SQL files
    query_investment = os.path.join(ruta_directorio, 'SQL', 'query_investment.sql')
    query_channels = os.path.join(ruta_directorio, 'SQL', 'query_channels.sql')
    query_weeks = os.path.join(ruta_directorio, 'SQL', 'query_weeks.sql')
    query_aov = os.path.join(ruta_directorio, 'SQL', 'query_aov.sql')
    query_markets = os.path.join(ruta_directorio, 'SQL', 'query_markets.sql')
    logger.info("Ceutics: The paths to the SQL files have been generated")

    # Opening fixed files saved in Google Storage
    cloudToLocal(bucket_name, 'SRC/tab-ccaa.xlsx', './tab-ccaa.xlsx')
    # Save the file in a variable
    tab_ccaa = pd.read_excel('./tab-ccaa.xlsx')
    # Delete the file
    os.remove('./tab-ccaa.xlsx')

    cloudToLocal(bucket_name, 'SRC/tab-prov.xlsx', './tab-prov.xlsx')
    # Save the file in a variable
    tab_prov = pd.read_excel('./tab-prov.xlsx')
    # Delete the file
    os.remove('./tab-prov.xlsx')

    ########### Cambiar
    cloudToLocal(bucket_name, 'SRC/tab-share-ccaa.xlsx', './tab-share-ccaa.xlsx')
    # Save the file in a variable
    tab_share_ccaa = pd.read_excel('./tab-share-ccaa.xlsx')
    # Delete the file
    os.remove('./tab-share-ccaa.xlsx')


    logger.info("Ceutics: The cloud SRC files has been downloaded, saved in a variable, and deleted")
    # Features used in the code:
    # Lists with the names of all investment channels
    list_bucket = np.append(generate_data(query_channels)['campaign_bucket_0'].tolist(), ['TV', 'CTV'])
    list_bucket_without_TV = generate_data(query_channels)['campaign_bucket_0'].tolist()
    # Other features
    weeks = generate_data(query_weeks)['week'].tolist()
    months = [f'month_{i}' for i in range(2, 13)]
    previous_units = [f'month_{i}' for i in range(2, 13)]

    df_raw = generate_data(query_investment, bu = '"Aesthetics"', index_bu='"0116"', country = '"ES"', list_bucket = tuple(list_bucket_without_TV)) # creating df with raw data for the BU of interest
    # dfs for prediction beyond Sales (model is trained to calculate units sold in sales): marketis and in euros by average order value
    df_aov = transform_df_aov(query_aov, bu = '"Aesthetics"')
    df_markets_week = transform_df_market(query_markets, df_raw, bu = '"Aesthetics"')
    
    # We add 4 lines corresponding to the 4 future weeks (we do this because the interface offers these production 
    # possibilities to the user, so we already pre-calculate things like Adstock)
    df_raw_fut_weeks = adding_rows_of_future_weeks_to_forecast(df_raw, list_bucket, country_id = 'ES', currency = 'EUR', bu = 'Aesthetics')
    df_transf, updated_list_bucket = transform_data(df_raw_fut_weeks, list_bucket)
    df_adstock, list_bucket_adstock,list_just_adstock = adstock_data(df = df_transf, theta = 0.6, updated_list_bucket = updated_list_bucket)
    # data_exploration(df_transf[df_transf['future_week'] == 0], updated_list_bucket)

    df_train, weather_data_ccaa = get_weather_data_api(df = df_adstock[df_adstock['future_week'] == 0],
                                    api_key = 'AIzaSyARAcLMJsP-4R-nM8QAw0QJKWWizM5lkdI', 
                                    df_cities = tab_prov, 
                                    df_provincias = tab_ccaa, 
                                    df_share_ccaa = tab_share_ccaa,
                                    bu='Aesthetics')    
    # Save the last 5 rows
    last_rows = df_train.tail(5)

    # Apply the filter to the rest of the DataFrame
    filtered_df_train = df_train.iloc[:-5]
    filtered_df_train = filtered_df_train[(filtered_df_train['units'] <= 12500) & (filtered_df_train['units'] >= 1000)]

    # Combine the filtered rows with the last 5 rows
    df_train = pd.concat([filtered_df_train, last_rows])   

    # Training and tuning of week models (separated by weeks, hence Week) and climate models
    model_regression, X_regression_train, X_regression_test, y_regression_train, y_regression_test = train_model(df_train, list_bucket_adstock, 'units', ['previous_units_1'])
    model_rf, X_rf_train, X_rf_test, y_rf_train, y_rf_test = train_model_random_forest(df_train, list_bucket_adstock, 'units', ['previous_units_1'])

    # # Model evaluation
    df_evaluation_regression = evaluation_model(model_regression, X_regression_test, y_regression_test)
    df_coef_model_regression = generating_model_coefficients(model_regression, list_bucket_adstock)
    df_cross_validation_regression = cross_validation_evaluation(df_train, list_bucket_adstock, 'units', ['previous_units_1'])

    df_evaluation_rf, df_importances_rf = evaluation_model_random_forest(model_rf, X_rf_test, y_rf_test)
    df_cross_validation_rf = cross_validation_evaluation_random_forest(df_train, list_bucket_adstock, 'units', ['previous_units_1'])

    df_saturation = generating_saturation_points(df_train, updated_list_bucket) # data saturation for CEUTICS

    # Residual information for regression model
    fig_residuals_diagnostic_regression, model_residual_tests_regression = model_residuals(model_regression, X_regression_train, y_regression_train, "ceutics_fig_residuals_diagnostic_regression.png")

    # Since we don't have weather data for future weeks (the interface doesn't allow for future predictions), we set it to 0.
    df_adstock_future = df_adstock[df_adstock['future_week'] == 1].copy()
    df_adstock_future.loc[:, 'tavg_weighted_mean'] = 0
    df_adstock_future.loc[:, 'prcp_weighted_mean'] = 0
    df_all_dates_and_weather = pd.concat([df_train, df_adstock_future], ignore_index=True)

    # final df used as input to the interface: it contains all features, all possible dates (including future ones), actual sales, predicted by the model corresponding 
    # to real investments and predicted sales for investments equal to zero (or organic sales).
    df_final = calculate_sales_with_and_without_investment(df_all_dates_and_weather, list_just_adstock, ['previous_units_1'], model_rf)    

    #RandomForest
    cloudToLocal(bucket_name, 'Current_model/ceutics_current_model_rf.pkl', './ceutics_current_model_rf.pkl')

    # Save the file in a variable as pickle format
    with open('./ceutics_current_model_rf.pkl', 'rb') as f:
        current_model_rf = pickle.load(f)

    # Delete the file
    os.remove('./ceutics_current_model_rf.pkl')
    logger.info("Ceutics: The file ceutics_current_model_rf.pkl has been downloaded, saved in a variable, and deleted")


    cloudToLocal(bucket_name, 'Current_model/ceutics_rf_models_in_production.csv', './ceutics_rf_models_in_production.csv')

    # Save the file in a variable
    with open('./ceutics_rf_models_in_production.csv', 'r', encoding='utf-8') as f:
        models_in_production_rf = pd.read_csv(f)
    
    # Delete the file
    os.remove('./ceutics_rf_models_in_production.csv')
    logger.info("Ceutics: The file ceutics_rf_models_in_production.csv has been downloaded, saved in a variable, and deleted")

    date_current_model_rf = models_in_production_rf["model_name"].iloc[-1].split("_")[-1]


    cloudToLocal(bucket_name, f'{date_current_model_rf}/ceutics_X_rf_train.pkl', './ceutics_X_rf_train.pkl')

    # Save the file in a variable as pickle format
    with open('./ceutics_X_rf_train.pkl', 'rb') as f:
        X_current_train_rf = pickle.load(f)

    # Delete the file
    os.remove('./ceutics_X_rf_train.pkl')
    logger.info("Ceutics: The file ceutics_X_rf_train.pkl has been downloaded, saved in a variable, and deleted")



    cloudToLocal(bucket_name, f'{date_current_model_rf}/ceutics_y_rf_train.pkl', './ceutics_y_rf_train.pkl')

    # Save the file in a variable as pickle format
    with open('./ceutics_y_rf_train.pkl', 'rb') as f:
        y_current_train_rf = pickle.load(f)

    # Delete the file
    os.remove('./ceutics_y_rf_train.pkl')
    logger.info("Ceutics: The file ceutics_y_rf_train.pkl has been downloaded, saved in a variable, and deleted")

    if set(X_current_train_rf.columns) != set(X_rf_train.columns):
        current_model_rf = model_rf
        X_current_train_rf = X_rf_train
        y_current_train_rf = y_rf_train

    ### best_models.pkl

    # To select the model in production, we need to compare the R² score of this week's model (trained above)
    # with the model currently in production, both tested on the same data for consistency.

    # First, ensure there’s no data leakage by removing any rows in X_rf_test that are also in the training set (X_current_train_rf).
    # This avoids evaluating the model on data it has already seen during training.

    # Step 1: Make a copy of X_rf_test to keep the original data intact and add an 'original_index' column 
    # to retain the original row positions.
    X_rf_test = X_rf_test.copy()  
    X_rf_test['original_index'] = X_rf_test.index
    # Step 2: Identify unique rows in X_rf_test by performing a left join with X_current_train_rf and marking entries
    # only present in X_rf_test. This filters out any rows that appear in both datasets, ensuring no overlap.
    X_rf_test_unique = pd.merge(X_rf_test, X_current_train_rf, how='left', indicator=True)
    X_rf_test_unique = X_rf_test_unique[X_rf_test_unique['_merge'] == 'left_only'].drop_duplicates(subset=X_rf_test.columns).drop(columns='_merge')
    # Step 3: Restore the original index from 'original_index' in X_rf_test_unique to maintain consistency.
    X_rf_test_unique.set_index('original_index', inplace=True)
    # Step 4: Extract indices of unique rows, which will be used to filter y_rf_test to match the cleaned X_rf_test_unique.
    indices_unique = X_rf_test_unique.index
    # Step 5: Filter y_rf_test to include only rows corresponding to the unique data in X_rf_test_unique.
    y_rf_test_unique = y_rf_test[indices_unique]
    # Step 6: Reindex X_rf_test_unique columns to align with X_current_train_rf’s structure, ensuring consistent feature order.
    X_rf_test_unique = X_rf_test_unique.reindex(columns=X_current_train_rf.columns)

    # We re-evaluated the current production model, using the most current test X:
    df_evaluation_current_old_model, df_importances_rf_current_old_model = evaluation_model_random_forest(current_model_rf, X_rf_test_unique, y_rf_test_unique)

    # Function to select the new model in production (the one that presents the highest r2 with current data)
    current_model_rf, evaluation_current_model_rf, model_name,start_date, end_date, X_current_train_rf, y_current_train_rf = selection_of_model_to_be_served(model_rf, current_model_rf, df_evaluation_rf, df_evaluation_current_old_model, models_in_production_rf, X_rf_train, y_rf_train, X_current_train_rf, y_current_train_rf)

    # We select the best tree from RandomForest, generate a table with details about its nodes and also plot the tree
    df_nodes_best_tree, fig_best_tree = analyze_best_tree(current_model_rf, X_current_train_rf, y_current_train_rf, ['year'] + list_bucket_adstock + ['previous_unit_1'], "ceutics_fig_best_tree.png")

    # Add information about the model in production to a table
    new_row = pd.DataFrame({'start_date': [start_date ], 'end_date': [end_date], 'model_name': [model_name]})
    if models_in_production_rf['end_date'].iloc[-1] != new_row['end_date'].iloc[-1]:
        models_in_production_rf = pd.concat([models_in_production_rf, new_row], ignore_index=True)

    # Creating the table shown in the interface: channel investments, sales_without_investment, Real, Predict (for sales and market)
    predict_table_rf = create_predict_table(X_rf_train, current_model_rf, df_final, updated_list_bucket, ['previous_units_1'], df_markets_week)

    # Generate current files (with NEW CURRENT MODEL - WITH THE WINNING MODEL)
    df_final_current = calculate_sales_with_and_without_investment(df_all_dates_and_weather, list_just_adstock, ['previous_units_1'], current_model_rf)
    predict_table_current = create_predict_table(X_current_train_rf, current_model_rf, df_final_current, updated_list_bucket, ['previous_units_1'], df_markets_week)
    df_evaluation_current_rf, df_importances_current_rf = evaluation_model_random_forest(current_model_rf, X_rf_test_unique, y_rf_test_unique)

    #Linear Regression
    cloudToLocal(bucket_name, 'Current_model/ceutics_current_model_regression.pkl', './ceutics_current_model_regression.pkl')

    # Save the file in a variable as pickle format
    with open('./ceutics_current_model_regression.pkl', 'rb') as f:
        current_model_regression = pickle.load(f)
    
    # Delete the file
    os.remove('./ceutics_current_model_regression.pkl')
    logger.info("Ceutics: The file ceutics_current_model_regression.pkl has been downloaded, saved in a variable, and deleted")


    cloudToLocal(bucket_name, 'Current_model/ceutics_regression_models_in_production.csv', './ceutics_regression_models_in_production.csv')

    # Save the file in a variable
    with open('./ceutics_regression_models_in_production.csv', 'r', encoding='utf-8') as f:
        models_in_production_regression = pd.read_csv(f)
    
    # Delete the file
    os.remove('./ceutics_regression_models_in_production.csv')
    logger.info("Ceutics: The file ceutics_regression_models_in_production.csv has been downloaded, saved in a variable, and deleted")

    date_current_model_regression = models_in_production_regression["model_name"].iloc[-1].split("_")[-1]


    cloudToLocal(bucket_name, f'{date_current_model_regression}/ceutics_X_regression_train.pkl', './ceutics_X_regression_train.pkl')

    # Save the file in a variable as pickle format
    with open('./ceutics_X_regression_train.pkl', 'rb') as f:
        X_current_train_regression = pickle.load(f)
    
    # Delete the file
    os.remove('./ceutics_X_regression_train.pkl')
    logger.info("Ceutics: The file ceutics_X_regression_train.pkl has been downloaded, saved in a variable, and deleted")



    cloudToLocal(bucket_name, f'{date_current_model_regression}/ceutics_y_regression_train.pkl', './ceutics_y_regression_train.pkl')

    # Save the file in a variable as pickle format
    with open('./ceutics_y_regression_train.pkl', 'rb') as f:
        y_current_train_regression = pickle.load(f)
    
    # Delete the file
    os.remove('./ceutics_y_regression_train.pkl')
    logger.info("Ceutics: The file ceutics_y_regression_train.pkl has been downloaded, saved in a variable, and deleted")

    if set(X_current_train_regression.columns) != set(X_regression_train.columns):
        current_model_regression = model_regression
        X_current_train_regression = X_regression_train
        y_current_train_regression = y_regression_train

    # Updating X_test
    X_regression_test_without_const = X_regression_test.loc[:, X_regression_test.columns != 'const'] #deleting constant
    X_current_train_regression = X_current_train_regression.loc[:, X_current_train_regression.columns != 'const'] #deleting constant

    # Compare and remove lines from X_regression_test that are also in X_current_train_regression, as we have to ensure that the model is not being tested with training data.
    X_regression_test_unique = pd.merge(X_regression_test_without_const, X_current_train_regression, how='left', indicator=True)
    X_regression_test_unique = X_regression_test_unique[X_regression_test_unique['_merge'] == 'left_only'].drop_duplicates(subset=X_regression_test.columns).drop(columns='_merge')
    # Getting the indices of the unique data
    indices_unique = X_regression_test_unique.index
    # Filtering y_test with the same indices
    y_regression_test_unique = y_regression_test[indices_unique]

    df_evaluation_current_old_model = evaluation_model(current_model_regression, X_regression_test_unique, y_regression_test_unique)

    # Selecting the model
    current_model_regression, evaluation_current_model_regression, model_name,start_date, end_date, X_current_train_regression, y_current_train_regression = selection_of_model_to_be_served(model_regression, current_model_regression, df_evaluation_regression, df_evaluation_current_old_model, models_in_production_regression, X_regression_train, y_regression_train, X_current_train_regression, y_current_train_regression)

    # Add information about the model in production to a table
    new_row = pd.DataFrame({'start_date': [start_date ], 'end_date': [end_date], 'model_name': [model_name]})
    if models_in_production_regression['end_date'].iloc[-1] != new_row['end_date'].iloc[-1]:
        models_in_production_regression = pd.concat([models_in_production_regression, new_row], ignore_index=True)

    # Generate current files
    df_coef_model_current_regression = generating_model_coefficients(current_model_regression, list_bucket_adstock)
    # Before calculating the residuals for the current regression, we must modify X_current_train_regression:
    # We add the constant to X again, since for the residue function the accepted X has the constant
    X_current_train_regression = sm.add_constant(X_current_train_regression, prepend=True)
    fig_residuals_diagnostic_current_regression, model_residual_tests_current_regression = model_residuals(current_model_regression, X_current_train_regression, y_current_train_regression, "ceutics_fig_residuals_diagnostic_current_regression.png")

    # Created a table with the importance values ​​of each channel according to RandomForest (current) and the coefficients according to the regression (current)
    # In the table, the feature importances of the channels from the Random Forest model are represented as percentages, normalized to sum to 100%. Meanwhile, 
    # the coefficients from the regression model are presented as obtained directly from the model, without normalization
    df_importances_and_coef = get_channel_importances_and_coefficients(df_importances_current_rf, df_coef_model_current_regression, list_bucket_adstock)
    ceutics_df_year = create_table_predict_units(df_final_current, df_coef_model_current_regression, updated_list_bucket, 'year')
    ceutics_df_month = create_table_predict_units(df_final_current, df_coef_model_current_regression, updated_list_bucket, 'month')
    ceutics_df_week = create_table_predict_units(df_final_current, df_coef_model_current_regression, updated_list_bucket, 'week')

    # Upload files to cloud storage
    filesToCloud('ceutics_model_rf.pkl', pickle.dumps(model_rf), 'application/octet-stream', bucket_name, current_date_format)
    filesToCloud('ceutics_model_regression.pkl', pickle.dumps(model_regression), 'application/octet-stream', bucket_name, current_date_format)
    filesToCloud('ceutics_evaluation_rf.csv', df_evaluation_rf.to_csv(index=False), 'text/csv', bucket_name, current_date_format)
    filesToCloud('ceutics_evaluation_regression.csv', df_evaluation_regression.to_csv(index=False), 'text/csv', bucket_name, current_date_format)
    filesToCloud('ceutics_importances_rf.csv', df_importances_rf.to_csv(index=False), 'text/csv', bucket_name, current_date_format)
    filesToCloud('ceutics_coefficients_regression.csv', df_coef_model_regression.to_csv(index=False), 'text/csv', bucket_name, current_date_format)
    filesToCloud('ceutics_saturation.csv', df_saturation.to_csv(index=False), 'text/csv', bucket_name, current_date_format)
    
    filesToCloud('ceutics_df.csv', df_final.to_csv(index=False), 'text/csv', bucket_name, current_date_format)
    
    filesToCloud('ceutics_current_model_rf.pkl', pickle.dumps(current_model_rf), 'application/octet-stream', bucket_name, 'Current_model')
    filesToCloud('ceutics_current_model_regression.pkl', pickle.dumps(current_model_regression), 'application/octet-stream', bucket_name, 'Current_model')
    filesToCloud('ceutics_evaluation_current_model_rf.csv', evaluation_current_model_rf.to_csv(index=False), 'text/csv', bucket_name, 'Current_model')
    filesToCloud('ceutics_evaluation_current_model_regression.csv', evaluation_current_model_regression.to_csv(index=False), 'text/csv', bucket_name, 'Current_model')
    filesToCloud('ceutics_models_in_production_rf.csv', models_in_production_rf.to_csv(index=False), 'text/csv', bucket_name, 'Current_model')
    filesToCloud('ceutics_models_in_production_regression.csv', models_in_production_regression.to_csv(index=False), 'text/csv', bucket_name, 'Current_model')
    filesToCloud('ceutics_df_importances_and_coef.csv', df_importances_and_coef.to_csv(index=False), 'text/csv', bucket_name, 'Current_model')
    
    filesToCloud('ceutics_predict_table_rf.csv', predict_table_rf.to_csv(index=False), 'text/csv', bucket_name, current_date_format)

    filesToCloud('ceutics_model_residual_tests_regression.pkl', pickle.dumps(model_residual_tests_regression), 'application/octet-stream', bucket_name, current_date_format)
    
    filesToCloud('ceutics_fig_residuals_diagnostic_regression.png', './ceutics_fig_residuals_diagnostic_regression.png', 'image/png', bucket_name, current_date_format)
    os.remove('./ceutics_fig_residuals_diagnostic_regression.png')
    filesToCloud('ceutics_fig_residuals_diagnostic_current_regression.png', './ceutics_fig_residuals_diagnostic_current_regression.png', 'image/png', bucket_name, 'Current_model')
    os.remove('./ceutics_fig_residuals_diagnostic_current_regression.png')

    filesToCloud('ceutics_X_rf_train.pkl', pickle.dumps(X_rf_train), 'application/octet-stream', bucket_name, current_date_format)
    filesToCloud('ceutics_X_regression_train.pkl', pickle.dumps(X_regression_train), 'application/octet-stream', bucket_name, current_date_format)
    filesToCloud('ceutics_y_rf_train.pkl', pickle.dumps(y_rf_train), 'application/octet-stream', bucket_name, current_date_format)
    filesToCloud('ceutics_y_regression_train.pkl', pickle.dumps(y_regression_train), 'application/octet-stream', bucket_name, current_date_format)
    
    filesToCloud('ceutics_df_final_current.csv', df_final_current.to_csv(index=False), 'text/csv', bucket_name, 'Current_model')
    filesToCloud('ceutics_predict_table_current.csv', predict_table_current.to_csv(index=False), 'text/csv', bucket_name, 'Current_model')
    
    filesToCloud('ceutics_model_residual_tests_current_regression.pkl', pickle.dumps(model_residual_tests_current_regression), 'application/octet-stream', bucket_name, 'Current_model')

    filesToCloud('ceutics_df_coef_model_current_regression.csv', df_coef_model_current_regression.to_csv(index=False), 'text/csv', bucket_name, 'Current_model')
    filesToCloud('ceutics_df_importances_current_rf.csv', df_importances_current_rf.to_csv(index=False), 'text/csv', bucket_name, 'Current_model')

    filesToCloud('ceutics_df_aov.csv', df_aov.to_csv(index=False), 'text/csv', bucket_name, current_date_format)
    filesToCloud('ceutics_df_markets_week.csv', df_markets_week.to_csv(index=False), 'text/csv', bucket_name, current_date_format)

    filesToCloud('ceutics_df_nodes_best_tree.csv', df_nodes_best_tree.to_csv(index=False), 'text/csv', bucket_name, 'Current_model')

    filesToCloud('ceutics_fig_best_tree.png', './ceutics_fig_best_tree.png', 'image/png', bucket_name, 'Current_model')
    os.remove('./ceutics_fig_best_tree.png')

    filesToCloud('ceutics_df_year.csv', ceutics_df_year.to_csv(index=False), 'text/csv', bucket_name, 'Current_model')
    filesToCloud('ceutics_df_month.csv', ceutics_df_month.to_csv(index=False), 'text/csv', bucket_name, 'Current_model')
    filesToCloud('ceutics_df_week.csv', ceutics_df_week.to_csv(index=False), 'text/csv', bucket_name, 'Current_model')

    logger.info("Ceutics: The files have been uploaded to Cloud Storage")

    logger.info("Ceutics: End of process")

def main_derma_acniben(request=None):

    # Get the current date
    current_date = datetime.now()

    # Format the date as a string in the desired format (YYYYMMDD)
    current_date_format = current_date.strftime("%Y%m%d")
    logger.info("Derma Acniben: The current date has been generated")

    # Get the full path of the current file
    ruta_fichero = os.path.abspath(__file__)

    # Get only the directory path
    ruta_directorio = os.path.dirname(ruta_fichero)
    # print(f"La ruta completa del fichero es: {ruta_directorio}")

    # Generation of the paths to the SQL files
    query_investment = os.path.join(ruta_directorio, 'SQL', 'query_investment_brand.sql')
    query_channels = os.path.join(ruta_directorio, 'SQL', 'query_channels.sql')
    query_weeks = os.path.join(ruta_directorio, 'SQL', 'query_weeks.sql')
    query_aov = os.path.join(ruta_directorio, 'SQL', 'query_aov.sql')
    query_markets = os.path.join(ruta_directorio, 'SQL', 'query_markets.sql')
    logger.info("Derma Acniben: The paths to the SQL files have been generated")

    # Opening fixed files saved in Google Storage
    cloudToLocal(bucket_name, 'SRC/tab-ccaa.xlsx', './tab-ccaa.xlsx')
    # Save the file in a variable
    tab_ccaa = pd.read_excel('./tab-ccaa.xlsx')
    # Delete the file
    os.remove('./tab-ccaa.xlsx')

    cloudToLocal(bucket_name, 'SRC/tab-prov.xlsx', './tab-prov.xlsx')
    # Save the file in a variable
    tab_prov = pd.read_excel('./tab-prov.xlsx')
    # Delete the file
    os.remove('./tab-prov.xlsx')

    cloudToLocal(bucket_name, 'SRC/tab-share-ccaa.xlsx', './tab-share-ccaa.xlsx')
    # Save the file in a variable
    tab_share_ccaa = pd.read_excel('./tab-share-ccaa.xlsx')
    # Delete the file
    os.remove('./tab-share-ccaa.xlsx')


    logger.info("Derma Acniben: The cloud SRC files has been downloaded, saved in a variable, and deleted")
    # Features used in the code:
    # Lists with the names of all investment channels
    list_bucket = np.append(generate_data(query_channels)['campaign_bucket_0'].tolist(), ['TV', 'CTV'])
    list_bucket_without_TV = generate_data(query_channels)['campaign_bucket_0'].tolist()
    # Other features
    weeks = generate_data(query_weeks)['week'].tolist()
    months = [f'month_{i}' for i in range(2, 13)]
    weather = ['tavg_weighted_mean', 'prcp_weighted_mean']
    months_weather = months + ['prcp_weighted_mean']
    months_licon = months + ['earned_real_acc_kpts']
    months_promo = months + ['Other_Diff_AVG_anual', 'ISDIN_Diff_AVG_anual']

    df_raw = generate_data(query_investment, bu = '"Derma"', brand = '"ACNIBEN"', index_bu='"2181"', country = '"ES"', list_bucket = tuple(list_bucket_without_TV))    # dfs for prediction beyond Sales (model is trained to calculate units sold in sales): marketis and in euros by average order value
    
    df_aov = transform_df_aov(query_aov, bu = '"Derma"')
    df_markets_week = transform_df_market(query_markets, df_raw, bu = '"Derma"')
    
    # We add 4 lines corresponding to the 4 future weeks (we do this because the interface offers these production 
    # possibilities to the user, so we already pre-calculate things like Adstock)
    df_raw_fut_weeks = adding_rows_of_future_weeks_to_forecast(df_raw, list_bucket, country_id = 'ES', currency = 'EUR', bu = 'Derma')
    df_transf, updated_list_bucket = transform_data(df_raw_fut_weeks, list_bucket, bu = 'Derma', brand = 'Acniben')
    df_adstock, list_bucket_adstock,list_just_adstock = adstock_data(df = df_transf, theta = 0.6, updated_list_bucket = updated_list_bucket)
    # data_exploration(df_transf[df_transf['future_week'] == 0], updated_list_bucket)

    df_train, weather_data_ccaa = get_weather_data_api(df = df_adstock[df_adstock['future_week'] == 0],
                                    api_key = 'AIzaSyARAcLMJsP-4R-nM8QAw0QJKWWizM5lkdI', 
                                    df_cities = tab_prov, 
                                    df_provincias = tab_ccaa, 
                                    df_share_ccaa = tab_share_ccaa,
                                    bu='Aesthetics')                  
    
    # Training and tuning of week models (separated by weeks, hence Week) and climate models
    model_months, X_months_train, X_months_test, y_months_train, y_months_test = train_model(df_train, list_bucket_adstock, 'units', months)
    
    # # Model evaluation
    df_evaluation_months = evaluation_model(model_months, X_months_test, y_months_test)
    df_coef_model_months = generating_model_coefficients(model_months, list_bucket_adstock)
    df_cross_validation_months = cross_validation_evaluation(df_train, list_bucket_adstock, 'units', months)

    df_saturation = generating_saturation_points(df_train, updated_list_bucket) # data saturation for DERMA ACNIBEN

    fig_residuals_diagnostic_months, model_residual_tests_months = model_residuals(model_months, X_months_train, y_months_train, "derma_acniben_fig_residuals_diagnostic_months.png")

    # Since we don't have weather data for future weeks (the interface doesn't allow for future predictions), we set it to 0.
    df_adstock_future = df_adstock[df_adstock['future_week'] == 1].copy()
    df_adstock_future.loc[:, 'tavg_weighted_mean'] = 0
    df_adstock_future.loc[:, 'prcp_weighted_mean'] = 0
    df_all_dates_and_weather = pd.concat([df_train, df_adstock_future], ignore_index=True)

    # final df used as input to the interface: it contains all features, all possible dates (including future ones), actual sales, predicted by the model corresponding 
    # to real investments and predicted sales for investments equal to zero (or organic sales).
    df_final = calculate_sales_with_and_without_investment(df_all_dates_and_weather, list_just_adstock, months, model_months)
    
    # with open("model_months.pkl", "wb") as file:
    #     pickle.dump(model_months, file)
    # with open("X_months_train.pkl", "wb") as file:
    #     pickle.dump(X_months_train, file)
    # with open("y_months_train.pkl", "wb") as file:
    #     pickle.dump(y_months_train, file)
    # df_coef_model_months.to_csv('df_coef_model_months')

    cloudToLocal(bucket_name, 'Current_model/derma_acniben_current_model.pkl', './derma_acniben_current_model.pkl')

    # Save the file in a variable as pickle format
    with open('./derma_acniben_current_model.pkl', 'rb') as f:
        current_model = pickle.load(f)
    
    # Delete the file
    os.remove('./derma_acniben_current_model.pkl')
    logger.info("Derma Acniben: The file derma_acniben_current_model.pkl has been downloaded, saved in a variable, and deleted")


    cloudToLocal(bucket_name, 'Current_model/derma_acniben_models_in_production.csv', './derma_acniben_models_in_production.csv')

    # Save the file in a variable
    with open('./derma_acniben_models_in_production.csv', 'r', encoding='utf-8') as f:
        models_in_production = pd.read_csv(f)
    
    # Delete the file
    os.remove('./derma_acniben_models_in_production.csv')
    logger.info("Derma Acniben: The file derma_acniben_models_in_production.csv has been downloaded, saved in a variable, and deleted")

    date_current_model = models_in_production["model_name"].iloc[-1].split("_")[-1]


    cloudToLocal(bucket_name, f'{date_current_model}/derma_acniben_X_months_train.pkl', './derma_acniben_X_months_train.pkl')

    # Save the file in a variable as pickle format
    with open('./derma_acniben_X_months_train.pkl', 'rb') as f:
        X_current_train = pickle.load(f)
    
    # Delete the file
    os.remove('./derma_acniben_X_months_train.pkl')
    logger.info("Derma: The file derma_acniben_X_months_train.pkl has been downloaded, saved in a variable, and deleted")



    cloudToLocal(bucket_name, f'{date_current_model}/derma_acniben_y_months_train.pkl', './derma_acniben_y_months_train.pkl')

    # Save the file in a variable as pickle format
    with open('./derma_acniben_y_months_train.pkl', 'rb') as f:
        y_current_train = pickle.load(f)
    
    # Delete the file
    os.remove('./derma_acniben_y_months_train.pkl')
    logger.info("Derma Acniben: The file derma_acniben_y_months_train.pkl has been downloaded, saved in a variable, and deleted")

    ### best_models.pkl
    # To select the model in production we have to compare the r2 of this week's model (trained in the previous lines) 
    # with the model that is currently in production (tested with the same data as the current week's model)

    # We must delete the constants from X, as future functions add
    # Updating X_test (This week's model testing dataset X): 
    X_months_test_without_const = X_months_test.loc[:, X_months_test.columns != 'const'] #deleting constant
    # X test dataset of the current production model:
    X_current_train = X_current_train.loc[:, X_current_train.columns != 'const'] #deleting constant

    # Compare and remove lines from X_months_test that are also in X_current_train, as we have to ensure that the model is not being tested with training data.
    X_months_test_unique = pd.merge(X_months_test_without_const, X_current_train, how='left', indicator=True)
    X_months_test_unique = X_months_test_unique[X_months_test_unique['_merge'] == 'left_only'].drop_duplicates(subset=X_months_test.columns).drop(columns='_merge')
    # Getting the indices of the unique data
    indices_unique = X_months_test_unique.index
    # Filtering y_test with the same indices
    y_months_test_unique = y_months_test[indices_unique]

    # We re-evaluated the current production model, using the most current test X:
    df_evaluation_current_old_model = evaluation_model(current_model, X_months_test_unique, y_months_test_unique)

    # Function to select the new model in production (the one that presents the highest r2 with current data)
    current_model, current_evaluation, model_name,start_date, end_date, X_current_train, y_current_train = selection_of_model_to_be_served(model_months, current_model, df_evaluation_months, df_evaluation_current_old_model, models_in_production, X_months_train, y_months_train, X_current_train, y_current_train)

    # Add information about the model in production to a table
    new_row = pd.DataFrame({'start_date': [start_date ], 'end_date': [end_date], 'model_name': [model_name]})
    if models_in_production['end_date'].iloc[-1] != new_row['end_date'].iloc[-1]:
        models_in_production = pd.concat([models_in_production, new_row], ignore_index=True)

    # Creating the table shown in the interface: channel investments, sales_without_investment, Real, Predict (for sales and market)
    predict_table_months = create_predict_table(X_months_train, current_model, df_final, updated_list_bucket, months, df_markets_week)

    # Generate current files (with NEW CURRENT MODEL - WITH THE WINNING MODEL)
    df_final_current = calculate_sales_with_and_without_investment(df_all_dates_and_weather, list_just_adstock, months, current_model)
    predict_table_current = create_predict_table(X_current_train, current_model, df_final_current, updated_list_bucket, months, df_markets_week)
    fig_residuals_diagnostic_current, model_residual_tests_current = model_residuals(current_model, X_current_train, y_current_train, "derma_acniben_fig_residuals_diagnostic_current.png")
    df_coef_model_current = generating_model_coefficients(current_model, list_bucket_adstock)
    derma_df_year = create_table_predict_units(df_final_current, df_coef_model_current, updated_list_bucket, 'year')
    derma_df_month = create_table_predict_units(df_final_current, df_coef_model_current, updated_list_bucket, 'month')
    derma_df_week = create_table_predict_units(df_final_current, df_coef_model_current, updated_list_bucket, 'week')

    # Upload files to cloud storage
    filesToCloud('derma_acniben_model_months.pkl', pickle.dumps(model_months), 'application/octet-stream', bucket_name, current_date_format)
    filesToCloud('derma_acniben_evaluation_months.csv', df_evaluation_months.to_csv(index=False), 'text/csv', bucket_name, current_date_format)
    filesToCloud('derma_acniben_coefficients_month.csv', df_coef_model_months.to_csv(index=False), 'text/csv', bucket_name, current_date_format)
    filesToCloud('derma_acniben_saturation.csv', df_saturation.to_csv(index=False), 'text/csv', bucket_name, current_date_format)
    
    filesToCloud('derma_acniben_df.csv', df_final.to_csv(index=False), 'text/csv', bucket_name, current_date_format)
    
    filesToCloud('derma_acniben_current_model.pkl', pickle.dumps(current_model), 'application/octet-stream', bucket_name, 'Current_model')
    filesToCloud('derma_acniben_evaluation_current_model.csv', current_evaluation.to_csv(index=False), 'text/csv', bucket_name, 'Current_model')
    filesToCloud('derma_acniben_models_in_production.csv', models_in_production.to_csv(index=False), 'text/csv', bucket_name, 'Current_model')
    
    filesToCloud('derma_acniben_predict_table_months.csv', predict_table_months.to_csv(index=False), 'text/csv', bucket_name, current_date_format)

    filesToCloud('derma_acniben_model_residual_tests_months.pkl', pickle.dumps(model_residual_tests_months), 'application/octet-stream', bucket_name, current_date_format)
    filesToCloud('derma_acniben_weather_data_ccaa.csv', weather_data_ccaa.to_csv(index=False), 'text/csv', bucket_name, current_date_format)
    
    filesToCloud('derma_acniben_fig_residuals_diagnostic_months.png', './derma_acniben_fig_residuals_diagnostic_months.png', 'image/png', bucket_name, current_date_format)
    os.remove('./derma_acniben_fig_residuals_diagnostic_months.png')
    filesToCloud('derma_acniben_fig_residuals_diagnostic_current.png', './derma_acniben_fig_residuals_diagnostic_current.png', 'image/png', bucket_name, 'Current_model')
    os.remove('./derma_acniben_fig_residuals_diagnostic_current.png')

    filesToCloud('derma_acniben_X_months_train.pkl', pickle.dumps(X_months_train), 'application/octet-stream', bucket_name, current_date_format)
    filesToCloud('derma_acniben_y_months_train.pkl', pickle.dumps(y_months_train), 'application/octet-stream', bucket_name, current_date_format)
    
    filesToCloud('derma_acniben_df_final_current.csv', df_final_current.to_csv(index=False), 'text/csv', bucket_name, 'Current_model')
    filesToCloud('derma_acniben_predict_table_current.csv', predict_table_current.to_csv(index=False), 'text/csv', bucket_name, 'Current_model')
    
    filesToCloud('derma_acniben_model_residual_tests_current.pkl', pickle.dumps(model_residual_tests_current), 'application/octet-stream', bucket_name, 'Current_model')

    filesToCloud('derma_acniben_df_coef_model_current.csv', df_coef_model_current.to_csv(index=False), 'text/csv', bucket_name, 'Current_model')

    filesToCloud('derma_acniben_df_aov.csv', df_aov.to_csv(index=False), 'text/csv', bucket_name, current_date_format)
    filesToCloud('derma_acniben_df_markets_week.csv', df_markets_week.to_csv(index=False), 'text/csv', bucket_name, current_date_format)

    filesToCloud('derma_df_year.csv', derma_df_year.to_csv(index=False), 'text/csv', bucket_name, 'Current_model')
    filesToCloud('derma_df_month.csv', derma_df_month.to_csv(index=False), 'text/csv', bucket_name, 'Current_model')
    filesToCloud('derma_df_week.csv', derma_df_week.to_csv(index=False), 'text/csv', bucket_name, 'Current_model')

    logger.info("Derma Acniben: The files have been uploaded to Cloud Storage")

    logger.info("Derma Acniben: End of process")


try:
    # Load google variables/secrets
    load_dotenv()
    email_pass_b = os.environ["SENDER_EMAIL_PASS"]
    bucket_name = os.environ["CLOUD_STORAGE_BUCKET"]
    email_receiver = os.environ["EMAIL_RECEIVER"]

    # Load the logging system
    logger = MyLogger(os.path.basename(__file__))
    logger.info("Start of process")

    # Run
    main_foto()
    main_ceutics()
    main_derma_acniben()

except Exception as e:
    error=f"""error:{str(e)}"""
    subject="Backend MMM ISDIN"
    body=f"Ha ocurrido un error:\n{error}"#.format(error)
    logger.error(f"""Ha ocurrido un error durante el proceso: {str(e)}""")
    send_email(email_pass_b, subject, body, email_receiver)
    print(error)

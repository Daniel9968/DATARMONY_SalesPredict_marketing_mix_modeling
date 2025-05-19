import os
from dotenv import load_dotenv
from datetime import datetime, timedelta
from load_model import foto_df_final, foto_df_final_weather, foto_saturation, foto_current_model, foto_model_weather, foto_evaluation_current_model, foto_evaluation_weather, foto_predict_table_weeks, foto_predict_table_weather, foto_df_aov, foto_df_markets_week, ceutics_df_final, ceutics_current_model, ceutics_evaluation_current_model, ceutics_saturation, ceutics_predict_table, ceutics_df_aov, ceutics_df_markets_week, ceutics_fig_best_tree, derma_acniben_df_final, derma_acniben_current_model, derma_acniben_evaluation_current_model, derma_acniben_saturation, derma_acniben_predict_table_weeks, derma_acniben_df_aov, derma_acniben_df_markets_week, foto_df_year, foto_df_month, foto_df_week, ceutics_df_year, ceutics_df_month, ceutics_df_week, derma_df_year, derma_df_month, derma_df_week
import statsmodels.api as sm
import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from sklearn.ensemble import RandomForestRegressor


load_dotenv()
# Constants and control variables
PERIOD_VARIABLE = st.session_state.get('period_variable', '24')

foto_investment_columns_to_plot = [col for col in foto_df_final.columns if '-adstock' in col] # = channels
ceutics_investment_columns_to_plot = [col for col in ceutics_df_final.columns if '-adstock' in col] # = channels
derma_acniben_investment_columns_to_plot = [col for col in derma_acniben_df_final.columns if '-adstock' in col] # = channels
sales_columns_to_plot = ['sales_without_investment', 'investment_sales']

foto_channels = foto_investment_columns_to_plot
ceutics_channels = ceutics_investment_columns_to_plot
derma_acniben_channels = derma_acniben_investment_columns_to_plot

foto_channels_raw = [col.replace('-adstock', '') for col in foto_channels]
ceutics_channels_raw = [col.replace('-adstock', '') for col in ceutics_channels]
derma_acniben_channels_raw = [col.replace('-adstock', '') for col in derma_acniben_channels]
foto_channels_raw_field = [f'inversion_field_{i}' for i in range(len(foto_channels_raw))]
ceutics_channels_raw_field = [f'inversion_field_{i}' for i in range(len(ceutics_channels_raw))]
derma_acniben_channels_raw_field = [f'inversion_field_{i}' for i in range(len(derma_acniben_channels_raw))]

foto_list_just_adstock = ['just_adstock_' + channel for channel in foto_channels_raw]
ceutics_list_just_adstock = ['just_adstock_' + channel for channel in ceutics_channels_raw]
derma_acniben_list_just_adstock = ['just_adstock_' + channel for channel in derma_acniben_channels_raw]

week_columns = [col for col in foto_df_final.columns if col.startswith('week_')]
month_columns = [col for col in foto_df_final.columns if col.startswith('month_')]
features_with_range = ['year'] + foto_channels + month_columns
features_with_range_weather = ['year'] + foto_channels + month_columns + ['prcp_weighted_mean']
features_ceutics = ceutics_current_model.feature_names_in_
features_derma_acniben = ['year'] + derma_acniben_channels + month_columns
range_of_features =  ['year'] + month_columns

investments=["Sales","Markets"]

def get_influencia_variables():
    """
    Retrieves coefficients of influence variables from the current model.

    Args:
        current_model (statsmodels.regression.linear_model.RegressionResultsWrapper):
            Current regression model containing coefficients.
        channels (list): List of channels to retrieve coefficients for.

    Returns:
        pandas.DataFrame: DataFrame containing channels and their scaled coefficients.
    """
    # Create a DataFrame with coefficients for specified channels, sorted in ascending order
    if st.session_state.model == foto_model_weather or st.session_state.model == foto_current_model:
        coeficientes = pd.DataFrame(st.session_state.model.params[foto_channels].items(), columns=['channel', 'coefficient']).sort_values(by= 'coefficient',ascending=True)
    if st.session_state.model == ceutics_current_model:
        importancias_caracteristicas = st.session_state.model.feature_importances_
        # Asociar importancias con los nombres de las características
        coeficientes = pd.DataFrame({
            'channel': features_ceutics,
            'coefficient': importancias_caracteristicas
        })
        coeficientes = coeficientes[coeficientes['channel'].isin(ceutics_channels)].sort_values(by='coefficient', ascending=False)
    if st.session_state.model == derma_acniben_current_model:
        coeficientes = pd.DataFrame(st.session_state.model.params[derma_acniben_channels].items(), columns=['channel', 'coefficient']).sort_values(by= 'coefficient',ascending=True)
    # Define scale limits (0 to 10)
    min_scale = 0
    max_scale = 10
    coeficientes["coefficient"] = coeficientes["coefficient"].apply(lambda x: max(x, 0))
    # Apply min-max scaling
    coeficientes['coeficientes_escalonadas'] = min_scale + ((coeficientes['coefficient'] - coeficientes['coefficient'].min()) / (coeficientes['coefficient'].max() - coeficientes['coefficient'].min())) * (max_scale - min_scale)
    return coeficientes



def get_shareSpend_shareEffect(df):
    """
    Calculates the share of spent and share of effect for each channel based on the provided DataFrame.

    Args:
        df (pandas.DataFrame): DataFrame containing columns 'channel', 'Inversión adstock', and 'Inversión semanal'.

    Returns:
        pandas.DataFrame: DataFrame with columns 'channel', 'share of spent', and 'share_of_effect'.
    """

    df_aux = df.copy()

    df_aux.columns = ['channel', 'Inversión adstock', 'Inversión semanal']
    df_aux = df_aux[['channel', 'Inversión semanal']]
    df_aux['channel'] = df_aux['channel'] + "-adstock"
    df_aux['Inversión semanal'] = df_aux['Inversión semanal'].astype(float)
    share_of_spent = df_aux[['channel', 'Inversión semanal']].copy()
    share_of_spent['share of spent'] = ((share_of_spent['Inversión semanal'] / (share_of_spent['Inversión semanal'].sum())) * 100).round(2)
    share_of_spent = share_of_spent[['channel', 'share of spent']]

    #### SHARE OF EFFECT ####
    # Obtaining channel coefficients
    # Note: 'current_model' and 'channels' variables are assumed to be defined elsewhere in the script
    if st.session_state.model == foto_model_weather or st.session_state.model == foto_current_model:
        share_of_effect = pd.DataFrame(st.session_state.model.params[foto_channels].items(), columns=['channel', 'coefficient'])
    if st.session_state.model == derma_acniben_current_model:
        share_of_effect = pd.DataFrame(st.session_state.model.params[derma_acniben_channels].items(), columns=['channel', 'coefficient'])

    share_of_effect['coefficient'] = share_of_effect['coefficient'].apply(lambda x: max(x, 0))
    # Scaling process (from 0 to 1, avoiding negative values)
    valor_minimo = share_of_effect['coefficient'].min()
    valor_maximo = share_of_effect['coefficient'].max()
    share_of_effect['scaled_values'] = (share_of_effect['coefficient'] - valor_minimo) / (valor_maximo - valor_minimo)

    # Calculate share of effect for each channel
    share_of_effect['share_of_effect'] = (share_of_effect['scaled_values'] * 100) / share_of_effect['scaled_values'].sum()
    share_of_effect = share_of_effect[['channel', 'share_of_effect']]

    #### MERGE INFORMATION ####
    # Merge share of spent and share of effect DataFrames
    df_share_of = share_of_spent.merge(share_of_effect, on='channel', how='inner')

    # Fill NaN values with 0 (if any)
    df_share_of['share of spent'] = df_share_of['share of spent'].fillna(0)

    df_share_of['channel'] = df_share_of['channel'].str.rstrip('-adstock')

    return df_share_of

def ratings_to_stars(ratings):
    """
    Converts a list of ratings into star ratings with Unicode characters.

    Args:
        ratings (list of int): List of ratings where each rating is an integer between 0 and 5.

    Returns:
        list of str: List of star ratings represented as strings with Unicode characters.
    """
    with_color = "✩"
    no_color = "⭑"
    results = []
    for rating in ratings:
        result_now = ""
        for i in range(5):
            if i < rating:
                result_now += with_color + " "
            else:
                result_now += no_color + " "
        results.append(result_now)

    return results


def calculate_inversionAdstock(adstocks):
    """
    Calculates the total investment including adstock for each item in the list.

    Args:
        adstocks (list of str): List of strings where each string represents the current adstock value.

    Returns:
        list of str: List of strings where each string represents the calculated total investment.
    """
    results = []
    for i in range(len(adstocks)):
        try:
            inversion_semanal = st.session_state["inversion_field_" + str(i)]
        except:
            inversion_semanal = "0,00"

        adstock_value = adstocks[i].replace(",", ".")

        try:
            if adstock_value and inversion_semanal:
                result = round(float(adstock_value) + float(inversion_semanal.replace(",", ".")), 2)
                results.append(str(result).replace(".", ","))
            else:
                st.session_state["inversion_field_" + str(i)] = "0,00"
                results.append(adstocks[i].replace(".", ","))
        except ValueError:
            st.session_state["inversion_field_" + str(i)] = "0,00" 
            results.append(adstocks[i].replace(".", ","))

    return results





def simulacion_initial(df):
    """
    Initializes a simulation table based on selected period and coefficients from the model.

    Args:
        df (DataFrame): DataFrame containing necessary data for simulation.

    Returns:
        DataFrame: Simulation table containing channels, star ratings, adstock values,
                   historical maximum spend, foto_saturation, weekly investment including adstock,
                   and total investment.
    """
    # Start building the simulator table based on the model coefficient parameter:
    ## With that, let's calculate the star level.
    # Simulator input data
    if st.session_state.model == foto_model_weather or st.session_state.model == foto_current_model:
        weekly_investment = {channel: 0.00 for channel in foto_channels_raw}
    if st.session_state.model == ceutics_current_model:
        weekly_investment = {channel: 0.00 for channel in ceutics_channels_raw}
    if st.session_state.model == derma_acniben_current_model:
        weekly_investment = {channel: 0.00 for channel in derma_acniben_channels_raw}


    period = {'year' : st.session_state.year,'week' : st.session_state.week}
    inversion_semanal = {**period, **weekly_investment}
    
    if st.session_state.model == foto_model_weather or st.session_state.model == foto_current_model:
        # Necessary definition for this session
        estrellas = pd.DataFrame(st.session_state.model.params[foto_channels].items(), columns=['channel', 'coefficient'])

    if st.session_state.model == ceutics_current_model:
        importancias_caracteristicas = st.session_state.model.feature_importances_
        # Asociar importancias con los nombres de las características
        estrellas = pd.DataFrame({
            'channel': features_ceutics,
            'coefficient': importancias_caracteristicas
        })
        estrellas = estrellas[estrellas['channel'].isin(ceutics_channels)].sort_values(by='coefficient', ascending=False)
    if st.session_state.model == derma_acniben_current_model:
        estrellas = pd.DataFrame(st.session_state.model.params[derma_acniben_channels].items(), columns=['channel', 'coefficient'])
    # Define star value considering that the maximum coefficient found corresponds to 5 stars
    estrellas['estrellas'] = (estrellas['coefficient']*5)/estrellas['coefficient'].max()

    # Define the scale limits (0 to 5)
    min_scale = 0
    max_scale = 5

    # Apply min-max scaling
    estrellas['estrellas'] = min_scale + ((estrellas['estrellas'] - estrellas['estrellas'].min()) / (estrellas['estrellas'].max() - estrellas['estrellas'].min())) * (max_scale - min_scale)

    ################### Creating the simulator df: ##################################
    tabla_simulador = estrellas[['channel', 'estrellas']].copy()
    
    #### Adding Adstock column ####    
    simulador_adstock = df.loc[(df['year'] == inversion_semanal['year']) & (df['week'] == inversion_semanal['week'])]
    if st.session_state.model == foto_model_weather or st.session_state.model == foto_current_model:
        simulador_adstock = simulador_adstock[foto_list_just_adstock].transpose()
    if st.session_state.model == ceutics_current_model:
        simulador_adstock = simulador_adstock[ceutics_list_just_adstock].transpose()
    if st.session_state.model == derma_acniben_current_model:
        simulador_adstock = simulador_adstock[derma_acniben_list_just_adstock].transpose()

    simulador_adstock = simulador_adstock.reset_index()
    simulador_adstock = simulador_adstock.iloc[:, :2]
    simulador_adstock.columns = ["channel", "Adstock [euros]"]
    simulador_adstock['channel'] = simulador_adstock['channel'].str.replace('just_adstock_', '') + '-adstock'

    tabla_simulador = tabla_simulador.merge(simulador_adstock, on='channel', how='inner')

    #### Adding Saturation column ####
    # Gathering information from the saturation curve dictionary
    if st.session_state.model == foto_model_weather or st.session_state.model == foto_current_model:
        simulador_saturacion = foto_saturation[['Channel', 'Max_Historical_Spend', 'Saturation']]
    if st.session_state.model == ceutics_current_model:
        simulador_saturacion = ceutics_saturation[['Channel', 'Max_Historical_Spend', 'Saturation']]
    if st.session_state.model == derma_acniben_current_model:
        simulador_saturacion = derma_acniben_saturation[['Channel', 'Max_Historical_Spend', 'Saturation']]

    simulador_saturacion['Channel'] = simulador_saturacion['Channel'] + '-adstock'

    #### Adding Maximum Historical column ####
    simulador_saturacion.columns = ['channel', 'Máximo histórico [euros]', 'Saturación']
    tabla_simulador = tabla_simulador.merge(simulador_saturacion, on='channel', how='inner')

    simulador_inversion = pd.DataFrame.from_dict([inversion_semanal])
    simulador_inversion = simulador_inversion.drop(columns=['year', 'week']).T.reset_index()
    simulador_inversion .rename(columns={'index': 'channel', 0: 'Inversión semanal'}, inplace=True)
    simulador_inversion['channel'] = simulador_inversion['channel'].apply(lambda x: x + '-adstock')

    tabla_simulador_con_inversion = tabla_simulador.merge(simulador_inversion, on='channel', how='inner')

    #### Adding Total column #####
    tabla_simulador_con_inversion['Inversión + Adstock'] = tabla_simulador_con_inversion['Adstock [euros]'] + tabla_simulador_con_inversion['Inversión semanal']

    return tabla_simulador_con_inversion

def initial_df(df):
    """
    Generates initial DataFrame based on selected period and features with ranges.

    Args:
        df (DataFrame): DataFrame containing relevant data.

    Returns:
        tuple: Tuple containing original DataFrame and a dictionary with confidence intervals.

    """
    ###### Add new dates: ##############################

    df_aux = df.copy()

    filtered_row = df_aux[(df_aux['year'] == st.session_state.year) & (df_aux['week'] == st.session_state.week)]

    if st.session_state.model == foto_current_model:
        features = features_with_range
    if st.session_state.model == foto_model_weather:
        features = features_with_range_weather
    if st.session_state.model == ceutics_current_model:
        features = features_ceutics
    if st.session_state.model == derma_acniben_current_model:
        features = features_derma_acniben

    if st.session_state.model == foto_model_weather or st.session_state.model == foto_current_model or st.session_state.model == derma_acniben_current_model:
        selected_features = filtered_row[features].copy()
        # bool -> INT
        columns_to_convert = features

        selected_features[columns_to_convert] = selected_features[columns_to_convert].astype(int)
        # Add a constant
        selected_features = sm.add_constant(selected_features, has_constant='add')

        eighty_perc = st.session_state.model.get_prediction(exog=selected_features).conf_int(alpha=0.2)
        ninetyFive_perc = st.session_state.model.get_prediction(exog=selected_features).conf_int(alpha=0.05)  
          
    if st.session_state.model == ceutics_current_model:
        selected_features = filtered_row[features].copy()
        # Obtenemos las predicciones individuales de cada árbol
        # Esta opción está disponible en `scikit-learn` con `estimators_`
        all_tree_predictions = np.array([tree.predict(selected_features) for tree in st.session_state.model.estimators_])

        # Cálculo de la media y de los intervalos de confianza del 95%
        lower_bound_95 = np.percentile(all_tree_predictions, 2.5, axis=0)
        upper_bound_95 = np.percentile(all_tree_predictions, 97.5, axis=0)
        ninetyFive_perc = [[lower_bound_95[0], upper_bound_95[0]],[lower_bound_95[0], upper_bound_95[0]]]
        lower_bound_80 = np.percentile(all_tree_predictions, 10, axis=0)
        upper_bound_80 = np.percentile(all_tree_predictions, 90, axis=0)
        eighty_perc = [[lower_bound_80[0], upper_bound_80[0]],[lower_bound_80[0], upper_bound_80[0]]]


    if st.session_state.investment == "Sales":
        if st.session_state.model == foto_model_weather or st.session_state.model == foto_current_model:
            df_aux = df_aux.merge(foto_df_aov[['year', 'week', 'aov']], on=['year', 'week'], how='left')
        if st.session_state.model == ceutics_current_model:
            df_aux = df_aux.merge(ceutics_df_aov[['year', 'week', 'aov']], on=['year', 'week'], how='left')
        if st.session_state.model == derma_acniben_current_model:
            df_aux = df_aux.merge(derma_acniben_df_aov[['year', 'week', 'aov']], on=['year', 'week'], how='left')


        df_aux['aov'] = df_aux['aov'].fillna(0)
        df_aux['investment_sales_aov'] = df_aux['investment_sales'] * df_aux['aov']
        df_aux['sales_without_investment_aov'] = df_aux['sales_without_investment'] * df_aux['aov']

    df_aux2 = df_aux

    if st.session_state.investment == "Markets":
        # Sales or markets button: when it is 'sales' (by default), only the top 3 lines, and if it is 'markets', the content immediately below
        if st.session_state.model == foto_model_weather or st.session_state.model == foto_current_model:
            df_aux = df_aux.merge(foto_df_markets_week[['year', 'week', 'diff_markets', 'aov_markets']], on=['year', 'week'], how='left')
        if st.session_state.model == ceutics_current_model:
            df_aux = df_aux.merge(ceutics_df_markets_week[['year', 'week', 'diff_markets', 'aov_markets']], on=['year', 'week'], how='left')
        if st.session_state.model == derma_acniben_current_model:
            df_aux = df_aux.merge(derma_acniben_df_markets_week[['year', 'week', 'diff_markets', 'aov_markets']], on=['year', 'week'], how='left')

        df_aux['diff_markets'] = df_aux['diff_markets'].fillna(0)
        df_aux['aov_markets'] = df_aux['aov_markets'].fillna(0)
        df_aux['investment_sales'] = df_aux['investment_sales'] * df_aux['diff_markets']
        df_aux['investment_sales_aov'] = df_aux['investment_sales'] * df_aux['aov_markets']
        df_aux['sales_without_investment'] = df_aux['sales_without_investment'] * df_aux['diff_markets']
        df_aux['sales_without_investment_aov'] = df_aux['sales_without_investment'] * df_aux['aov_markets']

    return (df_aux, {"80_perc": eighty_perc, "95_perc": ninetyFive_perc}, df_aux2)

def after_sim_df(df):
    """
    Generates a simulated DataFrame based on features with ranges.

    Args:
        df (DataFrame): Original DataFrame containing relevant data.

    Returns:
        tuple: Tuple containing simulated DataFrame (`df_ds`) and a dictionary with confidence intervals (`{"80_perc": eighty_perc, "95_perc": ninetyFive_perc}`).

    """
    # Start building the simulator table based on the model coefficients parameter:
    ## With that, let's calculate the star level.
    # Simulator input data

    weekly_investment = st.session_state["inversions_values"]
    period = {'year' : st.session_state.year,'week' : st.session_state.week}
    inversion_semanal = {**period, **weekly_investment}

    # Create df_ds based on df_e0:
    df_ds = df.copy()


    # Replace investment values in df_ds:
    if st.session_state.model == foto_model_weather or st.session_state.model == foto_current_model:
        df_ds.loc[(df_ds['year'] == inversion_semanal['year']) & (df_ds['week'] == inversion_semanal['week']), foto_channels_raw] = [inversion_semanal[column] for column in foto_channels_raw_field]
    if st.session_state.model == ceutics_current_model:
        df_ds.loc[(df_ds['year'] == inversion_semanal['year']) & (df_ds['week'] == inversion_semanal['week']), ceutics_channels_raw] = [inversion_semanal[column] for column in ceutics_channels_raw_field]
    if st.session_state.model == derma_acniben_current_model:
        df_ds.loc[(df_ds['year'] == inversion_semanal['year']) & (df_ds['week'] == inversion_semanal['week']), derma_acniben_channels_raw] = [inversion_semanal[column] for column in derma_acniben_channels_raw_field]

    # Add investment and Adstock:
    if 'year' in df_ds.columns and 'week' in df_ds.columns:
        mask = (df_ds['year'] == inversion_semanal['year']) & (df_ds['week'] == inversion_semanal['week'])
        if st.session_state.model == foto_model_weather or st.session_state.model == foto_current_model:
            original_columns = foto_channels_raw
        if st.session_state.model == ceutics_current_model:
            original_columns = ceutics_channels_raw
        if st.session_state.model == derma_acniben_current_model:
            original_columns = derma_acniben_channels_raw

        adstock_columns = [f'{column}-adstock' for column in original_columns]
        just_adstock_columns = [f'just_adstock_{column}' for column in original_columns]

        for original_col, adstock_col, just_adstock_col in zip(original_columns, adstock_columns, just_adstock_columns):
            df_ds.loc[mask, adstock_col] = df_ds.loc[mask, original_col] + df_ds.loc[mask, just_adstock_col]

    ######## Simulation ################## 

    if st.session_state.model == foto_current_model:
        features = features_with_range
    if st.session_state.model == foto_model_weather:
        features = features_with_range_weather
    if st.session_state.model == ceutics_current_model:
        features = features_ceutics
    if st.session_state.model == derma_acniben_current_model:
        features = features_derma_acniben

    
    X_simulacion = df_ds.loc[(df_ds['year'] == inversion_semanal['year']) & (df_ds['week'] == inversion_semanal['week'])][features]
    # Convert from bool to int
    if st.session_state.model == foto_model_weather or st.session_state.model == foto_current_model or st.session_state.model == derma_acniben_current_model:
        columns_to_convert = range_of_features

        X_simulacion[columns_to_convert] = X_simulacion[columns_to_convert].astype(int)

        # Adding the constant
        X_simulacion = sm.add_constant(X_simulacion, has_constant='add')
        # Perform prediction with user-provided information
        y_pred = st.session_state.model.predict(X_simulacion).values[0] # Result (y_pred) = units sold

    if st.session_state.model == ceutics_current_model:
        feature_order = ceutics_current_model.feature_names_in_
        X_simulacion = X_simulacion[feature_order]
        # Perform prediction with user-provided information
        y_pred = st.session_state.model.predict(X_simulacion)[0] 

    # Replace value in df_ds:
    if 'year' in df_ds.columns and 'week' in df_ds.columns:
        df_ds.loc[mask, 'investment_sales'] = y_pred

    # Adjust values with negative uplift (organic sales > sales with investment)
    # Check and adjust with sales_without_investment if needed
    if df_ds.loc[mask, 'investment_sales'].values[0] < df_ds.loc[mask, 'sales_without_investment'].values[0]:
        df_ds.loc[mask, 'investment_sales'] = df_ds.loc[mask, 'sales_without_investment'].values[0]
        X_simulacion = df.loc[(df['year'] == inversion_semanal['year']) & (df['week'] == inversion_semanal['week'])][features]

            # Convert from bool to int
        if st.session_state.model == foto_model_weather or st.session_state.model == foto_current_model or st.session_state.model == derma_acniben_current_model:
            columns_to_convert = range_of_features

            X_simulacion[columns_to_convert] = X_simulacion[columns_to_convert].astype(int)

            # Adding the constant
            X_simulacion = sm.add_constant(X_simulacion, has_constant='add')
        if st.session_state.model == ceutics_current_model:
            feature_order = ceutics_current_model.feature_names_in_
            X_simulacion = X_simulacion[feature_order]

    # Check and adjust with DataFrame 'df' if needed
    if df_ds.loc[mask, 'investment_sales'].values[0] < df.loc[mask, 'investment_sales'].values[0]:
        df_ds.loc[mask, 'investment_sales'] = df.loc[mask, 'investment_sales'].values[0]
        X_simulacion = df.loc[(df['year'] == inversion_semanal['year']) & (df['week'] == inversion_semanal['week'])][features]

        # Convert from bool to int
        if st.session_state.model == foto_model_weather or st.session_state.model == foto_current_model or st.session_state.model == derma_acniben_current_model:

            columns_to_convert = range_of_features

            X_simulacion[columns_to_convert] = X_simulacion[columns_to_convert].astype(int)

            # Adding the constant
            X_simulacion = sm.add_constant(X_simulacion, has_constant='add')
        if st.session_state.model == ceutics_current_model:
            feature_order = ceutics_current_model.feature_names_in_
            X_simulacion = X_simulacion[feature_order]


    if st.session_state.model == foto_model_weather or st.session_state.model == foto_current_model or st.session_state.model == derma_acniben_current_model:
        eighty_perc = st.session_state.model.get_prediction(exog=X_simulacion).conf_int(alpha=0.2)
        ninetyFive_perc = st.session_state.model.get_prediction(exog=X_simulacion).conf_int(alpha=0.05)

    if st.session_state.model == ceutics_current_model:
        # We obtain the individual predictions from each tree
        # This option is available in `scikit-learn` with `estimators_`
        all_tree_predictions = np.array([tree.predict(X_simulacion) for tree in st.session_state.model.estimators_])

        # Calculation of the mean and 95% confidence intervals
        lower_bound_95 = np.percentile(all_tree_predictions, 2.5, axis=0)
        upper_bound_95 = np.percentile(all_tree_predictions, 97.5, axis=0)
        ninetyFive_perc = [[lower_bound_95[0], upper_bound_95[0]],[lower_bound_95[0], upper_bound_95[0]]]
        lower_bound_80 = np.percentile(all_tree_predictions, 10, axis=0)
        upper_bound_80 = np.percentile(all_tree_predictions, 90, axis=0)
        eighty_perc = [[lower_bound_80[0], upper_bound_80[0]],[lower_bound_80[0], upper_bound_80[0]]]

    if 'year' in df_ds.columns and 'week' in df_ds.columns:
        y_pred = df_ds.loc[mask, 'investment_sales'] 

    df_ds['week_str'] = df_ds['week'].astype(str).str.zfill(2)

    # Create a new combined column of year and iso and convert it to a date format
    df_ds['year_numweek'] = pd.to_datetime(df_ds['year'].astype(str) + df_ds['week_str'] + '-1', format='%G%V-%u')

    # Convert the selected week to a date format
    selected_year_numweek = pd.to_datetime(str(st.session_state.year) + str(st.session_state.week) + '-1', format='%G%V-%u')

    # Filter the DataFrame to keep only the weeks that are less than or equal to the selected week
    df_ds = df_ds[df_ds['year_numweek'] <= selected_year_numweek]

    # Remove the temporary column 'iso_date'
    df_ds = df_ds.drop(columns=['year_numweek'])

    if st.session_state.investment == "Sales":
        df_ds['aov'] = df_ds['aov'].fillna(0)
        df_ds['investment_sales_aov'] = df_ds['investment_sales'] * df_ds['aov']
        df_ds['sales_without_investment_aov'] = df_ds['sales_without_investment'] * df_ds['aov']

    if st.session_state.investment == "Markets":
        # Sales or markets button: when it is 'sales' (by default), only the top 3 lines, and if it is 'markets', the content immediately below
        if st.session_state.model == foto_model_weather or st.session_state.model == foto_current_model:
            df_ds = df_ds.merge(foto_df_markets_week[['year', 'week', 'diff_markets', 'aov_markets']], on=['year', 'week'], how='left')
        if st.session_state.model == ceutics_current_model:
            df_ds = df_ds.merge(ceutics_df_markets_week[['year', 'week', 'diff_markets', 'aov_markets']], on=['year', 'week'], how='left')
        if st.session_state.model == derma_acniben_current_model:
            df_ds = df_ds.merge(derma_acniben_df_markets_week[['year', 'week', 'diff_markets', 'aov_markets']], on=['year', 'week'], how='left')
        
        df_ds['diff_markets'] = df_ds['diff_markets'].fillna(0)
        df_ds['aov_markets'] = df_ds['aov_markets'].fillna(0)
        df_ds['investment_sales'] = df_ds['investment_sales'] * df_ds['diff_markets']
        df_ds['investment_sales_aov'] = df_ds['investment_sales'] * df_ds['aov_markets']
        df_ds['sales_without_investment'] = df_ds['sales_without_investment'] * df_ds['diff_markets']
        df_ds['sales_without_investment_aov'] = df_ds['sales_without_investment'] * df_ds['aov_markets']
    
    return (df_ds, {"80_perc": eighty_perc, "95_perc": ninetyFive_perc})

def time_series(df_timeseries, period, columns_to_melt):
    """
    Creates a time series DataFrame from the input DataFrame, transforming 'year' and 'week' columns into a 'date' column.

    Args:
        df_timeseries (DataFrame): Input DataFrame containing time series data with 'year' and 'week' columns.
        period (int): Period length in weeks to filter the time series data.
        columns_to_melt (list): List of columns to melt into 'tipo' (type) and 'cantidad' (quantity).

    Returns:
        DataFrame: Transformed time series DataFrame with 'date', 'tipo', and 'cantidad' columns.

    """
    # Create a date column for the time series
    series_temporales = df_timeseries.copy()
    
    # Convert 'year' and 'week' columns to strings
    series_temporales['year'] = series_temporales['year'].astype(str)
    series_temporales['week'] = series_temporales['week'].astype(str)

    # Combine 'year' and 'week' columns into a single column
    series_temporales['date'] = series_temporales['year'] + '-' + series_temporales['week'] + '-1'

    # Convert string to datetime
    series_temporales['date'] = series_temporales['date'].apply(lambda x: datetime.strptime(x, "%Y-%W-%w"))

    # Extract date component
    series_temporales["date"] = series_temporales["date"].dt.date

    # Calculate maximum date based on period
    max_date = series_temporales['date'].max()
    max_date = str(max_date - timedelta(weeks=int(period)))
    
    # Filter series_temporales based on maximum date
    series_temporales = series_temporales.loc[series_temporales['date'] >= datetime.strptime(max_date, "%Y-%m-%d").date()]

    # Melt columns_to_melt into 'tipo' (type) and 'cantidad' (quantity)
    series_temporales = series_temporales[columns_to_melt].melt(id_vars=["date"], 
        var_name="tipo", 
        value_name="cantidad")
    
    if period == "4":
        series_temporales['date'] = series_temporales['date'].astype(str)
    else:
        # Convert the 'date' column to datetime type
        series_temporales['date'] = pd.to_datetime(series_temporales['date'])

        # Add a new column 'month_year' to group by month
        series_temporales['date'] = series_temporales['date'].dt.to_period('M')

        # Group by 'month_year' and 'tipo', summing 'cantidad'
        series_temporales = series_temporales.groupby(['date', 'tipo'], as_index=False)['cantidad'].sum()
        
        series_temporales['date'] = series_temporales['date'].astype(str)
    
    return series_temporales


def generate_week_options(df):
    """
    Generates a list of week options. The direction (forward or backward) depends on the model.

    Returns:
        list: List of week options formatted as strings ('DD/MM/YYYY - DD/MM/YYYY - Semana WW').
    """
    # Find the maximum year and week in the DataFrame

    df_aux = df.copy()
    df_aux = df_aux[df_aux['future_week'] == 0]

    max_year = df_aux['year'].max()
    max_week = df_aux[df_aux['year'] == max_year]['week'].max()
    
    # Convert max_year and max_week to a datetime object representing the start of the next week
    max_date = datetime.strptime(f'{max_year}-W{int(max_week)}-1', "%Y-W%W-%w") + timedelta(weeks=1)
    
    # Calculate the future date which is today plus four weeks
    current_date = max_date

    # Calculate the next Monday
    days_until_monday = (7 - current_date.weekday()) % 7
    next_week_monday = current_date + timedelta(days=days_until_monday)

    # Generate weeks forward or backward according to the model
    if st.session_state.model == foto_model_weather:
        start_week_monday = next_week_monday - timedelta(weeks=1)  # Empezar desde la semana pasada
        week_options = [
            f"{(start_week_monday - timedelta(weeks=i)).strftime('%d/%m/%Y')} - "
            f"{(start_week_monday - timedelta(weeks=i) + timedelta(days=6)).strftime('%d/%m/%Y')} - "
            f"Semana {(start_week_monday - timedelta(weeks=i)).strftime('%W')}"
            for i in range(4)
        ]
    if st.session_state.model == foto_current_model or st.session_state.model == ceutics_current_model or st.session_state.model == derma_acniben_current_model:
        start_week_monday = next_week_monday
        week_options = [
            f"{(start_week_monday + timedelta(weeks=i)).strftime('%d/%m/%Y')} - "
            f"{(start_week_monday + timedelta(weeks=i) + timedelta(days=6)).strftime('%d/%m/%Y')} - "
            f"Semana {(start_week_monday + timedelta(weeks=i)).strftime('%W')}"
            for i in range(4)
        ]
    
    return week_options


def _max_width_(prcnt_width:int = 75):
    """
    Generates a CSS style block to set the maximum width of a container in Streamlit.

    Args:
        prcnt_width (int): Percentage width to set as the maximum width of the container.

    """
    max_width_str = f"max-width: {prcnt_width}%;"
    st.markdown(f""" 
                <style> 
                .block-container{{{max_width_str}}}
                
                </style>    
                """, 
                unsafe_allow_html=True,
    )

    
def reset_simulador():
    """
    Resets the simulation state by setting 'en_simulacion' to False and resetting inversion fields.

    Args:
        len_channels (int): Number of channels for which inversion fields are reset.
    """
    for key in st.session_state.keys():
        if key.startswith("inversion_field_"):
            st.session_state[key] = "0,00"

    st.session_state["en_simulacion"] = False

def format_value(x):
    """
    Formats numeric values to a string with two decimal places, using commas as thousands separator and periods for decimals.

    Args:
        x (int or float): Numeric value to format.

    Returns:
        str or x: Formatted string if x is numeric, otherwise returns x unchanged.
    """
    if isinstance(x, (int, float)):
        return "{:,.2f}".format(x).replace(",", ";").replace(".", ",").replace(";", ".")
    return x

def format_float_predict_table_weeks(df):
    """
    Formats columns in a DataFrame:
    - Converts integers to strings with commas as thousands separator.
    - Formats floats to strings with two decimals, using commas as thousands separator and periods as decimal separator.

    Args:
        df (pd.DataFrame): DataFrame to format.

    Returns:
        pd.DataFrame: Formatted DataFrame.
    """
    formatted_df = df.copy()

    for col in formatted_df.columns:
        if pd.api.types.is_numeric_dtype(formatted_df[col]):
            if formatted_df[col].dtype == 'int64':
                formatted_df[col] = formatted_df[col].apply(lambda x: "{:,}".format(x))
            elif formatted_df[col].dtype == 'float64':
                formatted_df[col] = formatted_df[col].apply(lambda x: "{:,.2f}".format(x).replace(",", ";").replace(".", ",").replace(";", "."))
    return formatted_df

def update_inversions(sim_data):
    """
    Updates inversion values from session state based on simulation data.

    Args:
        sim_data (dict): Simulation data containing channels.

    """
    max_channels = len(sim_data["channels"]) - 1  # Subtracting 1 because the first element is a label
    inversions_values = {}
    for i in range(max_channels):
        inversion_key = f"inversion_field_{i}"
        if inversion_key in st.session_state:
            inversions_values[inversion_key] = float(st.session_state[inversion_key].replace(",", "."))
        else:
            inversions_values[inversion_key] = 0.0

    st.session_state["inversions_values"] = inversions_values
    st.session_state["en_simulacion"] = True

def main(login_role):

    # if login_role == 'Admin':
        # BU_MODEL_DICT = {
        #     "Producto 1": {
        #         "Model Weeks": current_model,
        #         "Model Weather": foto_model_weather
        #     },
        #     "Producto 2": {
        #         "Model Weeks": test
        #     }
        # }
        
    BU_MODEL_DICT = {
    "Producto 1": {
        "Model Weeks": foto_current_model,
        "Model Weather": foto_model_weather
    },
        "Producto 2": {
            "Model Weeks": ceutics_current_model
        },
        "Producto 3": {
            "Model Weeks": derma_acniben_current_model
        }
}

    _max_width_(75)
    with open('style.css') as f:
        css = f.read()
    st.markdown(f'<style>{css}</style>', unsafe_allow_html=True)

    # Initialize the state if it does not exist
    if 'business_unit' not in st.session_state:
        st.session_state.business_unit = list(BU_MODEL_DICT.keys())[0]

    if 'model' not in st.session_state:
        st.session_state.model_name = list(BU_MODEL_DICT[list(BU_MODEL_DICT.keys())[0]].keys())[0]
        st.session_state.model = BU_MODEL_DICT[list(BU_MODEL_DICT.keys())[0]][st.session_state.model_name]
    
    if 'week_option' not in st.session_state:
        st.session_state.week_option = generate_week_options(foto_df_final)[0]

    if 'investment' not in st.session_state:
        st.session_state.investment = investments[0]

    selector_container = st.container()
    col1, col2, col3, col4 = selector_container.columns([20,20,30,20])

    # Business Unit Selector in col0
    with col1:
        selected_business_unit = st.selectbox(
            "Selecciona el business unit",
            list(BU_MODEL_DICT.keys()),
            index=list(BU_MODEL_DICT.keys()).index(st.session_state.business_unit) if st.session_state.business_unit in BU_MODEL_DICT else 0
        )
        if selected_business_unit != st.session_state.business_unit:
            st.session_state.business_unit = selected_business_unit
            # Reset model when business unit changes
            st.session_state.model_name = list(BU_MODEL_DICT[selected_business_unit].keys())[0]
            st.session_state.model = BU_MODEL_DICT[selected_business_unit][st.session_state.model_name]
            reset_simulador()
            st.experimental_rerun()

    # Model Selector in col1 based on the selected business unit
    with col2:
        models_for_business_unit = BU_MODEL_DICT[st.session_state.business_unit]
        selected_model_name = st.selectbox(
        "Selecciona el modelo",
        models_for_business_unit.keys(), 
        index=investments.index(st.session_state.business_unit) if st.session_state.business_unit in models_for_business_unit.keys() else 0
        )
        
        if selected_model_name != st.session_state.model_name:
            st.session_state.model_name = selected_model_name
            st.session_state.model = models_for_business_unit[selected_model_name]
            reset_simulador()
            st.experimental_rerun()


    with col3:
        if st.session_state.model == foto_current_model:
            week_options = generate_week_options(foto_df_final)
            selected_week = st.selectbox(
                "Periodo de tiempo", 
                week_options,
                index=week_options.index(st.session_state.week_option) if st.session_state.week_option in week_options else 0
            )
        if st.session_state.model == foto_model_weather:
            week_options = generate_week_options(foto_df_final_weather)
            selected_week = st.selectbox(
                "Periodo de tiempo", 
                week_options,
                index=week_options.index(st.session_state.week_option) if st.session_state.week_option in week_options else 0
            )
        if st.session_state.model == ceutics_current_model:
            week_options = generate_week_options(ceutics_df_final)
            selected_week = st.selectbox(
                "Periodo de tiempo", 
                week_options,
                index=week_options.index(st.session_state.week_option) if st.session_state.week_option in week_options else 0
            )
        if st.session_state.model == derma_acniben_current_model:
            week_options = generate_week_options(derma_acniben_df_final)
            selected_week = st.selectbox(
                "Periodo de tiempo", 
                week_options,
                index=week_options.index(st.session_state.week_option) if st.session_state.week_option in week_options else 0
            )
        
        if selected_week != st.session_state.week_option:
            st.session_state.week_option = selected_week
            reset_simulador()
            st.experimental_rerun()


    with col4:
        # Determine the button text based on the current value
        selected_investment = st.selectbox(
            "Selecciona la inversión",
            investments,
            index=investments.index(st.session_state.investment) if st.session_state.investment in investments else 0
        )
        
        if selected_investment != st.session_state.investment:
            st.session_state.investment = selected_investment
            reset_simulador()
            st.experimental_rerun()
    
    
    ############################--------------- TIMESERIES GRAPHS ---------------###########################################
    first_graph_container = st.container()
    with first_graph_container:
        col1, col2 = st.columns([5,95])
        col1.image("assets/first_graph.png", width=40)
        col2.subheader('Ventas e inversiones por semana')
        button_container = st.container()
        with button_container:
            col1, col3, col4, col5, col6, col7, col8= st.columns([20,8, 8, 8, 10, 10,70])
            col1.write("Seleccionar periodo:")
            button_4s = col3.button("4s")
            button_3m = col4.button("3m")
            button_6m = col5.button("6m")    
            button_12m = col6.button("12m")
            button_24m = col7.button("24m")
            
    if "timeseries_period" not in st.session_state:
        st.session_state.timeseries_period = "24"
    if button_4s:
        st.session_state.timeseries_period = "4"
    elif button_3m:
        st.session_state.timeseries_period = "12"
    elif button_6m:
        st.session_state.timeseries_period = "24"
    elif button_12m:
        st.session_state.timeseries_period = "48"
    elif button_24m:
        st.session_state.timeseries_period = "96"
    else:
        st.session_state.timeseries_period = st.session_state.timeseries_period

    st.session_state.year = int(selected_week.split(" - ")[0].split("/")[2])
    st.session_state.week = int(selected_week.split(" - ")[2].split(" ")[1])

    if st.session_state.model == foto_model_weather:
        initial_calulation = initial_df(foto_df_final_weather)
    if st.session_state.model == foto_current_model:
        initial_calulation = initial_df(foto_df_final)
    if st.session_state.model == ceutics_current_model:
        initial_calulation = initial_df(ceutics_df_final)
    if st.session_state.model == derma_acniben_current_model:
        initial_calulation = initial_df(derma_acniben_df_final)

    limit_conf_80 = initial_calulation[1]["80_perc"]
    limit_conf_95 = initial_calulation[1]["95_perc"]
    df_initial = initial_calulation[0]
    df_initial2 = initial_calulation[2]
    ventas_previstas = int(round(df_initial.loc[(df_initial['year'] == st.session_state.year) & (df_initial['week'] == st.session_state.week)]['sales_without_investment'].values[0]))
    ventas_previstas_aov = int(round(df_initial.loc[(df_initial['year'] == st.session_state.year) & (df_initial['week'] == st.session_state.week)]['sales_without_investment_aov'].values[0]))
    ### Initial State of the Interface
    df_timeseries_sales = time_series(df_initial, st.session_state.timeseries_period, ["date", *sales_columns_to_plot])
    if st.session_state.model == foto_model_weather or st.session_state.model == foto_current_model:
        df_timeseries_investments = time_series(df_initial, st.session_state.timeseries_period, ["date", *foto_investment_columns_to_plot])
    if st.session_state.model == ceutics_current_model:
        df_timeseries_investments = time_series(df_initial, st.session_state.timeseries_period, ["date", *ceutics_investment_columns_to_plot])
    if st.session_state.model == derma_acniben_current_model:
        df_timeseries_investments = time_series(df_initial, st.session_state.timeseries_period, ["date", *derma_acniben_investment_columns_to_plot])
    df_ds = simulacion_initial(df_initial)
    dict_ds = df_ds.astype(str).to_dict()

    ### State After the Simulation
    if "en_simulacion" in st.session_state:
        if st.session_state["en_simulacion"]:
            despues_calculation = after_sim_df(df_initial2)
            df_despues = despues_calculation[0].loc[(despues_calculation[0]['year'] == st.session_state.year) & (despues_calculation[0]['week'] == st.session_state.week)]
            df_timeseries_sales = time_series(despues_calculation[0], st.session_state.timeseries_period, ["date", *sales_columns_to_plot])
            if st.session_state.model == foto_model_weather or st.session_state.model == foto_current_model:
                df_timeseries_investments = time_series(despues_calculation[0], st.session_state.timeseries_period, ["date", *foto_investment_columns_to_plot])
            if st.session_state.model == ceutics_current_model:
                df_timeseries_investments = time_series(despues_calculation[0], st.session_state.timeseries_period, ["date", *ceutics_investment_columns_to_plot])
            if st.session_state.model == derma_acniben_current_model:
                df_timeseries_investments = time_series(despues_calculation[0], st.session_state.timeseries_period, ["date", *derma_acniben_investment_columns_to_plot])
            ventas_previstas = int(round(df_despues.loc[(df_despues['year'] == st.session_state.year) & (df_despues['week'] == st.session_state.week)]['investment_sales'].values[0]))
            ventas_previstas_aov = int(round(df_despues.loc[(df_despues['year'] == st.session_state.year) & (df_despues['week'] == st.session_state.week)]['investment_sales_aov'].values[0]))
            limit_conf_80 = despues_calculation[1]["80_perc"]
            limit_conf_95 = despues_calculation[1]["95_perc"]
    
   
    
    lines = alt.Chart(df_timeseries_sales).mark_line(point=True).encode(
    x=alt.X('date', title=""),
    y=alt.Y('cantidad:Q', title=""),
    color=alt.Color('tipo', title=" ")
    
    ).configure(background='#222222').configure_legend(orient="bottom",labelFontSize=12)
    first_graph_container.altair_chart(lines, use_container_width=True)


    lines = alt.Chart(df_timeseries_investments).mark_line(point=True).encode(
    x=alt.X('date', title=""),
    y=alt.Y('cantidad', title=""),
    color=alt.Color('tipo', title=" ")
    ).configure(background='#222222').configure_legend(orient="bottom",labelFontSize=10)
    
    first_graph_container.altair_chart(lines, use_container_width=True)
    

    sim_data = {
        "channels": [""] + [channel_name.split("-")[0] for channel_name in list(dict_ds["channel"].values())],
        "ratings": ["Significancia"] + ratings_to_stars([round(float(rating), 2) for rating in list(dict_ds["estrellas"].values())]),
        "adstock": ["Adstock"] + ["€ " + str(round(float(adstock.replace(",", ".")), 2)).replace(".", ",") for adstock in list(dict_ds["Adstock [euros]"].values())],
        "saturacion": ["Saturación"] + [saturacion for saturacion in list(dict_ds["Saturación"].values())],
        "max_historico": ["Máximo \nhistórico"] + [str(max_historico).replace(".", ",") for max_historico in list(dict_ds["Máximo histórico [euros]"].values())],
        "inversion_adstock": ["Inversión +\n Adstock"] + calculate_inversionAdstock(list(dict_ds["Inversión + Adstock"].values()))
    }



    sim_data_columns = list(sim_data.keys())
    len_channels = len(sim_data["channels"]) - 1

    st.markdown("# Simulador de inversiones")
    sim_container = st.container()

    with sim_container:
        col1, col2 = st.columns([80, 20])
        with col1:
            for i in range(len(sim_data["channels"])):
                row_container = st.container()
                with row_container:
                    row_container_cols = st.columns(6)
                    for j in range(6):
                        with row_container_cols[j]:
                            st.text(sim_data[sim_data_columns[j]][i])

        with col2:
            st.text("Inversión semanal")
            
            col1, col2 = col2.columns([20, 80])
            with col1:
                for i in range(len_channels):
                    st.write("€")
            with col2:
                for i in range(len_channels):
                    st.text_input(value="0,00", label="inversion_field", label_visibility="collapsed", key="inversion_field_" + str(i))
        
        
        sim_result_container = st.container()
        with sim_result_container:
            col1, col2 = st.columns([80, 20])

            col1.write("Monto Invertido:")
            monto_invertido = round(sum([float(st.session_state["inversion_field_" + str(i)].replace(",", ".")) if st.session_state["inversion_field_" + str(i)] else 0.0 for i in range(len_channels)]), 2)
            col2.write("€ {:.2f}".format(monto_invertido).replace(".", ","))

            col1.write("Monto Invertido con Adstock:")
            monto_con_adstock = sum([float(adstock.replace("€ ", "").replace(",", ".")) for adstock in sim_data["adstock"][1:]])
            col2.write("€ {:.2f}".format(monto_con_adstock+monto_invertido).replace(".", ","))
            texto = f"<span>Unidades previstas: </span><span>{ventas_previstas}</span> unid."
            st.markdown(texto, unsafe_allow_html=True)
            limit_conf_80_0 = int(round(limit_conf_80[0][0])) if limit_conf_80[0][0] > 0 else 0
            limit_conf_80_1 = int(round(limit_conf_80[0][1]))
            limit_conf_95_0 = int(round(limit_conf_95[0][0])) if limit_conf_95[0][0] > 0 else 0
            limit_conf_95_1 = int(round(limit_conf_95[0][1]))
            st.markdown(f"IC 80%: {limit_conf_80_0} a {limit_conf_80_1} unid.  \nIC 95%: {limit_conf_95_0} a {limit_conf_95_1} unid.")
            
            texto = f"<span>Ventas previstas: </span><span>{ventas_previstas_aov} €</span>"
            st.markdown(texto, unsafe_allow_html=True)

        sim_button_container = st.container()
        with sim_button_container:
            col1, col2 = st.columns(2)
            col1.button("🗑️ Limpiar", on_click=lambda: reset_simulador())
            col2.button("📈 Predecir retorno", on_click=lambda: update_inversions(sim_data))
    

    st.markdown("# Atribución de canales")
    channel_attribution_container = st.container()
    with channel_attribution_container:
        # Set default value in the session state
        if "selected_view_df_period" not in st.session_state:
            st.session_state.selected_view_df_period = "Anual"

        view_df_period = pd.DataFrame()
        col1, col2 = st.columns([20, 80])

        with col1:
            selected_view_df_period = st.selectbox(
                "Selecciona el periodo",
                ["Anual", "Mensual", "Semanal"],
                index=["Anual", "Mensual", "Semanal"].index(st.session_state.selected_view_df_period),
                key="selected_view_df_period",
            )

            # Select DataFrame based on the selected period
            if selected_view_df_period == 'Anual':
                if st.session_state.business_unit == 'Producto 1':
                    view_df_period = pd.DataFrame(foto_df_year)
                elif st.session_state.business_unit == 'Producto 2':
                    view_df_period = pd.DataFrame(ceutics_df_year)
                elif st.session_state.business_unit == 'Producto 3':
                    view_df_period = pd.DataFrame(derma_df_year)
            elif selected_view_df_period == 'Mensual':
                if st.session_state.business_unit == 'Producto 1':
                    view_df_period = pd.DataFrame(foto_df_month)
                elif st.session_state.business_unit == 'Producto 2':
                    view_df_period = pd.DataFrame(ceutics_df_month)
                elif st.session_state.business_unit == 'Producto 3':
                    view_df_period = pd.DataFrame(derma_df_month)
            elif selected_view_df_period == 'Semanal':
                if st.session_state.business_unit == 'Producto 1':
                    view_df_period = pd.DataFrame(foto_df_week)
                elif st.session_state.business_unit == 'Producto 2':
                    view_df_period = pd.DataFrame(ceutics_df_week)
                elif st.session_state.business_unit == 'Producto 3':
                    view_df_period = pd.DataFrame(derma_df_week)

        if not view_df_period.empty:
            if "coefficient" in view_df_period.columns:
                view_df_period = view_df_period.drop(columns=["coefficient"])

            # Format numeric values (2 decimals, comma as decimal separator)
            view_df_period = view_df_period.applymap(
                lambda x: f"{x:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
                if isinstance(x, float) else x
            )

            # Column names
            header_container = st.container()
            with header_container:
                header_cols = st.columns(len(view_df_period.columns))
                for j, col_name in enumerate(view_df_period.columns):
                    with header_cols[j]:
                        st.markdown(f"**{col_name}**") 

            # Display rows of the DataFrame
            for i in range(len(view_df_period)):
                row_container = st.container()
                with row_container:
                    row_container_cols = st.columns(len(view_df_period.columns))
                    for j, col_name in enumerate(view_df_period.columns):
                        with row_container_cols[j]:
                            st.text(view_df_period.iloc[i, j])



    invs_retEsp_distInv_container = st.container()
    with invs_retEsp_distInv_container:
        col1, col2 = st.columns([50,50])
        ### Inversiones [Eur]
        with col1:
            list_inversion_semanal = []
            for i in range(len(sim_data["channels"])-1):
                list_inversion_semanal.append(st.session_state["inversion_field_" + str(i)].replace(",", "."))
            df_bar_inversion = pd.DataFrame({
                "channels": sim_data["channels"][1:],
                "inversion_adstock": [inversion.replace(",", ".") for inversion in sim_data["inversion_adstock"][1:]],
                "inversion_semanal": list_inversion_semanal
            })  

            
            # Reshape the data for side-by-side bars
            df_bar_inversion_melted = df_bar_inversion.melt('channels', var_name='Variable', value_name='Value')
            # Create the grouped bar chart
            chart = alt.Chart(df_bar_inversion_melted).mark_bar().encode(
                x=alt.X('channels:N', title=''),
                y=alt.Y('Value:Q', title=''),
                color=alt.Color('Variable:N', title=" "),
            ).configure(background='#222222').configure_legend(orient="bottom",labelFontSize=8)
            st.markdown("# Inversiones [euros]")
            st.altair_chart(chart, use_container_width=True)  

        ### Investment distribution   
        with col2:
            df_pie_inversion = df_bar_inversion[["channels", "inversion_adstock"]]
            df_pie_inversion['inversion_adstock'] = df_pie_inversion['inversion_adstock'].astype(float)
            df_pie_inversion["perc_inversion_adstock"] = ((df_pie_inversion['inversion_adstock']/(df_pie_inversion['inversion_adstock'].sum()))*100).round(2)
            df_pie_inversion = df_pie_inversion.loc[df_pie_inversion["perc_inversion_adstock"] > 0]
            
            
            ch = alt.Chart(df_pie_inversion).mark_arc().encode(
            theta=alt.Theta(field="perc_inversion_adstock", type="quantitative"),
            color=alt.Color(field="channels", type="nominal", title=''),).configure(background='#222222')
            st.markdown("# Distribución de inversiones [%]")
            st.altair_chart(ch, use_container_width=True)
        
    ####### Container Resultados reales x previsto, Modelo y OLS regression, Influencia de variables en el Modelo
    evaluacion_modelo_container_p1 = st.container()
    with evaluacion_modelo_container_p1:
        if st.session_state.model == foto_model_weather or st.session_state.model == foto_current_model or st.session_state.model == derma_acniben_current_model:
            ####### Share of Spend y Share of effect
            df_shared_melted = get_shareSpend_shareEffect(df_bar_inversion)

            # Reshape the data for side-by-side bars
            df_shared_melted = df_shared_melted.melt('channel', var_name='Variable', value_name='Value')

            # Sort the DataFrame by channel and Variable
            df_shared_melted = df_shared_melted.sort_values(by=['channel', 'Variable'], ascending=[True, False])

            # Concatenate channel and Variable with an underscore
            df_shared_melted['channel_Variable'] = df_shared_melted.apply(lambda row: f"{row['channel']}_{row['Variable']}", axis=1)

            # Reset the index
            df_shared_melted = df_shared_melted.reset_index(drop=True)

            # Specify the colors manually
            color_scale = alt.Scale(domain=['share of spent', 'share_of_effect'],
                        range=['#076bc8','#83cafa'])  # Cambia los colores según tus preferencias

            chart = alt.Chart(df_shared_melted).mark_bar().encode(
                x=alt.X('channel_Variable:N', title='', sort=list(df_shared_melted['channel_Variable'])),
                y=alt.Y('Value:Q', title=''),
                color=alt.Color('Variable:N', title=" ", scale=color_scale),
            ).configure(background='#222222').configure_legend(orient="bottom",labelFontSize=8)

            evaluacion_modelo_container_p1.markdown("# Share of Spend y Share of Effect")
            evaluacion_modelo_container_p1.altair_chart(chart, use_container_width=True)

        if st.session_state.model == ceutics_current_model:
            evaluacion_modelo_container_p1.markdown("# ")
            evaluacion_modelo_container_p1.markdown("# ")
        
    evaluacion_modelo_container_p2 = st.container()
    with evaluacion_modelo_container_p2:
        col1, col2 = st.columns([37, 63])
        with col1:
            if st.session_state.model == foto_current_model:
                ols_reg_data = foto_evaluation_current_model.copy()
            if st.session_state.model == foto_model_weather:
                ols_reg_data = foto_evaluation_weather.copy()
            if st.session_state.model == ceutics_current_model:
                ols_reg_data = ceutics_evaluation_current_model.copy()
            if st.session_state.model == derma_acniben_current_model:
                ols_reg_data = derma_acniben_evaluation_current_model.copy()

            ols_reg_data_formatted = ols_reg_data.applymap(format_value).melt(var_name='metric', value_name='value')

            ols_reg_data_styled = ols_reg_data_formatted.style.set_properties(**{'background-color': '#222222'})

            st.dataframe(ols_reg_data_styled, hide_index=True)

            df_influencia = get_influencia_variables()
            chart = alt.Chart(df_influencia).mark_bar().encode(
            x=alt.X('coeficientes_escalonadas:Q', title=''),
            y=alt.Y('channel:N', title='')
            ).configure(background='#222222')
            if st.session_state.model == foto_model_weather or st.session_state.model == foto_current_model or st.session_state.model == derma_acniben_current_model:
                st.markdown("# Influencia de las variables en el modelo")
                st.markdown("*Los coeficientes muestran la influencia relativa de cada variable en el modelo sin reflejar su valor original")
            if st.session_state.model == ceutics_current_model:
                st.markdown("# Importancia de las variables en el modelo")
                st.markdown("*Los coeficientes de las variables no indican la magnitud y dirección, sólo podemos estudiar su importancia en la predicción")

            st.altair_chart(chart, use_container_width=True)    

        with col2:
            st.markdown("# Resultados reales X previstos")

            if st.session_state.model == foto_model_weather:
                predict_table = foto_predict_table_weather
            if st.session_state.model == foto_current_model:
                predict_table = foto_predict_table_weeks
            if st.session_state.model == ceutics_current_model:
                predict_table = ceutics_predict_table
            if st.session_state.model == derma_acniben_current_model:
                predict_table = derma_acniben_predict_table_weeks

            csv_file_predict_table = predict_table.applymap(lambda x: f"{x:.2f}" if isinstance(x, float) else x)
            csv_file_predict_table = csv_file_predict_table.drop(columns=csv_file_predict_table.filter(like='markets').columns)

            # Convert the "year" column to string data type
            predict_table["year"] = predict_table["year"].astype(str)
            # Apply formats to sort predict_table_weeks
            predict_table = format_float_predict_table_weeks(predict_table)
            predict_table = predict_table.drop(columns=predict_table.filter(like='markets').columns)
            # Display the DataFrame in Streamlit
            st.dataframe(predict_table.reset_index(drop=True))
            # Button to export as CSV
            csv_file_predict_table = csv_file_predict_table.to_csv(index=False)
            st.download_button(
                label="Descargar como CSV",
                data=csv_file_predict_table,
                file_name="predict_table.csv",
                key="csv_download_button",
                help="Presiona para descargar el archivo CSV."
            )
    
    admin_google_redirect_uri = os.environ["ADMIN_GOOGLE_REDIRECT_URI"]
    st.markdown(f"[Sección Administrador]({admin_google_redirect_uri})")
    st.text("En caso de duda, póngase en contacto con nuestro equipo de soporte.")
    st.image("assets/datarmony_fin.webp")
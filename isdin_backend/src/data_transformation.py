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
from sklearn.ensemble import RandomForestRegressor

def generate_data(path_query, bu = None, brand = None, index_bu = None, country = None, list_bucket = None):
    """
    Generates a dynamic SQL file.

    Args:
    path_query (str): Path to the SQL query file.
    bu (str): Business unit to be used in the SQL query.

    Returns:
    df (DataFrame): Result of the executed SQL query as a DataFrame.
    """ 

    # Set the Google Cloud credentials environment variable
    google_credentials_json = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    credentials = service_account.Credentials.from_service_account_info(json.loads(google_credentials_json))
    project = credentials.project_id

    client = bigquery.Client(project, credentials)

    # Read the SQL query template from the specified file
    with open(path_query, 'r', encoding='utf-8') as file_sql:
        template_sql = file_sql.read()

    # Render the template with the dynamic value 'bu'
    template = Template(template_sql)
    query_sql = template.render(bu = bu, brand = brand, index_bu = index_bu, country = country, list_bucket = list_bucket)
    
    # Execute the rendered SQL query and return the result as a DataFrame
    df = client.query(query_sql).to_dataframe()
    return df


def adding_rows_of_future_weeks_to_forecast(df, list_bucket, country_id, currency, bu):
    """
    Add future weeks to a DataFrame from the week after the maximum date in the DataFrame to today plus four weeks (tool forecast options).
    
     Args:
        df (pd.DataFrame): training df with adstock.
        list_bucket: list of inversion channels (without -adstock suffix).

    Returns:
        df (pd.DataFrame): training df with future dates to feed the model in the prediction and to generate interface graphics.
    """
    
    # Find the maximum year and week in the DataFrame
    max_year = df['year'].max()
    max_week = df[df['year'] == max_year]['week'].max()
    
    # Convert max_year and max_week to a datetime object representing the start of the next week
    max_date = datetime.fromisocalendar(max_year, max_week, 1) + timedelta(weeks=1)
    # Calculate the future date which is today plus four weeks
    today = max_date
    future_date = today + timedelta(weeks=3) # Tool forecast options (maximum 4 weeks in the future)
    
    # Generate the weeks between the maximum date and the future date
    current_date = max_date
    new_rows = []
    while current_date <= future_date:
        year, week, _ = current_date.isocalendar()
        month = current_date.month
        new_row = {'year': year, 'week': week, 'month': month, 'country_id' : country_id, 'currency' : currency, 'bu' : bu}
        for column in list_bucket:
            new_row[column] = 0
        new_rows.append(new_row)
        current_date += timedelta(weeks=1)
    
    # Create a DataFrame with the new rows
    forecast_dates = pd.DataFrame(new_rows)
    
    # Concatenate the new rows to the original DataFrame
    df = pd.concat([df, forecast_dates], ignore_index=True)

    # adding 'future_week' column that indicates whether the row corresponds to a future date or not
    df['future_week'] = 0
    df.loc[df.index[df.index >= len(df) - len(forecast_dates)], 'future_week'] = 1
    df_filled = df.fillna(0)

    return df

def transform_data(df, list_bucket, bu = None, brand = None):
    """
    Processes raw data in several steps:
    - Fills null values with zero and removes columns with a total sum of zero.
    - Adds a feature for units sold in the same week of the previous year.
    - Handles collinearity by combining correlated columns.
    - Excludes outliers based on the IQR method (commented out in this code).
    - Deletes rows with null values in the 'units' column.
    - Applies one-hot encoding to the 'week' and 'month' columns.
    
    Args:
    df (pd.DataFrame): Raw data.
    list_bucket (list): Initial list of column names to process.


    Returns:
    pd.DataFrame: Processed DataFrame with training data.
    updated_list_bucket (list): Updated list of column names after processing.############################## no es la mejor explicacion!! Adstock.. one-hot encoding
    correlation_matrix (pd.DataFrame): Investment channel correlation matrix.
    """
    
    # Fill null values with zero and remove columns with zero sum
    df.fillna(0, inplace=True)
    if any(bucket in df.columns for bucket in list_bucket):
        for bucket in list_bucket:
            if bucket in df.columns:
                if df[bucket].fillna(0).sum() == 0:
                    df.drop(bucket, axis=1, inplace=True)
    updated_list_bucket = [bucket for bucket in list_bucket if bucket in df.columns]
    
    # Add feature for units sold in the same week of the previous year
    df = df.copy()
    df['previous_units_1'] = df.sort_values(by=['year']).groupby(['week'])['units'].shift(1)
    df['previous_units_1'].fillna(0, inplace=True)
    # Since we don't have data for 2020, 2021 would be 0, so we solve it like this:
    # Filter the data for years other than 2021 and calculate the weekly average of 'previous_units_1' ()
    weekly_average = df[df['year'] != 2021].groupby('week')['previous_units_1'].mean()
    # Now, for the year 2021, assign the calculated average by week
    df.loc[df['year'] == 2021, 'previous_units_1'] = df[df['year'] == 2021]['week'].map(weekly_average)
    df['previous_units_1'] = df['previous_units_1'].astype('float64')

    # Collinearity: This section must have human supervision, with the help of alerts for new collinearities
    correlation_matrix = df[updated_list_bucket].corr() # generating correlation matrix that must be saved in a BQ table and set an alert

    # Handle collinearity: combine correlated columns
    if bu != 'Derma' and brand != 'Acniben':
        # Handle collinearity: combine correlated columns
        df['SEM_Display'] = df['SEM'] + df['Display'] # known correlation
        df = df.drop(['SEM', 'Display', 'Otros'], axis=1) # Others are correlated with TV
        updated_list_bucket = list(set(updated_list_bucket) - {'SEM', 'Display', 'Otros'}) # Remove unwanted elements from updated_list_bucket
        updated_list_bucket.extend(['SEM_Display']) # Add new elements to updated list bucket
    
    # Exclude outliers from the columns of units sold and investments in advertising channels using the IQR method
    # outliers = pd.DataFrame()
    # for column in ['units'] + updated_list_bucket:
    #    temp_df = df.copy()
    
    #    Q1 = temp_df[column].quantile(0.25)
    #    Q3 = temp_df[column].quantile(0.75)
    #    IQR = Q3 - Q1
    
    #    lower_bound = Q1 - 1.5 * IQR
    #    upper_bound = Q3 + 1.5 * IQR
    
    #    outliers = temp_df[(temp_df[column] < lower_bound) | (temp_df[column] > upper_bound)]
    
    # df = df[~df.index.isin(outliers.index)]

    # Remove rows with null values in the 'units' column
    df = df.dropna(subset=['units'])

    # Apply one-hot encoding to 'week' column
    original_week = df['week']
    df = pd.get_dummies(df, columns=['week'], prefix='week')
    df.drop(columns=['week_1'], inplace=True) 
    week_columns = [col for col in df.columns if col.startswith('week_')]
    df[week_columns] = df[week_columns].astype(int)
    df['week'] = original_week

    # Apply one-hot encoding to 'month' column
    original_month = df['month']
    df = pd.get_dummies(df, columns=['month'], prefix='month')
    df.drop(columns=['month_1'], inplace=True) 
    month_columns = [col for col in df.columns if col.startswith('month_')]
    df[month_columns] = df[month_columns].astype(int)
    df['month'] = original_month
    return df, updated_list_bucket


def adstock_data(df, theta, updated_list_bucket):
    """
    Applies adstock transformation to investment columns in the DataFrame.

    Adstock transformation is a technique used in marketing analytics to model the
    lasting effect of advertising and other marketing inputs on consumer behavior.

    Args:
    df (pd.DataFrame): The input DataFrame containing the data to be transformed.
    theta (float): The decay factor for the adstock transformation. It represents
        the rate at which the effect of the input decreases over time.
    updated_list_bucket (list): List of column names to be transformed.

    Returns:
    pd.DataFrame: DataFrame with adstock-transformed columns and 'just_adstock' columns appended.
    list_bucket_adstock (list): List of the column names of the adstock-transformed columns (columns that are the sum of investment and adstock).
    list_just_adstock (list): List of the column names of the 'just_adstock' columns (columns with only the adstock value).
    """
    
    # Reset index to ensure continuous integer index
    df = df.reset_index()
    # Drop the old index column
    df = df.drop('index', axis=1)

    # Initialize adstock array with zeros
    adstock = np.zeros((df.shape[0], len(updated_list_bucket)))

    # Apply adstock transformation to each specified column
    for i, inv in enumerate(updated_list_bucket):
        tab_inv = df[inv].values
        adstock[0, i] = tab_inv[0]
        for j in range(1, df.shape[0]):
            # Calculate adstock effect for each time step
            adstock[j, i] = round(tab_inv[j] + theta * adstock[j-1, i], 2)

    # Create a DataFrame from the adstock array with appropriate column names
    adstock = pd.DataFrame(adstock, columns=[f"{inv}-adstock" for inv in updated_list_bucket])
    # Get the list of adstock-transformed column names
    list_bucket_adstock = adstock.columns.tolist()

    # Reset DataFrame index to ensure alignment during concatenation
    df = df.reset_index(drop=True)

    # Concatenate the original DataFrame with the adstock-transformed columns
    df = pd.concat([df, adstock], axis=1)

    # Initialize list for just_adstock column names
    list_just_adstock = []

    # Calculate just_adstock for each channel and add to DataFrame
    for inv in updated_list_bucket:
        just_adstock_col = f'just_adstock_{inv}'
        df[just_adstock_col] = df[f"{inv}-adstock"] - df[inv]
        list_just_adstock.append(just_adstock_col)

    return df, list_bucket_adstock, list_just_adstock

def get_weather_data_api(df, api_key, df_cities, df_provincias, df_share_ccaa, bu):

    """
    Executes a process to download weather data from a public API.

    Using get_coordinates(df, api_key), it adds geographic coordinates (latitude and longitude) 
    to each city in the DataFrame via the Google Maps API. Next, get_weather(df) 
    downloads daily weather data for each coordinate within the specified date range. Finally, the resulting 
    DataFrame is formatted appropriately and returned as weather_data, which contains comprehensive weather 
    data for each city and date.

    Args:
    df (pd.DataFrame): DataFrame containing initial data.
    api_key (str): API key to access Google services.
    path_query_cities (str): Path to the SQL file containing the query to get the city data.
    path_query_prov (str): Path to the SQL file containing the query to get the province data.
    path_query_sales_share (str): Path to the SQL file containing the query to get the sales share data.

    Returns:
    pd.DataFrame: DataFrame containing weather data for each date within the specified range.
    """

    # Set the Google Cloud credentials environment variable
    google_credentials_json = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    # Convert the JSON string into a credentials object.
    credentials = service_account.Credentials.from_service_account_info(json.loads(google_credentials_json))
    project = credentials.project_id

    def get_coordinates(df_coordinates, api_key):

        latitudes = []
        longitudes = []

        # Iterate over each row in the DataFrame
        for i, row in df_coordinates.iterrows():
            # Construct the address from the city and country columns
            address = row['city'] + ', ' + row['country']
            
            # URL for the Google Geocoding API
            url = 'https://maps.googleapis.com/maps/api/geocode/json'
            
            # Parameters for the API request
            params = {'address': address, 'key': api_key}
            
            # Send the request to the API
            response = requests.get(url, params=params)
            
            # Parse the response as JSON
            data = response.json()
            
            # Check if the response status is 'OK'
            if data['status'] == 'OK':
                # Extract the latitude and longitude from the response
                lat = data['results'][0]['geometry']['location']['lat']
                lng = data['results'][0]['geometry']['location']['lng']
                
                # Append the latitude and longitude to their respective lists
                latitudes.append(lat)
                longitudes.append(lng)
            else:
                # If the status is not 'OK', append None to the lists
                latitudes.append(None)
                longitudes.append(None)
        
        # Add the latitudes and longitudes as new columns in the DataFrame
        df_coordinates['lat'] = latitudes
        df_coordinates['lon'] = longitudes
        
        # Return the updated DataFrame
        return df_coordinates
    

    def get_weather(df):
        # Create an empty DataFrame to store the weather data
        df_weather = pd.DataFrame()
        start_date = datetime(2020, 1, 1)
        end_date = datetime.now()

        # Loop through each row of the input DataFrame
        for idx, row in df.iterrows():
            # Create a Point object for the latitude and longitude
            point = Point(row['lat'], row['lon'])

            # Fetch the weather data for the specified point and time period
            daily_data = Daily(point, start_date, end_date).fetch()  # Daily data is stored in a DataFrame called daily_data

            # Add a column to the weather data indicating the city
            daily_data['city'] = row['city']

            # Append the daily data to the weather data DataFrame
            df_weather = pd.concat([df_weather, daily_data], axis=0) # daily_data is appended (concatenated) to the end of the DataFrame weather_data

        return df_weather
        


    

    #df_share_ccaa = df_share_ccaa[df_share_ccaa['year'] == '2023'].drop(columns=['year'])


    # Add coordinates (latitude and longitude) to the DataFrame using the Google Maps API
    data_coordinates = get_coordinates(df_cities, api_key)

    # Fetch weather data for the cities within the specified date range
    weather_data = get_weather(data_coordinates)

    # Reset the index and rename the index column to 'date'
    weather_data.reset_index(inplace=True)
    weather_data.rename(columns={'index': 'date'}, inplace=True)

    # Merge with df_provincias
    weather_data = weather_data.merge(df_provincias, on='city', how='left')

    # Select relevant columns
    weather_data = weather_data[['date', 'CCAA', 'tavg', 'prcp']]

    # Calculate the mean by 'date' and 'CCAA'
    weather_data = weather_data.groupby(['date', 'CCAA']).mean().add_suffix('_mean').reset_index()

    # Merge with df_share_ccaa
    weather_data = weather_data.merge(df_share_ccaa[df_share_ccaa['bu'] == bu].drop(columns=['bu']), on='CCAA', how='left')

    # Filter out rows corresponding to 'CEUTA' and 'MELILLA'
    weather_data = weather_data[~weather_data['CCAA'].isin(['CEUTA', 'MELILLA'])]

    # Calculate the weighted values
    weather_data['tavg_weighted'] = weather_data['tavg_mean'] * weather_data['sales']
    weather_data['prcp_weighted'] = weather_data['prcp_mean'] * weather_data['sales']

    weather_data_ccaa = weather_data.copy()

    # Select final columns
    weather_data = weather_data[['date', 'tavg_weighted', 'prcp_weighted']]

    # Group by 'date' and sum the weighted values
    weather_data = weather_data.groupby('date').sum().add_suffix('_mean').reset_index()

    # Extract the week number and year from the 'date' column
    weather_data['week'] = weather_data['date'].dt.isocalendar().week
    weather_data['year'] = weather_data['date'].dt.year

    # Group by 'year' and 'week' and calculate the mean of the weighted values
    weather_data = weather_data.groupby(['year', 'week']).agg({'tavg_weighted_mean': 'mean',
                                                                    'prcp_weighted_mean': 'mean'}).reset_index()

    # Merge the aggregated weather data with the existing DataFrame 'df' on 'year' and 'week'
    df_join_weather = df.merge(weather_data, on=['year', 'week'], how='left')

    # Return the final DataFrame containing weather data
    return df_join_weather, weather_data_ccaa

def calculate_sales_with_and_without_investment(df, list_just_adstock, features, model):
    """
    Apply a trained model to predict sales without investment and, in the case of future dates, forecast sales with the investment of JUST adstock.

    Args:
        df (pd.DataFrame): DataFrame containing all dates and relevant data.
        list_just_adstock (list): List of column names representing adstock-adjusted investments (columns with only the adstock value).
        features (list): List of feature column names used in the model.
        model: Trained model object with a predict() method.

    Returns:
        pd.DataFrame: DataFrame with 'sales_without_investment' and 'investment_sales' columns added.
    """
    # Create a copy of the DataFrame
    df_final = df.copy()
     
    # Creating the X matrix that will be used for sales prediction without investment
    columns_to_include = ['year'] + list_just_adstock + features #note that here we use the channel columns with only adstock value -> list_just_adstock 
    X = df_final[columns_to_include]
    X.columns = [
        col.split('just_adstock_')[-1] + '-adstock' if 'just_adstock_' in col else col
        for col in X.columns
    ]

    if not isinstance(model, RandomForestRegressor):
        X = sm.add_constant(X, prepend=True)
    
    if isinstance(model, RandomForestRegressor):
        feature_order = model.feature_names_in_
        X = X[feature_order]

    # Apply the model to predict 'sales without investment'
    df_final['sales_without_investment'] = model.predict(X)

    # For future 4 weeks columns: predict sales with adstock
    # Initially, set 'investment_sales' to 0
    df_final['investment_sales'] = 0
    # For historical data columns, investment sales = units (actual sales)
    df_final.loc[df_final['future_week'] == 0, 'investment_sales'] = df_final['units']
    # For future weeks: Sales with investment are the same as sales without investment, as at the moment the only future investments are the adstock itself.
    # For future weeks, set 'investment_sales' to 'sales_without_investment'
    df_final.loc[df_final['future_week'] == 1, 'investment_sales'] = df_final['sales_without_investment']
    
    # Adjust values of 'sales_without_investment' where it's greater than 'investment_sales'
    df_final['sales_without_investment'] = df_final.apply(lambda row: row['investment_sales'] if row['sales_without_investment'] > row['investment_sales'] else row['sales_without_investment'], axis=1)

    # Set negative values   to 0
    df_final['sales_without_investment'] = df_final['sales_without_investment'].apply(lambda x: 0 if x < 0 else x)

    # Adjust values of 'sales_without_investment' where it's greater than 'investment_sales'
    df_final['sales_without_investment'] = df_final.apply(lambda row: row['investment_sales'] if row['sales_without_investment'] > row['investment_sales'] else row['sales_without_investment'], axis=1)

    # Return the updated DataFrame
    return df_final


def transform_df_aov(query_aov, bu):
    """
    Transforms the input DataFrame by izoom market data and calculating additional metrics.
    
    Args:
    query_markets (str): Query to fetch izoom market data.
    df (pd.DataFrame): DataFrame raw.
    
    Returns:
    pd.DataFrame: DataFrame with weekly data, including additional market metrics.
    """
    
    # Generate market data from the query
    df_aov = generate_data(query_aov, bu)

     # Find the maximum year and week in the DataFrame
    max_year = df_aov['year'].max()
    max_week = df_aov[df_aov['year'] == max_year]['week'].max()
    
    # Convert max_year and max_week to a datetime object representing the start of the next week
    max_date = datetime.strptime(f'{max_year}-W{int(max_week)}-1', "%Y-W%W-%w") + timedelta(weeks=1)
    
    # Calculate the future date which is today plus four weeks
    today = max_date
    future_date = today + timedelta(weeks=3) # Tool forecast options (maximum 4 weeks in the future)

    df_last_weeks = df_aov[(df_aov['year'] == max_year) | (df_aov['year'] == max_year - 1)].nlargest(4, 'week')

    mean_aov = round(df_last_weeks['aov'].mean(), 2)
    
    # Generate the weeks between the maximum date and the future date
    current_date = max_date
    new_rows = []
    while current_date <= future_date:
        year, week, _ = current_date.isocalendar()
        new_row = {'year': year, 'week': week, 'units': 0, 'pvp': 0, 'aov': mean_aov}
        new_rows.append(new_row)
        current_date += timedelta(weeks=1)
    
    df_future_aov = pd.DataFrame(new_rows)

    df_aov = pd.concat([df_aov, df_future_aov], ignore_index=True)
    df_aov['aov'] = df_aov['aov'].fillna(0)

    return df_aov


def transform_df_market(query_markets, df, bu):
    """
    Transforms the input DataFrame by izoom market data and calculating additional metrics.
    
    Args:
    query_markets (str): Query to fetch izoom market data.
    df (pd.DataFrame): DataFrame raw.
    
    Returns:
    pd.DataFrame: DataFrame with weekly data, including additional market metrics.
    """
    
    # Generate market data from the query
    df_markets = generate_data(query_markets, bu)

    # Aggregate sales data by year and month
    df_raw_month = df.groupby(['year', 'month'], as_index=False).agg(units_sales=('units', 'sum'))

    # Rename columns in the market data
    df_markets = df_markets.rename(columns={'units': 'units_markets', 'pvp': 'pvp_markets'})

    # Merge aggregated sales data with market data
    df_merged = pd.merge(df_raw_month, df_markets, on=['year', 'month'], how='left')
    df_merged['pvp_markets'] = pd.to_numeric(df_merged['pvp_markets'])

    # Calculate market share and average order value
    df_merged['diff_markets'] = df_merged['units_markets'] / df_merged['units_sales']
    df_merged['aov_markets'] = df_merged['pvp_markets'] / df_merged['units_markets']

    # Merge the result with weekly data
    df_markets_week = pd.merge(df[['year', 'month', 'week']], df_merged[['year', 'month', 'diff_markets', 'aov_markets']], on=['year', 'month'], how='left')


     # Find the maximum year and week in the DataFrame
    max_year = df_markets_week['year'].max()
    max_week = df_markets_week[df_markets_week['year'] == max_year]['week'].max()
    
    # Convert max_year and max_week to a datetime object representing the start of the next week
    max_date = datetime.strptime(f'{max_year}-W{int(max_week)}-1', "%Y-W%W-%w") + timedelta(weeks=1)
    
    # Calculate the future date which is today plus four weeks
    today = max_date
    future_date = today + timedelta(weeks=3) # Tool forecast options (maximum 4 weeks in the future)

    df_last_weeks = df_markets_week[(df_markets_week['year'] == max_year) | (df_markets_week['year'] == max_year - 1)].nlargest(4, 'week')

    mean_markets = round(df_last_weeks['diff_markets'].mean(), 2)
    mean_aov_markets = round(df_last_weeks['aov_markets'].mean(), 2)

    # Generate the weeks between the maximum date and the future date
    current_date = max_date
    new_rows = []
    while current_date <= future_date:
        year, week, _ = current_date.isocalendar()
        month = current_date.month
        new_row = {'year': year, 'week': week, 'month': month, 'diff_markets': mean_markets, 'aov_markets': mean_aov_markets}
        new_rows.append(new_row)
        current_date += timedelta(weeks=1)
    
    df_future_markets = pd.DataFrame(new_rows)

    df_markets_week = pd.concat([df_markets_week, df_future_markets], ignore_index=True)
    df_markets_week['diff_markets'] = df_markets_week['diff_markets'].fillna(0)
    df_markets_week['aov_markets'] = df_markets_week['aov_markets'].fillna(0)

    return df_markets_week

def create_table_predict_units(df, df_coef, updated_list_bucket, frequency):
    """
    Creates a table predicting units based on investment and coefficients for the given frequency.

    Args:
    df (pd.DataFrame): Input DataFrame containing historical data, including investment and units.
    df_coef (pd.DataFrame): DataFrame containing coefficients for each channel.
    updated_list_bucket (list): List of channel names to include in the analysis.
    frequency (str): Frequency of the analysis, e.g., 'year', 'month', or 'week'.

    Returns:
    pd.DataFrame: Transformed DataFrame with predicted units for each channel and an additional row 
                  for "Other factors".
    """

    # Filter and reshape the DataFrame to include frequency and channel-specific investments
    df_period = df[[frequency] + updated_list_bucket]
    df_period = pd.melt(
        df_period,
        id_vars=[frequency],  # Retain the frequency column as is
        var_name='channel',   # Create a column for channel names
        value_name='investment'  # Create a column for investment values
    )

    # Filter data for the current year, month, or week depending on the frequency
    df_period = df_period[df_period[frequency] == datetime.now().year]
    current_period = datetime.now().year
    if frequency == 'month':
        df_period = df_period[df_period[frequency] == datetime.now().month]
        current_period = datetime.now().month
    if frequency == 'week':
        df_period = df_period[df_period[frequency] == datetime.now().isocalendar().week]
        current_period = datetime.now().isocalendar().week

    # Aggregate investments by frequency and channel
    df_period = df_period.groupby([frequency, 'channel']).agg({'investment': 'sum'}).reset_index()

    # Modify coefficient DataFrame by standardizing channel names and retaining relevant columns
    df_coef_mod = df_coef
    df_coef_mod['channel'] = df_coef['channel'].str.replace('-adstock', '', regex=False)
    df_coef_mod = df_coef_mod[['channel', 'coefficient']]

    # Merge the investment data with coefficients
    df_period = pd.merge(df_period, df_coef_mod, on='channel', how='left')

    # Calculate predicted units based on investments and coefficients
    df_period['predict_units'] = np.where(df_period['coefficient'] > 0, 
                                          df_period['investment'] * df_period['coefficient'], 
                                          0)

    # Filter the original DataFrame to include data for the current period
    df_mod = df
    df_mod = df_mod[df_mod[frequency] == datetime.now().year]
    if frequency == 'month':
        df_mod = df_mod[df_mod[frequency] == datetime.now().month]
    if frequency == 'week':
        df_mod = df_mod[df_mod[frequency] == datetime.now().isocalendar().week]

    # Calculate total units and predicted units
    total_units = df_mod['units'].sum()
    predict_units = df_period['predict_units'].sum()

    # Add a new row for other factors affecting units
    new_row = {
        frequency: current_period,
        'channel': 'Otros factores',
        'predict_units': total_units - predict_units
    }
    df_period = pd.concat([df_period, pd.DataFrame([new_row])], ignore_index=True)

    return df_period

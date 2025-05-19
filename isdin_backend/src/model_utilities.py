from sklearn.model_selection import train_test_split
import statsmodels.api as sm
import pandas as pd
import numpy as np
from math import sqrt
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from datetime import datetime, timedelta
import statsmodels.stats.api as sms
from statsmodels.compat import lzip
from scipy import stats
import statsmodels.stats.diagnostic as dg
import matplotlib.pyplot as plt
import seaborn as sns
import random
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import plot_tree

def train_model(df, list_bucket_adstock, target, features = []):
    """
    Trains a linear regression model using specified features and target variable.

    This function splits the data into training and testing sets, adds a constant term 
    to the training data for the intercept, and fits an Ordinary Least Squares (OLS) 
    regression model.

    Args:
    df (pd.DataFrame): The input DataFrame containing the data.
    list_bucket_adstock (list): List of adstock-transformed column names.
    target (str): The target variable name.
    features (list): List of additional feature names.

    Returns:
    model (statsmodels.regression.linear_model.RegressionResultsWrapper): The trained model.
    X_train (pd.DataFrame): The training features after transformation.
    X_test (pd.DataFrame): The testing features after transformation.
    y_train (np.array): The training target variable.
    y_test (np.array): The testing target variable.
    """

    # Extract the target variable from the DataFrame
    y = df[target]

    # Combine 'year', adstock columns, and other features into the design matrix X
    X = df[['year'] + list_bucket_adstock + features]

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y.values,
        train_size=0.80,      # Use 80% of the data for training
        random_state=1234,    # Seed for reproducibility
        shuffle=True          # Shuffle the data before splitting
    )

    ### Convert from bool to int if necessary
    # Columns to convert to int
    columns_to_convert = ['year'] # + week_columns
    X_train[columns_to_convert] = X_train[columns_to_convert].astype(int)

    # Add a constant term to the design matrix for the intercept
    X_train = sm.add_constant(X_train, prepend=True)

    # Create the Ordinary Least Squares (OLS) regression model
    model = sm.OLS(endog=y_train, exog=X_train)
    model = model.fit()

    return model, X_train, X_test, y_train, y_test


def evaluation_model(model, X_test, y_test):
    """
    Evaluate the performance of a regression model on a test dataset.

    Args:
    model: The trained regression model.
    X_test: The features of the test dataset.
    y_test: The target variable of the test dataset.

    Returns:
    df_evaluation (dict): A dictionary containing evaluation metrics.
    """

    # Add a constant term to the test features
    X_test = sm.add_constant(X_test, has_constant='add')

    # Predict prices of X_test
    y_pred = model.predict(X_test)

    # Concatenate actual and predicted values for comparison
    comparation = pd.concat([pd.DataFrame(y_test), pd.DataFrame(y_pred)], axis=1)

    # Evaluate the model on the test set
    r2 = r2_score(y_test, y_pred)  # R-squared
    mse = mean_squared_error(y_test, y_pred)  # Mean Squared Error (MSE)
    rmse = sqrt(mse)  # Root Mean Squared Error (RMSE)
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100  # Mean Absolute Percentage Error (MAPE)

    # Calculate the standard deviation of the residuals
    standard_deviation = np.std(model.resid)

    # Create a dictionary containing evaluation metrics
    df_evaluation = pd.DataFrame([{
        'model': 'Linear regression',
        'r2': r2,
        'rmse': rmse,
        'mape': mape,
        'standard_deviation': standard_deviation
    }])

    return df_evaluation


def generating_model_coefficients(model, list_bucket_adstock):
    """
    Generate a DataFrame with coefficients, p-values, and confidence intervals for the given model and list_bucket_adstock.

    Args:
        model: Trained model object from which coefficients are extracted.
        list_bucket_adstock (list): List of channel names (updated and with -adstock suffix).

    Returns:
        df_coefficients (DataFrame): DataFrame with coefficients, p-values, lower and upper confidence intervals.
    """

    # Extract coefficients
    df_coefficients = pd.DataFrame(model.params[list_bucket_adstock].items(), columns=['channel', 'coefficient']).sort_values(by='coefficient', ascending=True)
    
    # Add p-values
    df_coefficients['p_value'] = model.pvalues.loc[list_bucket_adstock].values
    
    # Add lower and upper confidence intervals (80%)
    lower_limit_80 = model.conf_int(alpha=0.20).loc[list_bucket_adstock, 0]
    upper_limit_80 = model.conf_int(alpha=0.20).loc[list_bucket_adstock, 1]
    df_coefficients['lower_limit_80'] = lower_limit_80.values
    df_coefficients['upper_limit_80'] = upper_limit_80.values
    
    return df_coefficients

def selection_of_model_to_be_served(model, current_old_model, df_evaluation, df_evaluation_current_old_model, df_models_in_production, X_train, y_train, X_current_train, y_current_train):
    """
    Compare two models based on their R2 scores and return the model with the higher R2 score.

    Args:
        model_weeks: The model trained in the current week.
        current_old_model: The model that is currently in production.
        df_evaluation_weeks (dict): A dictionary containing the evaluation metrics for model_weeks, including 'r2'.
        df_evaluation_current_old_model (dict): A dictionary containing the evaluation metrics (with current validation data)
            for current_old_model, including 'r2'.
        df_models_in_production (pd.DataFrame): A DataFrame with information about the models in production.
        X_weeks_train (pd.DataFrame): The training data for the current week.
        y_weeks_train (np.array): The training target for the current week.
        X_current_train (pd.DataFrame): The training data for the current model.
        y_current_train (np.array): The training target for the current model.

    Returns:
        tuple: Contains the selected model, the evaluation metrics, the model name, the start date, the end date, 
        X_current_train, and y_current_train.
    """
    
    # Retrieve the R2 values of the evaluated models
    r2_1 = df_evaluation['r2'].iloc[0]
    r2_2 = df_evaluation_current_old_model['r2'].iloc[0]
    
    # Get the current date in YYYYMMDD format
    current_date_str = datetime.now().strftime('%Y%m%d')
    
    # Convert the date string to a date object
    current_date_obj = datetime.strptime(current_date_str, '%Y%m%d')
    
    # Calculate the end date by adding 6 days to the current date
    end_date_obj = current_date_obj + timedelta(days=6)
    end_date = end_date_obj.strftime('%Y-%m-%d')
    
    # Compare the R2 values to determine which model to select
    if r2_1 >= r2_2:
        selected_model = model
        selected_evaluation = df_evaluation
        model_name = f"week_{current_date_str}"
        start_date = datetime.now().strftime('%Y-%m-%d')
        X_current_train = X_train
        y_current_train = y_train
    else:
        selected_model = current_old_model
        selected_evaluation = df_evaluation_current_old_model
        model_name = df_models_in_production['model_name'].iloc[-1]
        start_date = df_models_in_production['start_date'].iloc[-1]
        X_current_train = sm.add_constant(X_current_train, prepend=True)

    return selected_model, selected_evaluation, model_name, start_date, end_date, X_current_train, y_current_train


def model_residuals(model, X_train, y_train, file_name):
    """
    Generates diagnostic plots and performs statistical tests on the residuals of a given model.

    Args:
        model: The fitted model object. This model should have the methods `predict` and `resid`.
        X_train: The training input data used for the model.
        y_train: The true values corresponding to the training input data.

    Returns:
        fig: The figure object containing the diagnostic plots.
        tests_results: A dictionary containing the results of the statistical tests:
            - 'Homoscedasticity': Results of the Breusch-Pagan-Godfrey test.
            - 'Normality': Results of the Shapiro-Wilk test.
            - 'Independence': Results of the Durbin-Watson test.
    """
    
    # Generate predictions for the training data
    prediction_train = model.predict(exog=X_train)
    # Get residuals from the model
    residuals_train = model.resid
    
    # Create subplots for diagnostic plots
    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(9, 8))

    # Plot predicted vs actual values
    axes[0, 0].scatter(y_train, prediction_train, edgecolors=(0, 0, 0), alpha=0.4)
    axes[0, 0].plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'k--', lw=2)
    axes[0, 0].set_title('Predicted vs Actual Value', fontsize=10, fontweight="bold")
    axes[0, 0].set_xlabel('Actual')
    axes[0, 0].set_ylabel('Prediction')
    axes[0, 0].tick_params(labelsize=7)

    # Plot residuals as a function of observation index
    axes[0, 1].scatter(list(range(len(y_train))), residuals_train, edgecolors=(0, 0, 0), alpha=0.4)
    axes[0, 1].axhline(y=0, linestyle='--', color='black', lw=2)
    axes[0, 1].set_title('Model Residuals', fontsize=10, fontweight="bold")
    axes[0, 1].set_xlabel('id')
    axes[0, 1].set_ylabel('Residual')
    axes[0, 1].tick_params(labelsize=7)

    # Plot histogram of residuals with density estimation
    sns.histplot(data=residuals_train, stat="density", kde=True, line_kws={'linewidth': 1}, color="firebrick", alpha=0.3, ax=axes[1, 0])
    axes[1, 0].set_title('Distribution of Model Residuals', fontsize=10, fontweight="bold")
    axes[1, 0].set_xlabel("Residual")
    axes[1, 0].tick_params(labelsize=7)

    # Q-Q plot of residuals to check normality
    sm.qqplot(residuals_train, fit=True, line='q', ax=axes[1, 1])
    axes[1, 1].set_title('Q-Q Plot of Model Residuals', fontsize=10, fontweight="bold")
    axes[1, 1].tick_params(labelsize=7)

    # Plot residuals vs predictions to check for patterns
    axes[2, 0].scatter(prediction_train, residuals_train, edgecolors=(0, 0, 0), alpha=0.4)
    axes[2, 0].axhline(y=0, linestyle='--', color='black', lw=2)
    axes[2, 0].set_title('Residuals vs Prediction', fontsize=10, fontweight="bold")
    axes[2, 0].set_xlabel('Prediction')
    axes[2, 0].set_ylabel('Residual')
    axes[2, 0].tick_params(labelsize=7)

    # Remove the unused subplot
    fig.delaxes(axes[2, 1])

    # Adjust layout and save the figure
    fig.tight_layout()
    plt.subplots_adjust(top=0.9)
    fig.suptitle('Residuals Diagnostic', fontsize=12, fontweight="bold")
    fig.savefig(file_name, format='png', dpi=300)
    
    # Perform Breusch-Pagan-Godfrey test for homoscedasticity
    bp_test = sms.het_breuschpagan(residuals_train, X_train)
    bp_test_results = {
        'Breusch-Pagan-Godfrey test': {
            'Lagrange multiplier statistic': round(bp_test[0], 4),
            'p-value': round(bp_test[1], 4),
            'f-value': round(bp_test[2], 4),
            'f p-value': round(bp_test[3], 4)
        }
    }

    # Perform Shapiro-Wilk test for normality
    shapiro_test = stats.shapiro(residuals_train)
    shapiro_test_results = {
        'Shapiro-Wilk test': {
            'Statistic': round(shapiro_test.statistic, 4),
            'P Value': round(shapiro_test.pvalue, 4)
        }
    }

    # Perform Durbin-Watson test for independence of residuals
    durbin_watson_test = sm.stats.durbin_watson(residuals_train)
    durbin_watson_results = {
        'Durbin-Watson test': round(durbin_watson_test, 4)
    }

    # Compile test results into a dictionary
    tests_results = {
        'Homoscedasticity': bp_test_results,
        'Normality': shapiro_test_results,
        'Independence': durbin_watson_results
    }

    return fig, tests_results


def stepwise_backward(X, y, significance_threshold=0.05):
    """
    Performs backward stepwise regression to select features based on p-values.

    Args:
        X: DataFrame containing the input features.
        y: Series or array containing the target variable.
        significance_threshold: The p-value threshold for including a feature in the model.

    Returns:
        model: The final fitted OLS model after feature selection.
        included: List of the variables included in the final model.
    """
    
    # Start with all the features in the dataset
    included = list(X.columns)
    
    while True:
        changed = False
        
        # Fit an Ordinary Least Squares (OLS) regression model using the included features
        model = sm.OLS(endog=y, exog=X[included])
        model = model.fit()  # Fit the model to the data
        
        # Retrieve the p-values for each feature in the model
        pvalues = model.pvalues
        
        # Identify the feature with the highest p-value
        max_pvalue = pvalues.idxmax()
        
        # If the feature with the highest p-value has a p-value greater than the significance threshold,
        # it means this feature is not statistically significant and should be removed
        if pvalues[max_pvalue] > significance_threshold:
            included.remove(max_pvalue)  # Remove the feature from the list of included features
            changed = True
        
        # If no features were removed in this iteration, the feature selection process is complete
        if not changed:
            break

    # If the constant term ('const') was included in the model, remove it from the final list of features
    if 'const' in included:
        included.remove('const')
    
    return model, included


def cross_validation_evaluation(df, list_bucket_adstock, target, features):

    """
    Cross-Validation: Trains a linear regression model multiple times and evaluates its performance.

    Args:
    df (pd.DataFrame): DataFrame containing the data.
    list_bucket_adstock (list): List of adstock feature column names.
    target (str): Name of the target variable column.
    features (list): List of other feature column names.

    Returns:
    pd.DataFrame: DataFrame containing the mean and median of R2, MAPE, and RMSE metrics.
    """

    # Lists to store evaluation metrics
    r2_results = []
    mse_results = []
    rmse_results = []
    mape_results = []

    # Extract the target variable from the DataFrame
    y = df[target]

    # Combine 'year', adstock columns, and other features into the design matrix X
    X = df[['year'] + list_bucket_adstock + features]

    i = 0
    while i < 100:
        # Generate a random integer for the random state
        random_int = random.randint(1, 10000)

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y.values,
            train_size=0.80,
            random_state=random_int,
            shuffle=True
        )

        ### Convert from bool to int if necessary
        # week_columns = [col for col in df.columns if col.startswith('week_')]
        columns_to_convert = ['year'] # + week_columns
        X_train[columns_to_convert] = X_train[columns_to_convert].astype(int)

        # Add a constant column to the predictors matrix for the intercept
        X_train = sm.add_constant(X_train, prepend=True)
        model = sm.OLS(endog=y_train, exog=X_train)
        model = model.fit()

        # Add a constant column to the test predictors matrix
        X_test = sm.add_constant(X_test, has_constant='add')

        # Predict the target variable for the test set
        y_pred = model.predict(X_test)

        # Evaluate the model on the test set
        r2 = r2_score(y_test, y_pred)
        r2_results.append(r2)

        # Calculate the Mean Squared Error (MSE)
        mse = mean_squared_error(y_test, y_pred)
        mse_results.append(mse)

        # Calculate the Root Mean Squared Error (RMSE)
        rmse = sqrt(mse)
        rmse_results.append(rmse)

        # Calculate the Mean Absolute Percentage Error (MAPE)
        mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
        mape_results.append(mape)

        i += 1

    # Calculate the mean and median for R2 results
    mean_r2 = np.mean(r2_results)
    median_r2 = np.median(r2_results)

    # Calculate the mean and median for MAPE results
    mean_mape = np.mean(mape_results)
    median_mape = np.median(mape_results)

    # Calculate the mean and median for RMSE results
    mean_rmse = np.mean(rmse_results)
    median_rmse = np.median(rmse_results)

    # Combine the results into a DataFrame
    results_table = pd.DataFrame({
        'Metric': ['R2', 'MAPE', 'RMSE'],
        'Mean': [mean_r2, mean_mape, mean_rmse],
        'Median': [median_r2, median_mape, median_rmse]
    })

    return results_table

def train_model_random_forest(df, list_bucket_adstock, target, features = []):

    # Extract the target variable from the DataFrame
    y = df[target]

    # Combine 'year', adstock columns, and other features into the design matrix X
    X = df[['year'] + list_bucket_adstock + features]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 1234)

    # Train the model
    rf = RandomForestRegressor(n_estimators = 200, max_depth = 10, max_features=1.0,  
                               min_samples_leaf = 2, min_samples_split = 5, random_state = 1234)
    rf.fit(X_train, y_train.values.ravel())



    return rf, X_train, X_test, y_train, y_test

def evaluation_model_random_forest(model, X_test, y_test):
    """
    Evaluate the performance of a regression model on a test dataset.

    Args:
    model: The trained regression model.
    X_test: The features of the test dataset.
    y_test: The target variable of the test dataset.

    Returns:
    df_evaluation (dict): A dictionary containing evaluation metrics.
    """

    feature_order = model.feature_names_in_
    X_test = X_test[feature_order]

    # Predict prices of X_test
    y_pred = model.predict(X_test)


    # Evaluate the model on the test set
    r2 = r2_score(y_test, y_pred)  # R-squared
    mse = mean_squared_error(y_test, y_pred)  # Mean Squared Error (MSE)
    rmse = sqrt(mse)  # Root Mean Squared Error (RMSE)
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100  # Mean Absolute Percentage Error (MAPE)

    # Calculate the standard deviation of the residuals
    # standard_deviation = np.std(model.resid)

    # Create a dictionary containing evaluation metrics
    df_evaluation = pd.DataFrame([{
        'model': 'Random forest',
        'r2': r2,
        'rmse': rmse,
        'mape': mape,
        #'standard_deviation': standard_deviation
    }])

    importancias_caracteristicas = model.feature_importances_


    importancias_df = pd.DataFrame({
        'feature': X_test.columns,
        'feature_importances': importancias_caracteristicas
    })
    importancias_df = importancias_df.sort_values(by='feature_importances', ascending=False)


    return df_evaluation, importancias_df

def cross_validation_evaluation_random_forest(df, list_bucket_adstock, target, features):

    """
    Cross-Validation: Trains a linear regression model multiple times and evaluates its performance.

    Args:
    df (pd.DataFrame): DataFrame containing the data.
    list_bucket_adstock (list): List of adstock feature column names.
    target (str): Name of the target variable column.
    features (list): List of other feature column names.

    Returns:
    pd.DataFrame: DataFrame containing the mean and median of R2, MAPE, and RMSE metrics.
    """

    # Lists to store evaluation metrics
    r2_results = []
    mse_results = []
    rmse_results = []
    mape_results = []

    # Extract the target variable from the DataFrame
    y = df[target]

    # Combine 'year', adstock columns, and other features into the design matrix X
    X = df[['year'] + list_bucket_adstock + features]

    i = 0
    while i < 100:
        # Generate a random integer for the random state
        random_int = random.randint(1, 10000)

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = random_int)

        # Add a constant column to the predictors matrix for the intercept
        rf = RandomForestRegressor(n_estimators = 200, max_depth = 10, max_features=1.0,  
                               min_samples_leaf = 2, min_samples_split = 5, random_state = 1234)
        rf.fit(X_train, y_train.values.ravel())

        # Predict the target variable for the test set
        y_pred = rf.predict(X_test)

        # Evaluate the model on the test set
        r2 = r2_score(y_test, y_pred)
        r2_results.append(r2)

        # Calculate the Mean Squared Error (MSE)
        mse = mean_squared_error(y_test, y_pred)
        mse_results.append(mse)

        # Calculate the Root Mean Squared Error (RMSE)
        rmse = sqrt(mse)
        rmse_results.append(rmse)

        # Calculate the Mean Absolute Percentage Error (MAPE)
        mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
        mape_results.append(mape)

        i += 1

    # Calculate the mean and median for R2 results
    mean_r2 = np.mean(r2_results)
    median_r2 = np.median(r2_results)

    # Calculate the mean and median for MAPE results
    mean_mape = np.mean(mape_results)
    median_mape = np.median(mape_results)

    # Calculate the mean and median for RMSE results
    mean_rmse = np.mean(rmse_results)
    median_rmse = np.median(rmse_results)

    # Combine the results into a DataFrame
    results_table = pd.DataFrame({
        'Metric': ['R2', 'MAPE', 'RMSE'],
        'Mean': [mean_r2, mean_mape, mean_rmse],
        'Median': [median_r2, median_mape, median_rmse]
    })

    return results_table

def analyze_best_tree(model_rf, X_rf_test, y_rf_test, feature_names, file_name):
    """
    Analyzes the best-performing tree in a random forest model based on the mean squared error (MSE).
    
    This function identifies the tree with the lowest MSE from a random forest model, extracts node 
    information for that tree, and generates a plot of the best tree's structure, saving it as a 
    variable.

    Parameters:
        model_rf (RandomForestRegressor): The random forest model containing multiple decision trees.
        X_rf_test (DataFrame or array-like): The feature set used for validation.
        y_rf_test (Series or array-like): The target variable used for validation.
        feature_names (list): List of feature names for interpreting tree structure.

    Returns:
        tuple: Contains two elements:
            - nodes_df (DataFrame): DataFrame with node information (feature index, threshold, 
              target mean, left and right child).
            - tree_plot (Figure): Matplotlib figure of the best tree's structure.
    """
    # Calculate MSE for each tree in the forest
    errors = [(i, mean_squared_error(y_rf_test, tree.predict(X_rf_test)))
              for i, tree in enumerate(model_rf.estimators_)]

    # Identify the tree with the lowest MSE
    best_tree_index = min(errors, key=lambda x: x[1])[0]
    best_tree = model_rf.estimators_[best_tree_index]
    #print(f"The best tree is at index {best_tree_index} with MSE of {errors[best_tree_index][1]}")
     # Calculate R² for the best tree
    # y_pred_best_tree = best_tree.predict(X_rf_test)
    # best_tree_r2 = r2_score(y_rf_test, y_pred_best_tree)
    # print(f"The R² score for the best tree is {best_tree_r2}")

    # Extract node details from the best tree
    tree = best_tree.tree_
    nodes_info = {
        "node": list(range(tree.node_count)),
        "feature index": tree.feature,
        "threshold value": tree.threshold,
        "target mean value": [tree.value[i][0][0] for i in range(tree.node_count)],
        "left child node": tree.children_left,
        "right child node": tree.children_right,
    }
    
    nodes_df = pd.DataFrame(nodes_info)
    nodes_df["feature name"] = nodes_df["feature index"].apply(lambda x: feature_names[x] if x != -2 else "leaf")

    # Plot the best tree
    fig, ax = plt.subplots(figsize=(20, 10))
    plot_tree(best_tree, filled=True, feature_names=feature_names, ax=ax)
    fig.savefig(file_name, format='png', dpi=300)
    plt.close(fig)  # Close to avoid automatic display in notebooks

    return nodes_df, fig
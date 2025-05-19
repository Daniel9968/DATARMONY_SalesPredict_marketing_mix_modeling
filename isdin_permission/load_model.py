from google.cloud import storage
import pandas as pd
import pickle
import joblib
import matplotlib
import matplotlib.pyplot as plt
from google.oauth2 import service_account
import json
import os
from src.cloud import cloudToLocal, get_latest_cloud_folder_name


bucket_name = os.getenv("CLOUD_STORAGE_BUCKET")



latest_folder_name = get_latest_cloud_folder_name(bucket_name)



### foto_coefficients_weather.csv
cloudToLocal(bucket_name, f'{latest_folder_name}/foto_coefficients_weather.csv', './foto_coefficients_weather.csv')
# Save the file in a variable as csv format
with open('./foto_coefficients_weather.csv', 'r', encoding='utf-8') as f:
    foto_coefficients_weather = pd.read_csv(f)

# Delete the file
os.remove('./foto_coefficients_weather.csv')



### foto_df_coef_model_current.csv
cloudToLocal(bucket_name, 'Current_model/foto_df_coef_model_current.csv', './foto_df_coef_model_current.csv')
# Save the file in a variable as csv format
with open('./foto_df_coef_model_current.csv', 'r', encoding='utf-8') as f:
    foto_df_coef_model_current = pd.read_csv(f)

# Delete the file
os.remove('./foto_df_coef_model_current.csv')



### foto_fig_residuals_diagnostic_current.png 
cloudToLocal(bucket_name, 'Current_model/foto_fig_residuals_diagnostic_current.png', './foto_fig_residuals_diagnostic_current.png')
with open('./foto_fig_residuals_diagnostic_current.png', 'rb') as f:
    foto_fig_residuals_diagnostic_current = f.read()

# Delete the file
os.remove('./foto_fig_residuals_diagnostic_current.png')



### foto_model_residual_tests_current.pkl 
cloudToLocal(bucket_name, 'Current_model/foto_model_residual_tests_current.pkl', './foto_model_residual_tests_current.pkl')
with open('./foto_model_residual_tests_current.pkl', 'rb') as f:
    foto_model_residual_tests_current = f.read()

# Delete the file
os.remove('./foto_model_residual_tests_current.pkl')



### foto_predict_table_current.csv
cloudToLocal(bucket_name, 'Current_model/foto_predict_table_current.csv', './foto_predict_table_current.csv')
# Save the file in a variable as csv format
with open('./foto_predict_table_current.csv', 'r', encoding='utf-8') as f:
    foto_predict_table_current = pd.read_csv(f)

# Delete the file
os.remove('./foto_predict_table_current.csv')



### foto_predict_table_months.csv
cloudToLocal(bucket_name, f'{latest_folder_name}/foto_predict_table_months.csv', './foto_predict_table_months.csv')
# Save the file in a variable as csv format
with open('./foto_predict_table_months.csv', 'r', encoding='utf-8') as f:
    foto_predict_table_months = pd.read_csv(f)

# Delete the file
os.remove('./foto_predict_table_months.csv')



### foto_predict_table_weather.csv
cloudToLocal(bucket_name, f'{latest_folder_name}/foto_predict_table_weather.csv', './foto_predict_table_weather.csv')
# Save the file in a variable as csv format
with open('./foto_predict_table_weather.csv', 'r', encoding='utf-8') as f:
    foto_predict_table_weather = pd.read_csv(f)

# Delete the file
os.remove('./foto_predict_table_weather.csv')



### foto_weather_data_ccaa.csv
cloudToLocal(bucket_name, f'{latest_folder_name}/foto_weather_data_ccaa.csv', './foto_weather_data_ccaa.csv')
# Save the file in a variable as csv format
with open('./foto_weather_data_ccaa.csv', 'r', encoding='utf-8') as f:
    foto_weather_data_ccaa = pd.read_csv(f)

# Delete the file
os.remove('./foto_weather_data_ccaa.csv')









### ceutics_coefficients_regression.csv
cloudToLocal(bucket_name, f'{latest_folder_name}/ceutics_coefficients_regression.csv', './ceutics_coefficients_regression.csv')
# Save the file in a variable as csv format
with open('./ceutics_coefficients_regression.csv', 'r', encoding='utf-8') as f:
    ceutics_coefficients_regression = pd.read_csv(f)

# Delete the file
os.remove('./ceutics_coefficients_regression.csv')



### ceutics_importances_rf.csv
cloudToLocal(bucket_name, f'{latest_folder_name}/ceutics_importances_rf.csv', './ceutics_importances_rf.csv')
# Save the file in a variable as csv format
with open('./ceutics_importances_rf.csv', 'r', encoding='utf-8') as f:
    ceutics_importances_rf = pd.read_csv(f)

# Delete the file
os.remove('./ceutics_importances_rf.csv')



### ceutics_fig_residuals_diagnostic_current_regression.png 
cloudToLocal(bucket_name, 'Current_model/ceutics_fig_residuals_diagnostic_current_regression.png', './ceutics_fig_residuals_diagnostic_current_regression.png')
with open('./ceutics_fig_residuals_diagnostic_current_regression.png', 'rb') as f:
    ceutics_fig_residuals_diagnostic_current_regression = f.read()

# Delete the file
os.remove('./ceutics_fig_residuals_diagnostic_current_regression.png')



### ceutics_model_residual_tests_current_regression.pkl 
cloudToLocal(bucket_name, 'Current_model/ceutics_model_residual_tests_current_regression.pkl', './ceutics_model_residual_tests_current_regression.pkl')
with open('./ceutics_model_residual_tests_current_regression.pkl', 'rb') as f:
    ceutics_model_residual_tests_current_regression = f.read()

# Delete the file
os.remove('./ceutics_model_residual_tests_current_regression.pkl')



### predict_table_current.csv
cloudToLocal(bucket_name, 'Current_model/ceutics_predict_table_current.csv', './ceutics_predict_table_current.csv')
# Save the file in a variable as csv format
with open('./ceutics_predict_table_current.csv', 'r', encoding='utf-8') as f:
    ceutics_predict_table_current = pd.read_csv(f)

# Delete the file
os.remove('./ceutics_predict_table_current.csv')



### ceutics_predict_table_rf.csv
cloudToLocal(bucket_name, f'{latest_folder_name}/ceutics_predict_table_rf.csv', './ceutics_predict_table_rf.csv')
# Save the file in a variable as csv format
with open('./ceutics_predict_table_rf.csv', 'r', encoding='utf-8') as f:
    ceutics_predict_table_rf = pd.read_csv(f)

# Delete the file
os.remove('./ceutics_predict_table_rf.csv')



### ceutics_df_nodes_best_tree.csv
cloudToLocal(bucket_name, 'Current_model/ceutics_df_nodes_best_tree.csv', './ceutics_df_nodes_best_tree.csv')
# Save the file in a variable as csv format
with open('./ceutics_df_nodes_best_tree.csv', 'r', encoding='utf-8') as f:
    ceutics_df_nodes_best_tree = pd.read_csv(f)

# Delete the file
os.remove('./ceutics_df_nodes_best_tree.csv')



### ceutics_fig_best_tree.png 
cloudToLocal(bucket_name, 'Current_model/ceutics_fig_best_tree.png', './ceutics_fig_best_tree.png')
with open('./ceutics_fig_best_tree.png', 'rb') as f:
    ceutics_fig_best_tree = f.read()

# Delete the file
os.remove('./ceutics_fig_best_tree.png')



### ceutics_df_importances_and_coef.csv
cloudToLocal(bucket_name, 'Current_model/ceutics_df_importances_and_coef.csv', './ceutics_df_importances_and_coef.csv')
# Save the file in a variable as csv format
with open('./ceutics_df_importances_and_coef.csv', 'r', encoding='utf-8') as f:
    ceutics_df_importances_and_coef = pd.read_csv(f)

# Delete the file
os.remove('./ceutics_df_importances_and_coef.csv')









### derma_acniben_df_coef_model_current.csv
cloudToLocal(bucket_name, 'Current_model/derma_acniben_df_coef_model_current.csv', './derma_acniben_df_coef_model_current.csv')
# Save the file in a variable as csv format
with open('./derma_acniben_df_coef_model_current.csv', 'r', encoding='utf-8') as f:
    derma_acniben_df_coef_model_current = pd.read_csv(f)

# Delete the file
os.remove('./derma_acniben_df_coef_model_current.csv')



### derma_acniben_fig_residuals_diagnostic_current.png 
cloudToLocal(bucket_name, 'Current_model/derma_acniben_fig_residuals_diagnostic_current.png', './derma_acniben_fig_residuals_diagnostic_current.png')
with open('./derma_acniben_fig_residuals_diagnostic_current.png', 'rb') as f:
    derma_acniben_fig_residuals_diagnostic_current = f.read()

# Delete the file
os.remove('./derma_acniben_fig_residuals_diagnostic_current.png')



### derma_acniben_model_residual_tests_current.pkl 
cloudToLocal(bucket_name, 'Current_model/derma_acniben_model_residual_tests_current.pkl', './derma_acniben_model_residual_tests_current.pkl')
with open('./derma_acniben_model_residual_tests_current.pkl', 'rb') as f:
    derma_acniben_model_residual_tests_current = f.read()

# Delete the file
os.remove('./derma_acniben_model_residual_tests_current.pkl')



### derma_acniben_predict_table_current.csv
cloudToLocal(bucket_name, 'Current_model/derma_acniben_predict_table_current.csv', './derma_acniben_predict_table_current.csv')
# Save the file in a variable as csv format
with open('./derma_acniben_predict_table_current.csv', 'r', encoding='utf-8') as f:
    derma_acniben_predict_table_current = pd.read_csv(f)

# Delete the file
os.remove('./derma_acniben_predict_table_current.csv')



### derma_acniben_predict_table_months.csv
cloudToLocal(bucket_name, f'{latest_folder_name}/derma_acniben_predict_table_months.csv', './derma_acniben_predict_table_months.csv')
# Save the file in a variable as csv format
with open('./derma_acniben_predict_table_months.csv', 'r', encoding='utf-8') as f:
    derma_acniben_predict_table_months = pd.read_csv(f)

# Delete the file
os.remove('./derma_acniben_predict_table_months.csv')


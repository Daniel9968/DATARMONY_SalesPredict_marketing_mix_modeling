from google.cloud import storage
import pandas as pd
import pickle
from google.oauth2 import service_account
import json
import os
from src.cloud import cloudToLocal, get_latest_cloud_folder_name


bucket_name = os.getenv("CLOUD_STORAGE_BUCKET")



latest_folder_name = get_latest_cloud_folder_name(bucket_name)


##### Foto #####

### foto_df_final_current.csv
cloudToLocal(bucket_name, 'Current_model/foto_df_final_current.csv', './foto_df_final_current.csv')
# Save the file in a variable as csv format
with open('./foto_df_final_current.csv', 'r', encoding='utf-8') as f:
    foto_df_final = pd.read_csv(f)

# Delete the file
os.remove('./foto_df_final_current.csv')



### foto_df_final_weather.csv
cloudToLocal(bucket_name, f'{latest_folder_name}/foto_df_final_weather.csv', './foto_df_final_weather.csv')
# Save the file in a variable as csv format
with open('./foto_df_final_weather.csv', 'r', encoding='utf-8') as f:
    foto_df_final_weather = pd.read_csv(f)

# Delete the file
os.remove('./foto_df_final_weather.csv')



### foto_current_model.pkl
cloudToLocal(bucket_name, 'Current_model/foto_current_model.pkl', './foto_current_model.pkl')
# Save the file in a variable as pkl format
with open('./foto_current_model.pkl', 'rb') as f:
    foto_current_model = pickle.load(f)

# Delete the file
os.remove('./foto_current_model.pkl')



### foto_evaluation_current_model.csv
cloudToLocal(bucket_name, 'Current_model/foto_evaluation_current_model.csv', './foto_evaluation_current_model.csv')
# Save the file in a variable as csv format
with open('./foto_evaluation_current_model.csv', 'r', encoding='utf-8') as f:
    foto_evaluation_current_model = pd.read_csv(f)

# Delete the file
os.remove('./foto_evaluation_current_model.csv')

### foto_evaluation_weather.csv
cloudToLocal(bucket_name, f'{latest_folder_name}/foto_evaluation_weather.csv', './foto_evaluation_weather.csv')
# Save the file in a variable as csv format
with open('./foto_evaluation_weather.csv', 'r', encoding='utf-8') as f:
    foto_evaluation_weather = pd.read_csv(f)

# Delete the file
os.remove('./foto_evaluation_weather.csv')



### foto_saturation.csv
cloudToLocal(bucket_name, f'{latest_folder_name}/foto_saturation.csv', './foto_saturation.csv')
# Save the file in a variable as csv format
with open('./foto_saturation.csv', 'r', encoding='utf-8') as f:
    foto_saturation = pd.read_csv(f)

# Delete the file
os.remove('./foto_saturation.csv')



### foto_predict_table_current.csv
cloudToLocal(bucket_name, 'Current_model/foto_predict_table_current.csv', './foto_predict_table_current.csv')
# Save the file in a variable as csv format
with open('./foto_predict_table_current.csv', 'r', encoding='utf-8') as f:
    foto_predict_table_weeks = pd.read_csv(f)

# Delete the file
os.remove('./foto_predict_table_current.csv')



### foto_predict_table_weather.csv
cloudToLocal(bucket_name, f'{latest_folder_name}/foto_predict_table_weather.csv', './foto_predict_table_weather.csv')
# Save the file in a variable as csv format
with open('./foto_predict_table_weather.csv', 'r', encoding='utf-8') as f:
    foto_predict_table_weather = pd.read_csv(f)

# Delete the file
os.remove('./foto_predict_table_weather.csv')



### foto_model_weather.pkl
cloudToLocal(bucket_name, f'{latest_folder_name}/foto_model_weather.pkl', './foto_model_weather.pkl')
# Save the file in a variable as pkl format
with open('./foto_model_weather.pkl', 'rb') as f:
    foto_model_weather = pickle.load(f)

# Delete the file
os.remove('./foto_model_weather.pkl')



### foto_df_aov.csv
cloudToLocal(bucket_name, f'{latest_folder_name}/foto_df_aov.csv', './foto_df_aov.csv')
# Save the file in a variable as csv format
with open('./foto_df_aov.csv', 'r', encoding='utf-8') as f:
    foto_df_aov = pd.read_csv(f)

# Delete the file
os.remove('./foto_df_aov.csv')



### foto_df_markets_week.csv
cloudToLocal(bucket_name, f'{latest_folder_name}/foto_df_markets_week.csv', './foto_df_markets_week.csv')
# Save the file in a variable as csv format
with open('./foto_df_markets_week.csv', 'r', encoding='utf-8') as f:
    foto_df_markets_week = pd.read_csv(f)

# Delete the file
os.remove('./foto_df_markets_week.csv')



### foto_df_year.csv
cloudToLocal(bucket_name, 'Current_model/foto_df_year.csv', './foto_df_year.csv')
# Save the file in a variable as csv format
with open('./foto_df_year.csv', 'r', encoding='utf-8') as f:
    foto_df_year = pd.read_csv(f)

# Delete the file
os.remove('./foto_df_year.csv')



### foto_df_month.csv
cloudToLocal(bucket_name, 'Current_model/foto_df_month.csv', './foto_df_month.csv')
# Save the file in a variable as csv format
with open('./foto_df_month.csv', 'r', encoding='utf-8') as f:
    foto_df_month = pd.read_csv(f)

# Delete the file
os.remove('./foto_df_month.csv')



### foto_df_week.csv
cloudToLocal(bucket_name, 'Current_model/foto_df_week.csv', './foto_df_week.csv')
# Save the file in a variable as csv format
with open('./foto_df_week.csv', 'r', encoding='utf-8') as f:
    foto_df_week = pd.read_csv(f)

# Delete the file
os.remove('./foto_df_week.csv')









##### Ceutics #####

### ceutics_df_final_current.csv
cloudToLocal(bucket_name, 'Current_model/ceutics_df_final_current.csv', './ceutics_df_final_current.csv')
# Save the file in a variable as csv format
with open('./ceutics_df_final_current.csv', 'r', encoding='utf-8') as f:
    ceutics_df_final = pd.read_csv(f)

# Delete the file
os.remove('./ceutics_df_final_current.csv')



### ceutics_current_model_rf.pkl
cloudToLocal(bucket_name, 'Current_model/ceutics_current_model_rf.pkl', './ceutics_current_model_rf.pkl')
# Save the file in a variable as pkl format
with open('./ceutics_current_model_rf.pkl', 'rb') as f:
    ceutics_current_model = pickle.load(f)

# Delete the file
os.remove('./ceutics_current_model_rf.pkl')



### ceutics_evaluation_current_model_rf.csv
cloudToLocal(bucket_name, 'Current_model/ceutics_evaluation_current_model_rf.csv', './ceutics_evaluation_current_model_rf.csv')
# Save the file in a variable as csv format
with open('./ceutics_evaluation_current_model_rf.csv', 'r', encoding='utf-8') as f:
    ceutics_evaluation_current_model = pd.read_csv(f)

# Delete the file
os.remove('./ceutics_evaluation_current_model_rf.csv')



### ceutics_saturation.csv
cloudToLocal(bucket_name, f'{latest_folder_name}/ceutics_saturation.csv', './ceutics_saturation.csv')
# Save the file in a variable as csv format
with open('./ceutics_saturation.csv', 'r', encoding='utf-8') as f:
    ceutics_saturation = pd.read_csv(f)

# Delete the file
os.remove('./ceutics_saturation.csv')



### ceutics_predict_table_current.csv
cloudToLocal(bucket_name, 'Current_model/ceutics_predict_table_current.csv', './ceutics_predict_table_current.csv')
# Save the file in a variable as csv format
with open('./ceutics_predict_table_current.csv', 'r', encoding='utf-8') as f:
    ceutics_predict_table = pd.read_csv(f)

# Delete the file
os.remove('./ceutics_predict_table_current.csv')



### ceutics_df_aov.csv
cloudToLocal(bucket_name, f'{latest_folder_name}/ceutics_df_aov.csv', './ceutics_df_aov.csv')
# Save the file in a variable as csv format
with open('./ceutics_df_aov.csv', 'r', encoding='utf-8') as f:
    ceutics_df_aov = pd.read_csv(f)

# Delete the file
os.remove('./ceutics_df_aov.csv')



### ceutics_df_markets_week.csv
cloudToLocal(bucket_name, f'{latest_folder_name}/ceutics_df_markets_week.csv', './ceutics_df_markets_week.csv')
# Save the file in a variable as csv format
with open('./ceutics_df_markets_week.csv', 'r', encoding='utf-8') as f:
    ceutics_df_markets_week = pd.read_csv(f)

# Delete the file
os.remove('./ceutics_df_markets_week.csv')



### ceutics_fig_best_tree.pkl
cloudToLocal(bucket_name, 'Current_model/ceutics_fig_best_tree.pkl', './ceutics_fig_best_tree.pkl')
# Save the file in a variable as pkl format
with open('./ceutics_fig_best_tree.pkl', 'rb') as f:
    ceutics_fig_best_tree = pickle.load(f)

# Delete the file
os.remove('./ceutics_fig_best_tree.pkl')



### ceutics_df_year.csv
cloudToLocal(bucket_name, 'Current_model/ceutics_df_year.csv', './ceutics_df_year.csv')
# Save the file in a variable as csv format
with open('./ceutics_df_year.csv', 'r', encoding='utf-8') as f:
    ceutics_df_year = pd.read_csv(f)

# Delete the file
os.remove('./ceutics_df_year.csv')



### ceutics_df_month.csv
cloudToLocal(bucket_name, 'Current_model/ceutics_df_month.csv', './ceutics_df_month.csv')
# Save the file in a variable as csv format
with open('./ceutics_df_month.csv', 'r', encoding='utf-8') as f:
    ceutics_df_month = pd.read_csv(f)

# Delete the file
os.remove('./ceutics_df_month.csv')



### ceutics_df_week.csv
cloudToLocal(bucket_name, 'Current_model/ceutics_df_week.csv', './ceutics_df_week.csv')
# Save the file in a variable as csv format
with open('./ceutics_df_week.csv', 'r', encoding='utf-8') as f:
    ceutics_df_week = pd.read_csv(f)

# Delete the file
os.remove('./ceutics_df_week.csv')









##### Derma Acniben #####

### derma_acniben_df_final_current.csv
cloudToLocal(bucket_name, 'Current_model/derma_acniben_df_final_current.csv', './derma_acniben_df_final_current.csv')
# Save the file in a variable as csv format
with open('./derma_acniben_df_final_current.csv', 'r', encoding='utf-8') as f:
    derma_acniben_df_final = pd.read_csv(f)

# Delete the file
os.remove('./derma_acniben_df_final_current.csv')



### derma_acniben_current_model.pkl
cloudToLocal(bucket_name, 'Current_model/derma_acniben_current_model.pkl', './derma_acniben_current_model.pkl')
# Save the file in a variable as pkl format
with open('./derma_acniben_current_model.pkl', 'rb') as f:
    derma_acniben_current_model = pickle.load(f)

# Delete the file
os.remove('./derma_acniben_current_model.pkl')



### derma_acniben_evaluation_current_model.csv
cloudToLocal(bucket_name, 'Current_model/derma_acniben_evaluation_current_model.csv', './derma_acniben_evaluation_current_model.csv')
# Save the file in a variable as csv format
with open('./derma_acniben_evaluation_current_model.csv', 'r', encoding='utf-8') as f:
    derma_acniben_evaluation_current_model = pd.read_csv(f)

# Delete the file
os.remove('./derma_acniben_evaluation_current_model.csv')



### derma_acniben_saturation.csv
cloudToLocal(bucket_name, f'{latest_folder_name}/derma_acniben_saturation.csv', './derma_acniben_saturation.csv')
# Save the file in a variable as csv format
with open('./derma_acniben_saturation.csv', 'r', encoding='utf-8') as f:
    derma_acniben_saturation = pd.read_csv(f)

# Delete the file
os.remove('./derma_acniben_saturation.csv')



### derma_acniben_predict_table_current.csv
cloudToLocal(bucket_name, 'Current_model/derma_acniben_predict_table_current.csv', './derma_acniben_predict_table_current.csv')
# Save the file in a variable as csv format
with open('./derma_acniben_predict_table_current.csv', 'r', encoding='utf-8') as f:
    derma_acniben_predict_table_weeks = pd.read_csv(f)

# Delete the file
os.remove('./derma_acniben_predict_table_current.csv')



### derma_acniben_df_aov.csv
cloudToLocal(bucket_name, f'{latest_folder_name}/derma_acniben_df_aov.csv', './derma_acniben_df_aov.csv')
# Save the file in a variable as csv format
with open('./derma_acniben_df_aov.csv', 'r', encoding='utf-8') as f:
    derma_acniben_df_aov = pd.read_csv(f)

# Delete the file
os.remove('./derma_acniben_df_aov.csv')



### derma_acniben_df_markets_week.csv
cloudToLocal(bucket_name, f'{latest_folder_name}/derma_acniben_df_markets_week.csv', './derma_acniben_df_markets_week.csv')
# Save the file in a variable as csv format
with open('./derma_acniben_df_markets_week.csv', 'r', encoding='utf-8') as f:
    derma_acniben_df_markets_week = pd.read_csv(f)

# Delete the file
os.remove('./derma_acniben_df_markets_week.csv')



### derma_df_year.csv
cloudToLocal(bucket_name, 'Current_model/derma_df_year.csv', './derma_df_year.csv')
# Save the file in a variable as csv format
with open('./derma_df_year.csv', 'r', encoding='utf-8') as f:
    derma_df_year = pd.read_csv(f)

# Delete the file
os.remove('./derma_df_year.csv')



### derma_df_month.csv
cloudToLocal(bucket_name, 'Current_model/derma_df_month.csv', './derma_df_month.csv')
# Save the file in a variable as csv format
with open('./derma_df_month.csv', 'r', encoding='utf-8') as f:
    derma_df_month = pd.read_csv(f)

# Delete the file
os.remove('./derma_df_month.csv')



### derma_df_week.csv
cloudToLocal(bucket_name, 'Current_model/derma_df_week.csv', './derma_df_week.csv')
# Save the file in a variable as csv format
with open('./derma_df_week.csv', 'r', encoding='utf-8') as f:
    derma_df_week = pd.read_csv(f)

# Delete the file
os.remove('./derma_df_week.csv')
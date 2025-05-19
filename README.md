# Marketing Mix Modeling

Developed by Datarmony for ISDIN.
Developed by Datarmony for ISDIN.

Marketing Mix Modeling is a statistical modeling technique that aims to identify the relationship between spending on marketing across each channel and the results achieved from it.

After an analysis of the initial data, feature engineering, and selection of the initial model, this code contains the production pipeline for the final model developed for ISDIN. 


This model is updated weekly with current data, and based on this, a new model is retrained. The model that is registered weekly for production is chosen using the Champion Challenge method. 
The table Current_model/models_in_production.csv saved in Cloud Storage contains the history and also records which model is served each week. 


The model is put into production through an application where the end user can simulate investments in different advertising channels and receive the expected sales as a return.

The backend, executed as a Cloud Job, connects to BigQuery to generate the data model, which is stored in Cloud Storage. The frontend, also loaded from a Cloud Service, accesses Cloud Storage to visualize data and calculate investments. To do this, users must authenticate via the Google Cloud OAuth 2.0 API. The admin panel, also loaded from a Cloud Service, functions similarly to the frontend, but only users with administrator privileges can modify access rights and download files.

## Table of contents
Here, include an index of all of the chapters included in the following format:

[1.- Repository Overview](#1--repository-overview)

[2.- Prerequisites](#2--prerequisites)

[3.- Usage](#3--usage)

[4.- Deployment](#4--deployment)

[5.- Maintainers](#5--maintainers)


## 1.- Repository Overview

Main modules required for the system's functioning are split across ```isdin_backend```, ```isdin_frontend``` and ```isdin_permissions``` folders.

```
.
├── README.md
├── isdin_backend
│   ├── Dockerfile
│   ├── SQL
│   ├── main.py
│   ├── requirements.txt
│   └── src
│       ├── artifacts.py
│       ├── cloud.py
│       ├── data_exploration.py
│       ├── data_transformation.py
│       └── model_utilities.py
├── isdin_frontend
│   ├── Dockerfile
│   ├── assets
│   ├── config.toml
│   ├── load_model.py
│   ├── login.py
│   ├── main.py
│   ├── requirements.txt
│   ├── src
│   │   └── cloud.py
│   └── style.css
├── isdin_permissions
│   ├── Dockerfile
│   ├── assets
│   ├── config.toml
│   ├── load_model.py
│   ├── login.py
│   ├── main.py
│   ├── requirements.txt
│   ├── src
│   │   └── cloud.py
│   ├── style.css
│   └── temp.json
└── permitted_emails.json
```

## 2.- Prerequisites

Prior to the execution of the code, replicate the following file structure on [Cloud Storage](https://docs.google.com/spreadsheets/d/1X********************/edit?usp=sharing):

* ```SRC``` folder: folder containing constant information, such as the names of the provinces and autonomous communities of Spain. Contains the following files:

    * tab-ccaa.xlsx
    * tab-prov.xlsx
    * tab-share-ccaa.xlsx

* ```Current_model``` folder: with the model in production and its metadata. This folder is necessary to compare the current model with the current one.

## Table of contents
Here, include an index of all of the chapters included in the following format:

[1.- Repository Overview](#1--repository-overview)

[2.- Prerequisites](#2--prerequisites)

[3.- Usage](#3--usage)

[4.- Deployment](#4--deployment)

[5.- Maintainers](#5--maintainers)


## 1.- Repository Overview

Main modules required for the system's functioning are split across ```isdin_backend```, ```isdin_frontend``` and ```isdin_permissions``` folders.

```
.
├── README.md
├── isdin_backend
│   ├── Dockerfile
│   ├── SQL
│   ├── main.py
│   ├── requirements.txt
│   └── src
│       ├── artifacts.py
│       ├── cloud.py
│       ├── data_exploration.py
│       ├── data_transformation.py
│       └── model_utilities.py
├── isdin_frontend
│   ├── Dockerfile
│   ├── assets
│   ├── config.toml
│   ├── load_model.py
│   ├── login.py
│   ├── main.py
│   ├── requirements.txt
│   ├── src
│   │   └── cloud.py
│   └── style.css
├── isdin_permissions
│   ├── Dockerfile
│   ├── assets
│   ├── config.toml
│   ├── load_model.py
│   ├── login.py
│   ├── main.py
│   ├── requirements.txt
│   ├── src
│   │   └── cloud.py
│   ├── style.css
│   └── temp.json
└── permitted_emails.json
```

## 2.- Prerequisites

Prior to the execution of the code, replicate the following file structure on [Cloud Storage](https://docs.google.com/spreadsheets/d/1X********************//edit?usp=sharing):

* ```SRC``` folder: folder containing constant information, such as the names of the provinces and autonomous communities of Spain. Contains the following files:

    * tab-ccaa.xlsx
    * tab-prov.xlsx
    * tab-share-ccaa.xlsx

* ```Current_model``` folder: with the model in production and its metadata. This folder is necessary to compare the current model with the current one.

* The folder from last week: because this is the Current model, and we need information about its training data.

* ```permitted_emails.json```: JSON file mapping user emails to their privileges on the application, being:
    * ```normal:``` viewing rights
    * ```admin```: access to the administration module

This [spreadsheet](https://docs.google.com/spreadsheets/d/1t********************/edit?usp=sharing&ouid=110974016208092973379&rtpof=true&sd=true) describes how files are generated in a structured manner.

## 3.- Usage

This code is intended to be ran on GCP environment.

* Backend

The first action performed in the Backend main.py is **data extraction**, done through the queries contained in the *SQL folder*, as most of the data originates from BigQuery. 

Additionally, three tables containing information on province names, autonomous communities (CCAA), and Share by CCAA are stored in Cloud Storage and read into the code. More details about the input data can be found in [this table](https://docs.google.com/spreadsheets/d/1X********************/edit?usp=sharing).

* ```permitted_emails.json```: JSON file mapping user emails to their privileges on the application, being:
    * ```normal:``` viewing rights
    * ```admin```: access to the administration module

## 3.- Usage

This code is intended to be ran on GCP environment.

* Backend

The first action performed in the Backend main.py is **data extraction**, done through the queries contained in the *SQL folder*, as most of the data originates from BigQuery. 

Additionally, three tables containing information on province names, autonomous communities (CCAA), and Share by CCAA are stored in Cloud Storage and read into the code. More details about the input data can be found in [this table](https://docs.google.com/spreadsheets/d/1X********************/edit?usp=sharing).

Soon, with the help of the functions found in the *src folder*, we do the following: we create a data frame with the unified **raw data** (df_raw), add future prediction periods (so the end user can select them in the interface, df_raw_fut_weeks), perform the necessary data transformations to input them into the models (df_transf), and calculate the adstock (df_adstock).

For the Weather model, we use an API from Meteostat to obtain weather forecast information. After incorporating this data, we have all the features needed to **train the model** (df_train).

Currently, we train two types of models: the model by months (model_months) and the model that combines monthly and weather information (model_months_weather). The created models are then **evaluated** (df_evaluation_modelname, df_coef_model_modelname, df_cross_validation_modelname, fig_residuals_diagnostic_modelname, model_residual_tests_modelname).

The feature and target data are saved in a data frame called df_final. Using the function calculate_sales_with_and_without_investment, we add to this data frame the model's prediction of sales with and without investments (information needed for the interface).

Following these steps, **we compare this model with the current model**, which is the one currently in production. To do this, we recalculate the R² of the registered model using the most recent data (ensuring that training data is excluded) and compare the R² values of both models. The model with the better performance will be selected as the new current model (current_model).

Once the current model has been selected, we add the name of the model in production for that week to the models_in_production table. Additionally, we conduct a detailed evaluation of this model (fig_residuals_diagnostic_current, model_residual_tests_current, df_coef_model_current), generate the table with predictions (displayed in the interface, predict_table_current), and create the **df_final** using the model in production (df_final_current).

All the necessary files for the frontend and model metadata are saved in Cloud Storage.


* Frontend


* Frontend

The frontend, loaded by a cloud service, generates a dashboard where users can log in via the Google Cloud OAuth API. It retrieves the permitted_emails.json file from Cloud Storage to verify access permissions. Once logged in, the system displays the data stored in Cloud Storage by the backend and runs the logic for investment predictions.

* Admin pannel

* Admin pannel

The admin panel, loaded by a cloud service, provides a dashboard where users can log in via the Google Cloud OAuth API. It retrieves the permitted_emails.json file from Cloud Storage to verify access permissions, with administrator privileges required to gain access. Once logged in, the panel allows users to modify permissions and download files.

## 4.- Deployment

**Please replicate the GCS folder structure as specified in the [prerequisites](#2--prerequisites) chapter.**

GCP services used in this project are:

* ```OAuth API```  for authenticating in the frontend

* ```GCS``` for storing additional and required files

* ```AR``` for storing the different Docker images used in CloudRun

The development of this pipeline was made using the infra at the GCP project ```marketing-mix-modeling-386411```

### Steps for deployment

#### Secrets
Secrets must be set to be used by the different modules implemented in this pipeline:

* ```DATARMONY-sender_email_pass```
* ```GOOGLE_APPLICATION_CREDENTIALS-bq-user```
* ```MMM_GOOGLE_CLIENT_ID```
* ```MMM_GOOGLE_CLIENT_SECRET```


#### Frontend

* Create a **CloudRunService** following the configuration from ```marketing-mix-modeling-frontend```.
    * ```marketing-mix-modeling-frontend``` Docker Image
    * Container port 8501
    * The following environment variables
        * ```CLOUD_STORAGE_BUCKET```=```marketing-mix-modeling```
        * ```MMM_GOOGLE_REDIRECT_URI```=```https://marketing-mix-modeling-frontend-oc42nla5rq-ew.a.run.app```
        * ```ADMIN_GOOGLE_REDIRECT_URI```=```https://marketing-mix-modeling-permissions-oc42nla5rq-ew.a.run.app```
    * The following secrets
        * Secret Name: ```SENDER_EMAIL_PASS``` Secret ID:```DATARMONY-sender_email_pass```
        * Secret Name: ```GOOGLE_APPLICATION_CREDENTIALS``` Secret ID: ```GOOGLE_APPLICATION_CREDENTIALS-bq-user```
        * Secret Name: ```MMM_GOOGLE_CLIENT_ID``` Secret ID: ```MMM_GOOGLE_CLIENT_ID```
        * Secret Name: ```MMM_GOOGLE_CLIENT_SECRET``` Secret ID: ```MMM_GOOGLE_CLIENT_SECRET```

#### Backend

* Create a **CloudRunJob** following the configuration from ```marketing-mix-modeling-backend```
    * ```marketing-mix-modeling-backend``` Docker Image
    * The following environment variables:
        * ```CLOUD_STORAGE_BUCKET```=```marketing-mix-modeling```
        * ```EMAIL_RECEIVER```=```daniel.gonzalez@datarmony.com, ca********************@datarmony.com, ma********************@datarmony.com```
    * The following secrets:
        * Secret Name: ```SENDER_EMAIL_PASS``` Secret ID:```DATARMONY-sender_email_pass```
        * Secret Name: ```GOOGLE_APPLICATION_CREDENTIALS``` Secret ID: ```GOOGLE_APPLICATION_CREDENTIALS-bq-user```
    * Configure this job to run every Monday in the triggers section of the CloudRunJob itself, adjusting the time zone to that of Spain.

#### Admin

* Create a **CloudRunService** following the configuration from ```marketing-mix-modeling-permissions```
    * ```marketing-mix-modeling-permissions``` Docker Image
    * Same exact configuration as the frontend CloudRunService.

## 5.- Maintainers

* Daniel Gonzalez - Cloud Architect (Datarmony) - daniel.gonzalez@datarmony.com
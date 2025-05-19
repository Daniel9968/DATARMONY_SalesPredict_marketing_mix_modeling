from google.cloud import bigquery, storage
import google.auth
import os
import requests
from meteostat import Point, Daily
import pandas as pd
from datetime import datetime
from google.oauth2 import service_account
import json

def cloudToLocal(bucket_name, source_blob_name, destination_file_name):
    """
    Downloads a file from Google Cloud Storage to the local file system.

    Args:
        bucket_name (str): Name of the bucket in Google Cloud Storage where the file is located.
        source_blob_name (str): Name of the file in Google Cloud Storage to be downloaded.
        destination_file_name (str): Local file path and name where the downloaded file will be saved.
    """
    # Set the environment variable with the path to your credentials file

    # Create the Cloud Storage client using the credentials
    google_credentials_json = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

    # Convert the JSON string into a credentials object
    credentials = service_account.Credentials.from_service_account_info(json.loads(google_credentials_json))
    project = credentials.project_id

    # Create the client
    client = storage.Client(credentials=credentials, project=project)

    # Get the bucket
    bucket = client.get_bucket(bucket_name)

    # Get the blob (the file in the bucket)
    blob = bucket.blob(source_blob_name)

    # Download the blob to the local file
    blob.download_to_filename(destination_file_name)

    # print(f'Fichero {source_blob_name} descargado a {destination_file_name}.') 

def get_latest_cloud_folder_name(bucket_name):
    """
    Retrieves the highest numeric folder name from Google Cloud Storage bucket.

    Args:
        bucket_name (str): Name of the bucket in Google Cloud Storage.

    Returns:
        str or None: The highest numeric folder name found in the bucket, or None if no valid folders are found.
    """
    # Autenticación con Google Cloud
    google_credentials_json = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

    # Convert the JSON string into a credentials object
    credentials = service_account.Credentials.from_service_account_info(json.loads(google_credentials_json))
    project = credentials.project_id

    # Create the client
    client = storage.Client(credentials=credentials, project=project)

    # Get the bucket
    bucket = client.get_bucket(bucket_name)

    # Retrieve the list of blobs in the bucket
    blobs = bucket.list_blobs()

    # Filter and obtain folder names (assuming folders end with '/')
    folder_numbers = []
    for blob in blobs:
        # Retrieve the first 8 characters of the blob's name
        if len(blob.name) >= 8:
            first_8_chars = blob.name[:8]
            # Check if they are all digits
            if first_8_chars.isdigit():
                folder_numbers.append(int(first_8_chars))

    # Find the highest number
    latest_folder = max(folder_numbers) if folder_numbers else None

    # print(latest_folder)  

    return latest_folder
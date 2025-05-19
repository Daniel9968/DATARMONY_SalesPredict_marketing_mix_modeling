from google.cloud import bigquery, storage
import google.auth
import os
import requests
from meteostat import Point, Daily
import pandas as pd
from datetime import datetime
from google.oauth2 import service_account
import json
import smtplib
import logging
import google.cloud.logging
from google.cloud.logging_v2.handlers import CloudLoggingHandler

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

def filesToCloud(file_name, file_content, content_type, bucket_name, file_path):
    """
    Uploads a file to Google Cloud Storage.

    Args:
        file_name (str): Name of the file to be created in the bucket.
        file_content (str): Content of the file to be uploaded.
        content_type (str): Content type of the file.
        bucket_name (str): Name of the bucket in Google Cloud Storage where the file will be stored.
        file_path (str): Path in the bucket where the file will be saved.
    """

    # Create the Cloud Storage client using credentials
    google_credentials_json = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

    # Convert the JSON string into a credentials object
    credentials = service_account.Credentials.from_service_account_info(json.loads(google_credentials_json))
    project = credentials.project_id
    
    # Create the client
    client = storage.Client(credentials=credentials, project=project)

    # Get the bucket
    bucket = client.get_bucket(bucket_name)

    # Create the full file name with the current date
    file_name_with_date = f"{file_path}/{file_name}"

    # Create the blob
    blob = bucket.blob(file_name_with_date)

    if content_type == "text/csv" or content_type == "application/octet-stream":
        # Upload the file
        blob.upload_from_string(file_content, content_type=content_type)

    if content_type == "image/png":
        # Upload the file
        with open(file_content, "rb") as file_obj:
            blob.upload_from_file(file_obj, content_type=content_type)

def send_email(sender_pass, subject, body, receiver):
    """
    Sends an email from the Datarmony notifications account to a group of recipients.

    Args:
        sender_pass (str): Password of the email sender.
        subject (str): Subject of the email.
        body (str): Body of the email message.
        receiver (str): List of the emails to send the message.
    """
    sender="notifier.datarmony@gmail.com"
    email_receiver = receiver.split(",") if receiver is not None else sender
    with smtplib.SMTP("smtp.gmail.com", 587) as smtp:
        smtp.ehlo()
        smtp.starttls()
        smtp.ehlo()
        smtp.login(sender, sender_pass)
        msg = f'Subject: {subject}\n\n{body}'
        smtp.sendmail(sender, email_receiver, msg=msg)

class MyLogger(logging.Logger):
    """This class extends `Logger` class from `logging` module and logs personalized messages for debugs, error and info etc.
    It can be also used to log to **Google cloud logger** so user can view and investigate the execution 
    process.
    """

    def __init__(self, python_file, log_file_name=None, level=logging.DEBUG):
        """Instantiates a `MyLogger` object and sets default filehandler and streamhandler. It 
        also sets the default format for log messages.

        :param str python_file: Name of python file.
        :param str log_file_name: Name of log file (creates if not exist).
        :param str level: Optional Default log level. 
        """
        super(MyLogger, self).__init__(python_file, level)
        self.__set_formatter()
        #self.__set_fileHandler(log_file_name)
        #self.__set_cloudHandler(log_file_name)
        self.__set_streamHandler()
       


    def __set_formatter(self):
        """
        Sets the default format for log messages.
        """
        #self.formatter = logging.Formatter('[%(asctime)s] - [%(lineno)s] - [%(levelname)s]:  %(message)s ---->  %(name)s', "%Y-%m-%d %H:%M:%S") 
                            
        self.formatter = logging.Formatter('%(message)s ----> %(name)s', "%Y-%m-%d %H:%M:%S")
                              

    def __set_fileHandler(self, log_file_name):
        """
        Sets the default file handler for our class. 


        """
        if log_file_name is not None:
            self.file_handler = logging.FileHandler(log_file_name)
            self.file_handler.setLevel(logging.INFO)
            self.file_handler.setFormatter(self.formatter)
            self.addHandler(self.file_handler)

    def __set_streamHandler(self):
        """
        Sets the default stream handler for our class.
        """
        self.stream_handler = logging.StreamHandler()
        self.stream_handler.setFormatter(self.formatter)
        self.addHandler(self.stream_handler)

    def __set_cloudHandler(self, log_file_name):
        """
        Sets **Google cloud handler** for class thus enables cloud logging.
        """
        client = google.cloud.logging_v2.Client()
        cloud_handler = CloudLoggingHandler(client, name=log_file_name)
        cloud_handler.setLevel(logging.INFO)
        cloud_handler.setFormatter(self.formatter)
        self.addHandler(cloud_handler)
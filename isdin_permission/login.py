from google.cloud import storage
from dotenv import load_dotenv
import streamlit as st
import streamlit_google_oauth as oauth
import json
import os
from google.cloud import storage
from google.oauth2 import service_account

load_dotenv()

google_credentials_json = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
credentials = service_account.Credentials.from_service_account_info(json.loads(google_credentials_json))
project = credentials.project_id
 
os.environ["GOOGLE_CLIENT_ID"]=os.environ["MMM_GOOGLE_CLIENT_ID"]
os.environ["GOOGLE_CLIENT_SECRET"]=os.environ["MMM_GOOGLE_CLIENT_SECRET"]
os.environ["GOOGLE_REDIRECT_URI"]=os.environ["ADMIN_GOOGLE_REDIRECT_URI"]

storage_client = storage.Client(credentials=credentials, project=project)
# Replace these values with your specific GCS bucket and file name 
bucket_name = os.getenv("CLOUD_STORAGE_BUCKET")
file_name = 'permitted_emails.json'


# Access the specified bucket
bucket = storage_client.get_bucket(bucket_name)

# Access the blob (file) in the bucket
blob = bucket.blob(file_name)

# Download the JSON file as a string
json_string = blob.download_as_text()

# Parse the JSON string into a Python dictionary
permitted_emails = json.loads(json_string)["permitted_emails"]
# st.text(permitted_emails)

client_id = os.environ["GOOGLE_CLIENT_ID"]
client_secret = os.environ["GOOGLE_CLIENT_SECRET"]
redirect_uri = os.environ["GOOGLE_REDIRECT_URI"]
mmm_google_redirect_uri = os.environ["MMM_GOOGLE_REDIRECT_URI"]


# Inject custom CSS into the page
# st.markdown(custom_css, unsafe_allow_html=True)

# Create a Streamlit web page title and some introductory text
st.image("assets/header.png")



# Use the streamlit_google_oauth library for login inside the container
login_info = oauth.login(
    client_id=client_id,
    client_secret=client_secret,
    redirect_uri=redirect_uri,
    logout_button_text="Logout",
    app_name="Continuar con Google",

)

if login_info:
    user_id, user_email = login_info
    if user_email.lower() not in map(str.lower, permitted_emails.keys()):
        st.write(f"Este correo electrónico \"{user_email}\" no está en la lista de correos electrónicos permitidos. Por favor, cierre sesión e intente nuevamente con un correo electrónico correcto.")
    elif permitted_emails[user_email.lower()] != "Admin":
        st.write(f"Lo siento, {user_email} no tiene privilegios de administrador.")
    else:
        from main import main
        main()
else:
    st.write('\n')
    # Define the layout of the columns
    col1, col2 = st.columns([0.3, 0.8])
    with col1:
        if st.button("Marketing Mix Modeling"):
            st.markdown(f"<meta http-equiv='refresh' content='0;URL={mmm_google_redirect_uri}' />", unsafe_allow_html=True)
    st.title("Google Login")
    st.write("Inicie sesión con su cuenta de Google.")
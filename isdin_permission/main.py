import streamlit as st
from dotenv import load_dotenv
import os
import streamlit_google_oauth as oauth
from google.cloud import storage
from google.oauth2 import service_account
import json
import pickle
from load_model import foto_coefficients_weather, foto_df_coef_model_current, foto_fig_residuals_diagnostic_current, foto_model_residual_tests_current, foto_predict_table_current, foto_predict_table_months, foto_predict_table_weather, foto_weather_data_ccaa, ceutics_coefficients_regression, ceutics_importances_rf, ceutics_fig_residuals_diagnostic_current_regression, ceutics_model_residual_tests_current_regression, ceutics_predict_table_current, ceutics_predict_table_rf, ceutics_df_nodes_best_tree, ceutics_fig_best_tree, ceutics_df_importances_and_coef, derma_acniben_df_coef_model_current, derma_acniben_fig_residuals_diagnostic_current, derma_acniben_model_residual_tests_current, derma_acniben_predict_table_current, derma_acniben_predict_table_months


load_dotenv()

google_credentials_json = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
credentials = service_account.Credentials.from_service_account_info(json.loads(google_credentials_json))
project = credentials.project_id

storage_client = storage.Client(credentials=credentials, project=project)

# Replace these values with the specific name of your GCS bucket and the file name
bucket_name = os.getenv("CLOUD_STORAGE_BUCKET")
file_name = 'permitted_emails.json'

# Access the specified bucket
bucket = storage_client.get_bucket(bucket_name)

# Access the blob (file) in the bucket
blob = bucket.blob(file_name)

def update_json_file(permitted_emails):
    # Upload the modified dictionary to cloud storage
    with open("temp.json", "w") as json_file:
        json.dump({"permitted_emails": permitted_emails}, json_file)

    # Upload the updated JSON file to cloud storage
    blob.upload_from_filename("temp.json")

def main():
    # Download the JSON file as a string
    json_string = blob.download_as_text()

    # Parse the JSON string into a Python dictionary
    permitted_emails = json.loads(json_string)["permitted_emails"]

    # Add New Email Addresses
    st.title("Agregar Nuevo Correo Electrónico")
    new_email = st.text_input("Nuevo Correo Electrónico:")
    email_type = st.selectbox("Tipo de Correo Electrónico:", ["Admin", "Viewer"])

    if st.button("Agregar Correo Electrónico"):
        permitted_emails[new_email] = email_type
        update_json_file(permitted_emails)
        st.success(f"Correo electrónico '{new_email}' agregado como '{email_type}'.")
        st.experimental_rerun()

    st.title("Correos Electrónicos Permitidos")

    # Display Users and Their Allowed Email Addresses as a Table within an Expander
    with st.expander("Correos Electrónicos Permitidos", expanded=True):
        for user, email_type in permitted_emails.items():
            col1, col2, col3 = st.columns([0.6, 0.2, 0.2])
            with col1:
                st.write(user)
            with col2:
                st.write(email_type)
            with col3:
                delete_btn = st.button("Eliminar", key=f"delete_{user}")
                if delete_btn:
                    del permitted_emails[user]
                    update_json_file(permitted_emails)
                    st.success(f"Correo electrónico '{user}' eliminado exitosamente.")
                    st.experimental_rerun()
    



    st.title("Descargar ficheros")
    st.markdown("## Producto 1")
    st.download_button(
        label="Descargar Coeficientes del modelo clima",
        data=foto_coefficients_weather.applymap(lambda x: f"{x:.2f}" if isinstance(x, float) else x).to_csv(),
        file_name="foto_coefficients_weather.csv",
        key="foto_coefficients_weather",
        help="Presiona para descargar el archivo"
    )
    st.download_button(
        label="Descargar Coeficientes del modelo semanal en producción",
        data=foto_df_coef_model_current.applymap(lambda x: f"{x:.2f}" if isinstance(x, float) else x).to_csv(),
        file_name="foto_df_coef_model_current.csv",
        key="foto_df_coef_model_current",
        help="Presiona para descargar el archivo"
    )

    st.download_button(
        label="Descargar Diagnóstico residual del modelo semanal en producción",
        data=foto_fig_residuals_diagnostic_current,
        file_name="foto_fig_residuals_diagnostic_current.png",
        mime="image/png",
        key="foto_fig_residuals_diagnostic_current",
        help="Presiona para descargar el archivo"
    )
    st.download_button(
        label="Descargar Pruebas residuales del modelo semanal en producción",
        data=pickle.dumps(foto_model_residual_tests_current),
        file_name="foto_model_residual_tests_current.pkl",
        key="foto_model_residual_tests_current",
        help="Presiona para descargar el archivo"
    )

    st.download_button(
        label="Descargar Tabla de predicción del modelo semanal en producción",
        data=foto_predict_table_current.applymap(lambda x: f"{x:.2f}" if isinstance(x, float) else x).to_csv(),
        file_name="foto_predict_table_current.csv",
        key="foto_predict_table_current",
        help="Presiona para descargar el archivo"
    )
    
    st.download_button(
        label="Descargar Tabla de predicción del modelo semanal actual",
        data=foto_predict_table_months.applymap(lambda x: f"{x:.2f}" if isinstance(x, float) else x).to_csv(),
        file_name="foto_predict_table_months.csv",
        key="foto_predict_table_months",
        help="Presiona para descargar el archivo"
    )
    st.download_button(
        label="Descargar Tabla de predicción del modelo clima",
        data=foto_predict_table_weather.applymap(lambda x: f"{x:.2f}" if isinstance(x, float) else x).to_csv(),
        file_name="foto_predict_table_weather.csv",
        key="foto_predict_table_weather",
        help="Presiona para descargar el archivo"
    )
    st.download_button(
        label="Descargar datos de clima por comunidad autónoma",
        data=foto_weather_data_ccaa.applymap(lambda x: f"{x:.2f}" if isinstance(x, float) else x).to_csv(),
        file_name="foto_weather_data_ccaa.csv",
        key="foto_weather_data_ccaa",
        help="Presiona para descargar el archivo"
    )


    st.markdown("## Producto 2")
    st.download_button(
        label="Descargar Coeficientes del modelo de regresión en producción",
        data=ceutics_coefficients_regression.applymap(lambda x: f"{x:.2f}" if isinstance(x, float) else x).to_csv(),
        file_name="ceutics_coefficients_regression.csv",
        key="ceutics_coefficients_regression",
        help="Presiona para descargar el archivo"
    )

    st.download_button(
        label="Descargar Importancia de las variables del modelo RandomForest en producción",
        data=ceutics_importances_rf.applymap(lambda x: f"{x:.2f}" if isinstance(x, float) else x).to_csv(),
        file_name="ceutics_importances_rf.csv",
        key="ceutics_importances_rf",
        help="Presiona para descargar el archivo"
    )

    st.download_button(
        label="Descargar Diagnóstico residual del modelo de regresión en producción",
        data=ceutics_fig_residuals_diagnostic_current_regression,
        file_name="ceutics_fig_residuals_diagnostic_current_regression.png",
        mime="image/png",
        key="ceutics_fig_residuals_diagnostic_current_regression",
        help="Presiona para descargar el archivo"
    )

    st.download_button(
        label="Descargar Pruebas residuales del modelo de regresión en producción",
        data=pickle.dumps(ceutics_model_residual_tests_current_regression),
        file_name="ceutics_model_residual_tests_current_regression.pkl",
        key="ceutics_model_residual_tests_current_regression",
        help="Presiona para descargar el archivo"
    )

    st.download_button(
        label="Descargar Tabla de predicción del modelo de RandomForest en producción",
        data=ceutics_predict_table_current.applymap(lambda x: f"{x:.2f}" if isinstance(x, float) else x).to_csv(),
        file_name="ceutics_predict_table_current.csv",
        key="ceutics_predict_table_current",
        help="Presiona para descargar el archivo"
    )
    
    st.download_button(
        label="Descargar Tabla de predicción del modelo de RandomForest actual",
        data=ceutics_predict_table_rf.applymap(lambda x: f"{x:.2f}" if isinstance(x, float) else x).to_csv(),
        file_name="ceutics_predict_table_rf.csv",
        key="ceutics_predict_table_rf",
        help="Presiona para descargar el archivo"
    )

    st.download_button(
        label="Descargar Tabla con detalles del mejor arbol (RandomForest)",
        data=ceutics_df_nodes_best_tree.applymap(lambda x: f"{x:.2f}" if isinstance(x, float) else x).to_csv(),
        file_name="ceutics_df_nodes_best_tree.csv",
        key="ceutics_df_nodes_best_tree",
        help="Presiona para descargar el archivo"
    )
    
    st.download_button(
        label="Descargar Imagen del mejor arbol (RandomForest)",
        data=ceutics_fig_best_tree,
        file_name="ceutics_fig_best_tree.png",
        mime="image/png",
        key="ceutics_fig_best_tree",
        help="Presiona para descargar el archivo"
    )

    st.download_button(
        label="Descargar Tabla con importancias RandomForest y coeficientes regresión",
        data=ceutics_df_importances_and_coef.applymap(lambda x: f"{x:.2f}" if isinstance(x, float) else x).to_csv(),
        file_name="ceutics_df_importances_and_coef.csv",
        key="ceutics_df_importances_and_coef",
        help="Presiona para descargar el archivo"
    )



    st.markdown("## Producto 3")
    st.download_button(
        label="Descargar Coeficientes del modelo semanal en producción",
        data=derma_acniben_df_coef_model_current.applymap(lambda x: f"{x:.2f}" if isinstance(x, float) else x).to_csv(),
        file_name="derma_acniben_df_coef_model_current.csv",
        key="derma_acniben_df_coef_model_current",
        help="Presiona para descargar el archivo"
    )

    st.download_button(
        label="Descargar Diagnóstico residual del modelo semanal en producción",
        data=derma_acniben_fig_residuals_diagnostic_current,
        file_name="derma_acniben_fig_residuals_diagnostic_current.png",
        mime="image/png",
        key="derma_acniben_fig_residuals_diagnostic_current",
        help="Presiona para descargar el archivo"
    )

    st.download_button(
        label="Descargar Pruebas residuales del modelo semanal en producción",
        data=pickle.dumps(derma_acniben_model_residual_tests_current),
        file_name="derma_acniben_model_residual_tests_current.pkl",
        key="derma_acniben_model_residual_tests_current",
        help="Presiona para descargar el archivo"
    )

    st.download_button(
        label="Descargar Tabla de predicción del modelo semanal en producción",
        data=derma_acniben_predict_table_current.applymap(lambda x: f"{x:.2f}" if isinstance(x, float) else x).to_csv(),
        file_name="derma_acniben_predict_table_current.csv",
        key="derma_acniben_predict_table_current",
        help="Presiona para descargar el archivo"
    )
    
    st.download_button(
        label="Descargar Tabla de predicción del modelo semanal actual",
        data=derma_acniben_predict_table_months.applymap(lambda x: f"{x:.2f}" if isinstance(x, float) else x).to_csv(),
        file_name="derma_acniben_predict_table_months.csv",
        key="derma_acniben_predict_table_months",
        help="Presiona para descargar el archivo"
    )

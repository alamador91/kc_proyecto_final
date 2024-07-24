# main.py

import setuptools
from drive_downloader import download_from_drive
from functions import load_data, best_model

# Definimos la URL de la carpeta compartida para el modelo y el diccionario
folder_url_modelo = 'https://drive.google.com/drive/folders/1wQv6lTixINg17x9E2JNhWoDO1fi7l4Xx?usp=sharing'
folder_id_modelo = folder_url_modelo.split('/')[-1].split('?')[0]

# Descargamos los archivos en la carpeta usando gdown
download_from_drive(folder_id_modelo)

# Definimos las rutas de los archivos
dict_model_results_path = '/Users/otgerpeidro/Library/CloudStorage/OneDrive-Personal/Bootcamp IA/practica_final/FraudDetection_Pipeline/dict_model_results.pkl'
model_random_forest_path = '/Users/otgerpeidro/Library/CloudStorage/OneDrive-Personal/Bootcamp IA/practica_final/FraudDetection_Pipeline/model_random_forest.pkl'

# Cargamos el modelo y los resultados del pipeline
model_randomforest, results_pipeline = load_data(model_random_forest_path, dict_model_results_path)

# Parámetros de configuración de MLflow
remote_server_uri = "https://the-balloon-project.com"
exp_name = "Test Final"

# Encontrar el mejor modelo y registrar los resultados en MLFlow
best_model, best_model_name = best_model(results_pipeline, remote_server_uri, exp_name)
print(f"El mejor modelo es: {best_model_name}")


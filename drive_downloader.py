# drive_downloader.py

import os

def download_from_drive(folder_id):
    """
    Descarga todos los archivos de una carpeta de Google Drive usando gdown.

    Parameters:
    folder_id (str): El ID de la carpeta de Google Drive.
    """
    os.system(f'gdown --folder {folder_id}')

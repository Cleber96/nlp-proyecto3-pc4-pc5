# download_and_prepare_hf_data.py

import pandas as pd
import os
import sys
from datasets import load_dataset
import logging

# Configuración del logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Aseguramos que el script pueda encontrar el directorio de config
# Añadir la raíz del proyecto al path de Python si no está ya
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = script_dir
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from config import settings # Importamos las configuraciones para las rutas

def download_and_prepare_hf_dataset(dataset_name="amazon_polarity", sample_size=50000):
    """
    Descarga un dataset de Hugging Face, lo procesa y lo guarda
    como product_reviews.csv en el directorio raw del proyecto.

    Args:
        dataset_name (str): Nombre del dataset en Hugging Face a descargar.
        sample_size (int, optional): Número de ejemplos a muestrear del dataset
                                      para crear un corpus más pequeño.
                                      Si es None o 0, usa todo el dataset de entrenamiento.
    """
    output_csv_path = os.path.join(settings.RAW_DATA_DIR, settings.RAW_DATA_FILENAME)

    logger.info(f"Iniciando descarga y preparación del dataset '{dataset_name}' de Hugging Face.")
    logger.info(f"El archivo se guardará en: {output_csv_path}")

    # Bloque 1: Crear directorios necesarios
    # ---
    # Asegurarse de que la carpeta data/raw exista antes de guardar el archivo.
    settings.create_necessary_directories()

    # Bloque 2: Cargar el dataset desde Hugging Face
    # ---
    # Usamos load_dataset para descargar el dataset directamente.
    try:
        # Cargamos solo la parte de 'train' para evitar cargar todo si no es necesario.
        # amazon_polarity tiene splits 'train' y 'test'.
        dataset = load_dataset(dataset_name, split='train')
        logger.info(f"Dataset '{dataset_name}' cargado exitosamente. Ejemplos totales: {len(dataset)}")
    except Exception as e:
        logger.error(f"Error al cargar el dataset '{dataset_name}' de Hugging Face: {e}", exc_info=True)
        logger.error("Asegúrate de tener conexión a internet y que el nombre del dataset sea correcto.")
        sys.exit(1)

    # Bloque 3: Muestrear el dataset (opcional)
    # ---
    # Si se especifica un sample_size, tomamos una muestra aleatoria.
    # Esto es crucial para trabajar con un "corpus pequeño" y para pruebas rápidas.
    if sample_size and sample_size > 0 and sample_size < len(dataset):
        logger.info(f"Muestreando {sample_size} ejemplos del dataset total de {len(dataset)}.")
        dataset = dataset.shuffle(seed=settings.RANDOM_SEED).select(range(sample_size))
        logger.info(f"Tamaño de la muestra del dataset: {len(dataset)} ejemplos.")
    elif sample_size >= len(dataset):
        logger.info(f"El tamaño de muestra ({sample_size}) es mayor o igual al total del dataset ({len(dataset)}). Se usará el dataset completo.")
    else:
        logger.info(f"No se especificó un tamaño de muestra o es 0. Se usará el dataset completo de {len(dataset)} ejemplos.")

    # Bloque 4: Convertir a Pandas DataFrame y renombrar columnas
    # ---
    # Los datasets de Hugging Face suelen tener nombres de columna estandarizados,
    # pero los renombraremos para que coincidan con 'text' y 'label'.
    # amazon_polarity tiene 'content' y 'label'.
    
    # Asegurarse de que las columnas esperadas existan en el dataset
    expected_text_col = 'content'
    expected_label_col = 'label'

    if expected_text_col not in dataset.column_names or expected_label_col not in dataset.column_names:
        logger.error(f"El dataset '{dataset_name}' no contiene las columnas esperadas ('{expected_text_col}', '{expected_label_col}').")
        logger.error(f"Columnas disponibles: {dataset.column_names}")
        sys.exit(1)

    df = pd.DataFrame(dataset)
    df.rename(columns={expected_text_col: 'text', expected_label_col: 'label'}, inplace=True)
    
    # Bloque 5: Mostrar información sobre las etiquetas
    # ---
    # Verificamos la distribución de las etiquetas.
    logger.info("\nDistribución de etiquetas en el dataset:")
    logger.info(df['label'].value_counts())

    # Bloque 6: Guardar el DataFrame como CSV
    # ---
    df.to_csv(output_csv_path, index=False)
    logger.info(f"\n¡Dataset 'product_reviews.csv' generado exitosamente en: {output_csv_path}!")
    logger.info("Ahora puedes proceder con la fase de preprocesamiento de tu proyecto.")

if __name__ == "__main__":
    # Puedes ajustar el tamaño de la muestra aquí.
    # Pon sample_size=None o 0 para descargar el dataset completo.
    # El dataset amazon_polarity (train) tiene ~3.6 millones de ejemplos,
    # por lo que un sample_size de 50,000 o 100,000 es razonable para pruebas.
    download_and_prepare_hf_dataset(dataset_name="amazon_polarity", sample_size=50000)
# utils/preprocess.py

import os
import pandas as pd
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer
import logging
from sklearn.model_selection import train_test_split

# Importar configuración global
from config import settings

# Configuración de logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_raw_data(file_path: str) -> pd.DataFrame:

    if not os.path.exists(file_path):
        logger.error(f"Error: El archivo de datos raw no se encontró en '{file_path}'.")
        raise FileNotFoundError(f"Archivo no encontrado: {file_path}")
    
    try:
        df = pd.read_csv(file_path)
        logger.info(f"Datos raw cargados exitosamente desde '{file_path}'. Filas: {len(df)}")
        # Asegúrate de que las columnas esperadas existan
        if settings.TEXT_COLUMN not in df.columns:
            raise ValueError(f"Columna de texto '{settings.TEXT_COLUMN}' no encontrada en el CSV.")
        if settings.LABEL_COLUMN not in df.columns:
            raise ValueError(f"Columna de etiquetas '{settings.LABEL_COLUMN}' no encontrada en el CSV.")
        
        # Opcional: Limpieza básica o renombramiento de columnas si es necesario
        df = df[[settings.TEXT_COLUMN, settings.LABEL_COLUMN]]
        df = df.dropna().reset_index(drop=True) # Eliminar filas con valores nulos
        logger.info(f"Datos después de limpieza básica: Filas: {len(df)}")
        return df
    except Exception as e:
        logger.error(f"Error al cargar o procesar el archivo CSV: {e}")
        raise

def split_data(df: pd.DataFrame, test_size: float = 0.2, val_size: float = 0.1, random_state: int = 42) -> DatasetDict:

    logger.info(f"Dividiendo datos con test_size={test_size}, val_size={val_size}...")

    # Primero, separamos un conjunto combinado de validación/prueba del entrenamiento
    train_val_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state, stratify=df[settings.LABEL_COLUMN])
    
    # Luego, separamos el conjunto de validación del conjunto combinado de entrenamiento/validación
    # Ajustamos val_size para que sea una proporción del train_val_df
    val_proportion_of_train_val = val_size / (1 - test_size)
    if val_proportion_of_train_val > 0.0: # Asegurarse de que val_size sea > 0 para evitar error en split
        train_df, val_df = train_test_split(train_val_df, test_size=val_proportion_of_train_val, random_state=random_state, stratify=train_val_df[settings.LABEL_COLUMN])
    else: # Si val_size es 0, todo train_val_df es entrenamiento
        train_df = train_val_df
        val_df = pd.DataFrame(columns=df.columns) # DataFrame vacío para validación si no hay

    logger.info(f"Tamaño de los splits: Entrenamiento={len(train_df)}, Validación={len(val_df)}, Prueba={len(test_df)}")

    # Convertir a datasets de Hugging Face
    raw_datasets = DatasetDict({
        'train': Dataset.from_pandas(train_df, preserve_index=False),
        'validation': Dataset.from_pandas(val_df, preserve_index=False),
        'test': Dataset.from_pandas(test_df, preserve_index=False)
    })
    return raw_datasets


def tokenize_function(examples, tokenizer, text_column, max_length):
    return tokenizer(examples[text_column], truncation=True, max_length=max_length)

def preprocess_and_tokenize(raw_datasets: DatasetDict, model_name: str, text_column: str, max_length: int) -> DatasetDict:
    logger.info(f"Cargando tokenizer para '{model_name}' y tokenizando datasets...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        logger.info("Tokenizer cargado.")
    except Exception as e:
        logger.error(f"Error al cargar el tokenizer '{model_name}': {e}")
        raise

    # Mapear las etiquetas a enteros si aún no lo están y definir num_labels
    # Esto es crucial para la clasificación con Hugging Face Trainer
    unique_labels = sorted(list(set(raw_datasets['train'][settings.LABEL_COLUMN])))
    label_to_id = {label: i for i, label in enumerate(unique_labels)}
    id_to_label = {i: label for label, i in label_to_id.items()}
    settings.NUM_LABELS = len(unique_labels) # Actualizar número de etiquetas en settings
    
    logger.info(f"Etiquetas detectadas: {unique_labels}")
    logger.info(f"Mapeo de etiquetas a IDs: {label_to_id}")
    logger.info(f"Número de etiquetas: {settings.NUM_LABELS}")


    def map_labels_to_ids(examples):
        # Asegúrate de que la columna de etiquetas sea 'labels' para el Trainer de HF
        examples['labels'] = [label_to_id[label] for label in examples[settings.LABEL_COLUMN]]
        return examples

    tokenized_datasets = raw_datasets.map(
        lambda examples: tokenize_function(examples, tokenizer, text_column, max_length),
        batched=True,
        desc="Tokenizando datos..."
    )
    
    # Aplicar mapeo de etiquetas y remover columnas originales si no son necesarias
    tokenized_datasets = tokenized_datasets.map(
        map_labels_to_ids,
        batched=True,
        desc="Mapeando etiquetas a IDs numéricos..."
    )

    columns_to_remove = [col for col in tokenized_datasets['train'].column_names if col not in ['input_ids', 'attention_mask', 'labels', 'token_type_ids']]
    tokenized_datasets = tokenized_datasets.remove_columns(columns_to_remove)
    
    logger.info("Tokenización y mapeo de etiquetas completados.")
    logger.info(f"Columnas finales en el dataset: {tokenized_datasets['train'].column_names}")
    return tokenized_datasets


def save_processed_data(tokenized_datasets: DatasetDict, output_dir: str):

    os.makedirs(output_dir, exist_ok=True)
    for split, dataset in tokenized_datasets.items():
        split_path = os.path.join(output_dir, f"{split}_dataset")
        dataset.save_to_disk(split_path)
        logger.info(f"Dataset '{split}' guardado en '{split_path}'.")

def run_preprocessing():
    # Rutas de archivos y configuraciones desde settings.py
    raw_data_path = settings.RAW_DATA_PATH
    processed_data_dir = settings.PROCESSED_DATA_DIR
    model_name = settings.MODEL_NAME
    text_column = settings.TEXT_COLUMN
    label_column = settings.LABEL_COLUMN # Aunque se mapeará a 'labels'
    max_length = settings.MAX_LENGTH
    test_size = settings.TEST_SPLIT_SIZE
    val_size = settings.VALIDATION_SPLIT_SIZE

    logger.info("Iniciando pipeline de preprocesamiento de datos...")
    df_raw = load_raw_data(raw_data_path)
    raw_datasets = split_data(df_raw, test_size=test_size, val_size=val_size, random_state=settings.SEED)
    tokenized_datasets = preprocess_and_tokenize(raw_datasets, model_name, text_column, max_length)
    save_processed_data(tokenized_datasets, processed_data_dir)

    logger.info("Preprocesamiento de datos completado exitosamente.")

if __name__ == "__main__":
    run_preprocessing()
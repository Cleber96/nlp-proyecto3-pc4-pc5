# evaluation/eval.py

import os
import torch
import logging
import json
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import load_from_disk, DatasetDict

# Importar módulos locales
from config import settings
from utils.metrics import compute_metrics

# Configuración de logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def evaluate_model(model_path: str = None, dataset_split: str = "validation"):
    logger.info(f"Iniciando el proceso de evaluación en el split: {dataset_split}...")

    # Determinar la ruta del modelo a evaluar
    if model_path is None:
        # Usamos el path donde se guarda el modelo final por defecto en train.py
        model_path = os.path.join(settings.CHECKPOINT_DIR, "final_model")
        logger.info(f"No se especificó model_path. Usando el modelo final por defecto en: {model_path}")
    else:
        logger.info(f"Evaluando modelo desde la ruta: {model_path}")

    # Cargar el tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        logger.info(f"Tokenizer cargado exitosamente desde: {model_path}")
    except Exception as e:
        logger.error(f"Error al cargar el tokenizer desde '{model_path}': {e}")
        return

    # Cargar el modelo para clasificación
    try:
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        logger.info(f"Modelo cargado exitosamente desde: {model_path}")
        # Mover el modelo a la GPU si está disponible
        if torch.cuda.is_available():
            model.to("cuda")
            logger.info("Modelo movido a GPU.")
        else:
            model.to("cpu")
            logger.info("Modelo cargado en CPU.")
    except Exception as e:
        logger.error(f"Error al cargar el modelo desde '{model_path}': {e}")
        return

    # Cargar el dataset para evaluación
    try:
        dataset_dir = os.path.join(settings.PROCESSED_DATA_DIR, f"{dataset_split}_dataset")
        eval_dataset = load_from_disk(dataset_dir)
        logger.info(f"Dataset '{dataset_split}' cargado. Contiene {len(eval_dataset)} muestras.")
        if isinstance(eval_dataset, DatasetDict):
            eval_dataset = eval_dataset[dataset_split] # Asegurar que es un Dataset simple
    except Exception as e:
        logger.error(f"Error al cargar el dataset procesado para '{dataset_split}' desde '{dataset_dir}': {e}")
        logger.error("Asegúrate de que '01_Data_Preprocessing_and_Tokenization.ipynb' haya sido ejecutado y el split exista.")
        return

    eval_output_dir = os.path.join(settings.LOGS_DIR, "evaluation_results", os.path.basename(model_path))
    os.makedirs(eval_output_dir, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=eval_output_dir, # Directorio donde se guardarán los resultados de la evaluación
        per_device_eval_batch_size=settings.EVAL_BATCH_SIZE,
        fp16=settings.USE_FP16, # Usa FP16 si se entrenó con él
        report_to="none", # Desactivar reportes adicionales para solo evaluación
        disable_tqdm=False,
        remove_unused_columns=False, # Importante si tu dataset tiene columnas extra
    )
    logger.info("TrainingArguments configurados para evaluación.")

    trainer = Trainer(
        model=model,
        args=training_args,
        eval_dataset=eval_dataset, # El dataset para la evaluación
        tokenizer=tokenizer,
        compute_metrics=compute_metrics, # Función para calcular métricas
    )
    logger.info("Trainer inicializado para evaluación.")

    try:
        # Realizar la evaluación
        evaluation_results = trainer.evaluate()
        logger.info("Evaluación completada.")
        logger.info("\n--- Resultados de la Evaluación ---")
        for key, value in evaluation_results.items():
            logger.info(f"{key}: {value:.4f}")
        logger.info("----------------------------------\n")

        # Guardar los resultados en un archivo JSON
        results_file_path = os.path.join(eval_output_dir, f"evaluation_metrics_{dataset_split}.json")
        with open(results_file_path, 'w') as f:
            json.dump(evaluation_results, f, indent=4)
        logger.info(f"Resultados de la evaluación guardados en: {results_file_path}")

    except Exception as e:
        logger.error(f"Error durante la ejecución de la evaluación: {e}")
        return

    logger.info("Proceso de evaluación finalizado.")


if __name__ == "__main__":
    evaluate_model(dataset_split="validation")

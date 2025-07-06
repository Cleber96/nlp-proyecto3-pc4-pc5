# run.py

import argparse
import logging
import os
import sys

from typing import Dict
from config import settings
from utils.preprocess import run_preprocessing
from training.train import train_model as train_main
from evaluation.eval import main as eval_main # Importa la función main del script eval.py

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Script de entrada para el proyecto de fine-tuning de Transformers.")

    parser.add_argument(
        "action",
        type=str,
        choices=["preprocess", "train", "evaluate", "all"],
        help="Acción a realizar: 'preprocess', 'train', 'evaluate', o 'all' (ejecuta todas en secuencia)."
    )
    
    # Argumento opcional para especificar el path del modelo a evaluar (solo para 'evaluate')
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Ruta al directorio del modelo entrenado para evaluación. Requerido para 'evaluate' si no es el último checkpoint."
    )

    args = parser.parse_args()
    settings.create_necessary_directories()
    if args.action == "preprocess" or args.action == "all":
        logger.info("Iniciando fase de preprocesamiento de datos...")
        try:
            # Código actual en run.py
            run_preprocessing()
            logger.info("Preprocesamiento de datos completado exitosamente.")
        except Exception as e:
            logger.error(f"Error durante el preprocesamiento de datos: {e}", exc_info=True)
            sys.exit(1) # Salir con código de error si falla

    if args.action == "train" or args.action == "all":
        logger.info("Iniciando fase de entrenamiento del modelo...")
        try:
            # train_main() orquesta el entrenamiento usando settings.py
            train_main() 
            logger.info("Entrenamiento del modelo completado exitosamente.")
        except Exception as e:
            logger.error(f"Error durante el entrenamiento del modelo: {e}", exc_info=True)
            sys.exit(1)

    if args.action == "evaluate" or args.action == "all":
        logger.info("Iniciando fase de evaluación del modelo...")
        evaluation_model_path = args.model_path
        if evaluation_model_path is None:
            try:
                checkpoints = [d for d in os.listdir(settings.CHECKPOINTS_DIR) if os.path.isdir(os.path.join(settings.CHECKPOINTS_DIR, d))]
                if checkpoints:
                    # Ordenar por fecha de modificación para obtener el más reciente
                    checkpoints.sort(key=lambda d: os.path.getmtime(os.path.join(settings.CHECKPOINTS_DIR, d)), reverse=True)
                    evaluation_model_path = os.path.join(settings.CHECKPOINTS_DIR, checkpoints[0])
                    logger.info(f"No se especificó --model_path. Usando el último checkpoint: {evaluation_model_path}")
                else:
                    logger.warning("No se encontraron checkpoints para evaluar. Por favor, entrena un modelo primero o especifica --model_path.")
                    sys.exit(1)
            except FileNotFoundError:
                logger.error(f"Directorio de checkpoints no encontrado: {settings.CHECKPOINTS_DIR}. Asegúrate de haber entrenado un modelo.")
                sys.exit(1)
        
        if evaluation_model_path:
            try:
                eval_main(
                    model_path=evaluation_model_path,
                    processed_data_dir=settings.PROCESSED_DATA_DIR,
                    model_name=settings.MODEL_NAME # Usar MODEL_NAME para cargar el tokenizador base
                )
                logger.info("Evaluación del modelo completada exitosamente.")
            except Exception as e:
                logger.error(f"Error durante la evaluación del modelo en {evaluation_model_path}: {e}", exc_info=True)
                sys.exit(1)

    logger.info("Todas las acciones solicitadas han sido completadas.")

if __name__ == "__main__":
    main()
# training/callbacks/custom_callbacks.py

import os
import json
import logging
from typing import List
from transformers import TrainerCallback, TrainingArguments, TrainerState, TrainerControl

# Configuración de logging para este módulo
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s') # Esto ya está en run.py, puedes quitarlo si solo quieres un logging global.

class CustomEarlyStoppingCallback(TrainerCallback):
    """
    Callback para detener el entrenamiento tempranamente si la métrica de monitoreo
    no mejora durante un número de épocas.
    """
    def __init__(self, early_stopping_patience: int = 3, early_stopping_threshold: float = 0.0):
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_threshold = early_stopping_threshold
        self.best_metric = -float('inf') # Asumimos que una métrica más alta es mejor (ej. F1, Accuracy)
        self.patience_counter = 0
        logger.info(f"Custom Early Stopping inicializado: paciencia={early_stopping_patience}, umbral={early_stopping_threshold}")

    def on_evaluate(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, metrics: dict, **kwargs):
        # Obtener la métrica configurada para el "mejor modelo" (ej. 'eval_f1', 'eval_accuracy')
        current_metric = metrics.get(args.metric_for_best_model)

        if current_metric is None:
            logger.warning(f"La métrica '{args.metric_for_best_model}' no se encontró para Early Stopping. Verifique TrainingArguments.")
            return

        # Si la métrica mejora (más allá del umbral)
        if current_metric > self.best_metric + self.early_stopping_threshold:
            logger.info(f"Nueva mejor métrica: {current_metric:.4f} (anterior mejor: {self.best_metric:.4f})")
            self.best_metric = current_metric
            self.patience_counter = 0 # Reiniciar contador de paciencia
        else:
            self.patience_counter += 1
            logger.info(f"Métrica no mejoró. Contador de paciencia: {self.patience_counter}/{self.early_stopping_patience}")

        # Si el contador de paciencia excede el límite, detener el entrenamiento
        if self.patience_counter >= self.early_stopping_patience:
            logger.info("¡Early stopping activado!")
            control.should_training_stop = True

class CustomLoggingCallback(TrainerCallback):
    """
    Callback para registrar métricas de entrenamiento y evaluación en un archivo JSON.
    """
    def __init__(self, log_file_path: str = "logs/run_history.json"):
        self.log_file_path = log_file_path
        self.run_history = []
        logger.info(f"Custom Logging Callback inicializado. Los logs se guardarán en: {log_file_path}")

    def on_log(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, logs=None, **kwargs):
        # Captura los logs cada vez que el Trainer los registra
        _logs = logs if logs is not None else {}
        _logs['step'] = state.global_step
        _logs['epoch'] = state.epoch
        self.run_history.append(_logs)

    def on_train_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        # Guarda todo el historial de entrenamiento al finalizar
        try:
            os.makedirs(os.path.dirname(self.log_file_path), exist_ok=True)
            with open(self.log_file_path, 'w', encoding='utf-8') as f:
                json.dump(self.run_history, f, indent=4)
            logger.info(f"Historial de entrenamiento guardado en {self.log_file_path}")
        except Exception as e:
            logger.error(f"Error al guardar el historial de entrenamiento en {self.log_file_path}: {e}", exc_info=True)


# Esta es la función que trainer_config.py espera importar
def get_custom_callbacks(
    early_stopping_patience: int = 3,
    early_stopping_threshold: float = 0.0,
    log_file_path: str = "logs/run_history.json"
) -> List[TrainerCallback]:
    """
    Devuelve una lista de instancias de callbacks personalizados.
    """
    callbacks = []
    callbacks.append(CustomEarlyStoppingCallback(
        early_stopping_patience=early_stopping_patience,
        early_stopping_threshold=early_stopping_threshold
    ))
    callbacks.append(CustomLoggingCallback(
        log_file_path=log_file_path
    ))
    logger.info("Callbacks personalizados creados: EarlyStopping, Logging.")
    return callbacks
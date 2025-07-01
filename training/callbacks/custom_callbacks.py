# training/callbacks/custom_callbacks.py

from transformers import TrainerCallback, EarlyStoppingCallback
from transformers.integrations import TensorBoardCallback
from transformers import TrainingArguments, TrainerState, TrainerControl
import logging
import json # Para logging de historial de ejecución
from typing import Dict, Any, Union

# Configuración básica de logging
logger = logging.getLogger(__name__)
# Asegura que el logger solo tenga un handler para evitar mensajes duplicados
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class CustomEarlyStoppingCallback(EarlyStoppingCallback):
    def __init__(self, early_stopping_patience: int = 3, early_stopping_threshold: float = 0.0):
        super().__init__(early_stopping_patience=early_stopping_patience,
                         early_stopping_threshold=early_stopping_threshold)
        logger.info(f"CustomEarlyStoppingCallback inicializado con paciencia={early_stopping_patience} y umbral={early_stopping_threshold}")

    def on_evaluate(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, metrics: Dict[str, float], **kwargs):
        # Llama al método on_evaluate del callback padre para la lógica de early stopping
        super().on_evaluate(args, state, control, metrics, **kwargs)

        # Añadir lógica de logging adicional
        if control.should_training_stop:
            logger.info(f"¡Early Stopping activado! El entrenamiento se detendrá en la época {state.epoch:.2f}.")
            # `state.best_metric` y `state.best_model_checkpoint` se actualizan por el EarlyStoppingCallback padre
            logger.info(f"Mejor métrica de validación hasta ahora: {state.best_metric:.4f}")
            if state.best_model_checkpoint:
                logger.info(f"Mejor modelo guardado en: {state.best_model_checkpoint}")
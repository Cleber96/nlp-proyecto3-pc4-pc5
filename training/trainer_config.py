# training/trainer_config.py
import os
import logging # Asegúrate de que logging esté importado
from transformers import TrainingArguments, TrainerCallback
from typing import List, Optional

# Importar callbacks personalizados (asegúrate de que estas rutas sean correctas)
from training.callbacks.custom_callbacks import get_custom_callbacks
from training.callbacks.fingerprint_callback import FingerprintCallback

# Importamos settings si es necesario, aunque en esta función
# estamos pasando todos los parámetros explícitamente.
from config import settings # Asegúrate de que esto esté importado aquí

logger = logging.getLogger(__name__) # Definir el logger aquí

def get_training_arguments(
    output_dir: str,
    logging_dir: str,
    per_device_train_batch_size: int,
    per_device_eval_batch_size: int,
    num_train_epochs: float,
    learning_rate: float,
    weight_decay: float,
    warmup_steps: int,
    save_strategy: str,
    evaluation_strategy: str,
    load_best_model_at_end: bool,
    metric_for_best_model: str,
    fp16: bool,
    gradient_accumulation_steps: int,
    gradient_checkpointing: bool,
    logging_steps: int,
    # ===> ¡Añade esta línea aquí! <===
    save_steps: int,
    save_total_limit: Optional[int],
    seed: int,
) -> TrainingArguments:
    """
    Configura y devuelve un objeto TrainingArguments para el Trainer de Hugging Face.
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(logging_dir, exist_ok=True)

    logger.info("Configurando argumentos de entrenamiento...")
    training_args = TrainingArguments(
        output_dir=output_dir,
        logging_dir=logging_dir,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        num_train_epochs=num_train_epochs,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        warmup_steps=warmup_steps,
        save_strategy=save_strategy,
        eval_strategy=evaluation_strategy,
        load_best_model_at_end=load_best_model_at_end,
        metric_for_best_model=metric_for_best_model,
        fp16=fp16,
        gradient_accumulation_steps=gradient_accumulation_steps,
        gradient_checkpointing=gradient_checkpointing,
        logging_steps=logging_steps,
        # ===> Y asegúrate de que se use aquí <===
        save_steps=save_steps,
        save_total_limit=save_total_limit,
        seed=seed,
        report_to="tensorboard",
        disable_tqdm=False,
        log_level="info",
        log_level_replica="info",
        logging_first_step=True,
    )
    return training_args


def get_all_callbacks(
    early_stopping_patience: int,
    early_stopping_threshold: float,
    log_file_path: str
) -> List[TrainerCallback]:
    """
    Configura y devuelve una lista de todos los callbacks a usar en el Trainer.
    """
    logger.info("Configurando callbacks para el entrenamiento...")
    callbacks = get_custom_callbacks(
        early_stopping_patience=early_stopping_patience,
        early_stopping_threshold=early_stopping_threshold,
        log_file_path=log_file_path
    )
    callbacks.append(FingerprintCallback())
    return callbacks
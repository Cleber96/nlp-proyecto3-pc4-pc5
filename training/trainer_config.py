# training/trainer_config.py

import os
from transformers import TrainingArguments, TrainerCallback
from typing import List, Optional

# Importar callbacks personalizados
from training.callbacks.custom_callbacks import get_custom_callbacks
from training.callbacks.fingerprint_callback import FingerprintCallback

def get_training_arguments(
    output_dir: str = "logs/checkpoints",
    logging_dir: str = "logs/tensorboard",
    per_device_train_batch_size: int = 8,
    per_device_eval_batch_size: int = 8,
    num_train_epochs: float = 3.0,
    learning_rate: float = 5e-5,
    weight_decay: float = 0.01,
    warmup_steps: int = 500,
    save_strategy: str = "epoch",  # "steps" o "epoch"
    evaluation_strategy: str = "epoch", # "steps" o "epoch"
    load_best_model_at_end: bool = True,
    metric_for_best_model: str = "f1",
    fp16: bool = False,
    gradient_accumulation_steps: int = 1,
    gradient_checkpointing: bool = False, # Para modelos muy grandes, intercambia velocidad por memoria
    logging_steps: int = 100, # Frecuencia de logging
    save_total_limit: Optional[int] = 1, # Limitar el número de checkpoints guardados
    seed: int = 42,
) -> TrainingArguments:
    # Asegurarse de que los directorios existan
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(logging_dir, exist_ok=True)

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
        evaluation_strategy=evaluation_strategy,
        load_best_model_at_end=load_best_model_at_end,
        metric_for_best_model=metric_for_best_model,
        fp16=fp16,
        gradient_accumulation_steps=gradient_accumulation_steps,
        gradient_checkpointing=gradient_checkpointing,
        logging_steps=logging_steps,
        save_total_limit=save_total_limit,
        seed=seed,
        report_to="tensorboard", # Asegura que los logs vayan a TensorBoard
        # Desactiva el logging interno de HF para dar más control a nuestros callbacks
        disable_tqdm=False, # Mantener barra de progreso
        log_level="info",
        log_level_replica="info",
        logging_first_step=True,
    )
    return training_args


def get_all_callbacks(
    early_stopping_patience: int = 3,
    early_stopping_threshold: float = 0.0,
    log_file_path: str = "logs/run_history.json"
) -> List[TrainerCallback]:
    callbacks = get_custom_callbacks(
        early_stopping_patience=early_stopping_patience,
        early_stopping_threshold=early_stopping_threshold,
        log_file_path=log_file_path
    )
    callbacks.append(FingerprintCallback())
    return callbacks
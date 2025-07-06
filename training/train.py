# training/train.py

import os
import torch
from typing import Dict, Union, List, Any
import logging
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainerCallback,
    TrainerState,
    TrainerControl
)
from datasets import load_from_disk, DatasetDict

# Import módulos locales
from config import settings
from utils.metrics import compute_metrics
from training.trainer_config import get_training_arguments, get_all_callbacks # <--- Asegúrate de importar estas funciones
from training.adversarial.awp import AWP
from training.adversarial.mixout import apply_mixout_to_model

# Configuración de logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class CustomTrainer(Trainer):
    def __init__(self, *args, awp_enabled: bool = False, awp_config: dict = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.awp_enabled = awp_enabled
        self.awp_config = awp_config if awp_config else {}
        self.awp_adversary = None
        if self.awp_enabled:
            logger.info("AWP habilitado. Inicializando AWP adversary.")

    def training_step(self, model: torch.nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        model.train()
        inputs = self._prepare_inputs(inputs)

        # Asegúrate de que self.args.device.type esté definido (e.g., "cuda" o "cpu")
        device_type = self.args.device.type if hasattr(self.args, 'device') and self.args.device is not None else ("cuda" if torch.cuda.is_available() else "cpu")

        with self.autocast_dtensor_enabled():
            # Paso 1: Forward pass y cálculo de la pérdida normal
            # Usa el device_type determinado
            with torch.autocast(device_type, dtype=torch.float16, enabled=self.args.fp16):
                outputs = model(**inputs)
                loss = self.compute_loss(model, outputs, inputs)

            if self.args.n_gpu > 1:
                loss = loss.mean() # Promedio si hay múltiples GPUs

            if self.args.gradient_accumulation_steps > 1:
                loss = loss / self.args.gradient_accumulation_steps

            # Paso 2: Backward pass para obtener los gradientes iniciales
            self.accelerator.backward(loss)

            # Paso 3: Aplicar AWP si está habilitado
            if self.awp_enabled and self.state.global_step >= self.awp_config.get('awp_start_step', 0): # Aplicar AWP a partir de cierto paso
                if self.awp_adversary is None:
                    self.awp_adversary = AWP(
                        model=model,
                        optimizer=self.optimizer,
                        adv_lr=self.awp_config.get('adv_lr', 0.001),
                        adv_eps=self.awp_config.get('adv_eps', 0.001),
                        param_name=self.awp_config.get('awp_emb_name', 'word_embeddings') # Usar un nombre de parámetro o una lista de nombres
                    )
                
                # Guardar los pesos originales antes de la perturbación
                self.awp_adversary._save()
                
                # Calcular y aplicar la perturbación adversaria
                self.awp_adversary.attack_step()

                # Paso 4: Recalcular la pérdida con los pesos perturbados (AWP loss)
                with torch.autocast(device_type, dtype=torch.float16, enabled=self.args.fp16):
                    outputs_awp = model(**inputs)
                    loss_awp = self.compute_loss(model, outputs_awp, inputs)

                if self.args.n_gpu > 1:
                    loss_awp = loss_awp.mean()

                if self.args.gradient_accumulation_steps > 1:
                    loss_awp = loss_awp / self.args.gradient_accumulation_steps
                    
                self.accelerator.backward(loss_awp)
                self.awp_adversary.restore() # Restaurar pesos para el siguiente forward pass

            return loss.detach() # Retornar la pérdida normal, pero la optimización incluye AWP

def train_model():
    logger.info("Iniciando el proceso de entrenamiento...")

    # Cargar los datos preprocesados
    try:
        train_dataset = load_from_disk(os.path.join(settings.PROCESSED_DATA_DIR, "train_dataset"))
        eval_dataset = load_from_disk(os.path.join(settings.PROCESSED_DATA_DIR, "validation_dataset"))
        logger.info(f"Datasets cargados. Entrenamiento: {len(train_dataset)} muestras, Validación: {len(eval_dataset)} muestras.")
        if isinstance(train_dataset, DatasetDict):
            train_dataset = train_dataset['train']
        if isinstance(eval_dataset, DatasetDict):
            eval_dataset = eval_dataset['validation']
    except Exception as e:
        logger.error(f"Error al cargar los datasets procesados: {e}")
        logger.error("Asegúrate de que '01_Data_Preprocessing_and_Tokenization.ipynb' haya sido ejecutado.")
        return

    # Cargar el tokenizer pre-entrenado
    try:
        tokenizer = AutoTokenizer.from_pretrained(settings.MODEL_NAME)
        logger.info(f"Tokenizer '{settings.MODEL_NAME}' cargado exitosamente.")
    except Exception as e:
        logger.error(f"Error al cargar el tokenizer '{settings.MODEL_NAME}': {e}")
        return

    # Cargar el modelo pre-entrenado para clasificación
    try:
        original_pretrained_model = AutoModelForSequenceClassification.from_pretrained(
            settings.MODEL_NAME,
            num_labels=settings.NUM_LABELS
        )
        model = AutoModelForSequenceClassification.from_pretrained(
            settings.MODEL_NAME,
            num_labels=settings.NUM_LABELS
        )
        logger.info(f"Modelo '{settings.MODEL_NAME}' cargado exitosamente para {settings.NUM_LABELS} etiquetas.")
    except Exception as e:
        logger.error(f"Error al cargar el modelo '{settings.MODEL_NAME}': {e}")
        return

    if settings.USE_MIXOUT:
        try:
            original_pretrained_model_copy = AutoModelForSequenceClassification.from_pretrained(
                settings.MODEL_NAME,
                num_labels=settings.NUM_LABELS
            )
            model = apply_mixout_to_model(model, original_pretrained_model_copy, settings.MIXOUT_PROBABILITY)
            logger.info(f"Mixout aplicado al modelo con probabilidad p={settings.MIXOUT_PROBABILITY}.")
        except Exception as e:
            logger.error(f"Error al aplicar Mixout: {e}")
            settings.USE_MIXOUT = False
            logger.warning("El entrenamiento continuará sin Mixout debido al error anterior.")
            
    # --- CÁLCULO DE WARMUP_STEPS AQUÍ ---
    # ESTO ES LO QUE ESTABA CAUSANDO EL ERROR 'WARMUP_STEPS'
    num_training_steps = int(len(train_dataset) / settings.PER_DEVICE_TRAIN_BATCH_SIZE * settings.NUM_TRAIN_EPOCHS)
    calculated_warmup_steps = int(num_training_steps * settings.WARMUP_RATIO) # Usar WARMUP_RATIO de settings

    logger.info(f"Número total de pasos de entrenamiento: {num_training_steps}")
    logger.info(f"Pasos de calentamiento (warmup): {calculated_warmup_steps}")

    training_args = get_training_arguments(
        output_dir=settings.CHECKPOINTS_DIR,
        logging_dir=settings.TENSORBOARD_LOG_DIR,
        per_device_train_batch_size=settings.PER_DEVICE_TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=settings.PER_DEVICE_EVAL_BATCH_SIZE,
        num_train_epochs=settings.NUM_TRAIN_EPOCHS,
        learning_rate=settings.LEARNING_RATE,
        weight_decay=settings.WEIGHT_DECAY,
        warmup_steps=calculated_warmup_steps, # <--- ¡AHORA ESTO ES CORRECTO!
        save_strategy=settings.SAVE_STRATEGY,
        evaluation_strategy=settings.EVALUATION_STRATEGY,
        load_best_model_at_end=settings.LOAD_BEST_MODEL_AT_END,
        metric_for_best_model=settings.METRIC_FOR_BEST_MODEL,
        fp16=settings.FP16,
        gradient_accumulation_steps=settings.GRADIENT_ACCUMULATION_STEPS,
        gradient_checkpointing=settings.USE_GRADIENT_CHECKPOINTING,
        logging_steps=settings.LOGGING_STEPS,
        save_steps=settings.SAVE_STEPS,
        save_total_limit=settings.SAVE_TOTAL_LIMIT,
        seed=settings.SEED,
    )

    all_callbacks = get_all_callbacks(
        early_stopping_patience=settings.EARLY_STOPPING_PATIENCE,
        early_stopping_threshold=settings.EARLY_STOPPING_THRESHOLD,
        log_file_path=settings.RUN_HISTORY_PATH
    )

    trainer_class = CustomTrainer if settings.USE_AWP else Trainer
    awp_config = {
        'adv_lr': settings.AWP_LR,
        'adv_eps': settings.AWP_EPS,
        'awp_start_step': settings.AWP_START_STEP,
        'awp_emb_name': settings.AWP_EMB_NAME,
    }

    # Diccionario para almacenar los argumentos comunes del Trainer
    common_trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
        "tokenizer": tokenizer,
        "compute_metrics": compute_metrics,
        "callbacks": all_callbacks,
    }

    if settings.USE_AWP:
        # Si AWP está habilitado, añade los argumentos específicos de AWP
        # SOLO si estamos usando CustomTrainer
        trainer = CustomTrainer(
            **common_trainer_kwargs,
            awp_enabled=settings.USE_AWP,
            awp_config=awp_config,
        )
        logger.info("CustomTrainer con AWP habilitado.")
    else:
        # Si AWP no está habilitado, usa el Trainer base sin argumentos AWP
        trainer = Trainer(
            **common_trainer_kwargs
        )
        logger.info("Trainer base de Hugging Face en uso.")

    train_result = None
    try:
        resume_from_checkpoint = training_args.resume_from_checkpoint
        if resume_from_checkpoint and not os.path.exists(resume_from_checkpoint):
            logger.warning(f"El checkpoint {resume_from_checkpoint} no existe. Ignorando la reanudación.")
            resume_from_checkpoint = None

        logger.info("Iniciando el entrenamiento...")
        train_result = trainer.train(resume_from_checkpoint=resume_from_checkpoint)
        logger.info("Entrenamiento completado.")

    except torch.cuda.OutOfMemoryError as e:
        logger.error(f"¡ERROR: CUDA Out of Memory! Considera reducir el batch_size, aumentar gradient_accumulation_steps, o usar gradient_checkpointing/fp16.")
        logger.error(f"Detalles del error: {e}")
        return
    except Exception as e:
        logger.error(f"Ocurrió un error inesperado durante el entrenamiento: {e}", exc_info=True)
        raise

    if train_result:
        final_model_path = os.path.join(settings.CHECKPOINTS_DIR, "final_model")
        trainer.save_model(final_model_path)
        tokenizer.save_pretrained(final_model_path)
        logger.info(f"Modelo y tokenizer finales guardados en: {final_model_path}")

        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

        logger.info("Proceso de entrenamiento finalizado.")
    else:
        logger.warning("El entrenamiento no se completó exitosamente o no se generó un resultado de entrenamiento.")

if __name__ == "__main__":
    train_model()
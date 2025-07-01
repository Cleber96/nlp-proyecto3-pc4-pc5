# training/train.py

import os
import torch
import logging
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    TrainerCallback,
    TrainerState,
    TrainerControl
)
from datasets import load_from_disk, DatasetDict

# Import módulos locales
from config import settings
from utils.metrics import compute_metrics
from training.trainer_config import get_training_arguments, get_all_callbacks
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

        with self.autocast_dtensor_enabled():
            # Paso 1: Forward pass y cálculo de la pérdida normal
            with torch.autocast(self.args.device.type, dtype=torch.float16, enabled=self.args.fp16):
                outputs = model(**inputs)
                loss = self.compute_loss(model, outputs, inputs)

            if self.args.n_gpu > 1:
                loss = loss.mean() # Promedio si hay múltiples GPUs

            # Si se usa acumulación de gradientes, la pérdida se escala automáticamente por HF Trainer
            # para que el gradiente sea correcto al final de la acumulación.
            if self.args.gradient_accumulation_steps > 1:
                loss = loss / self.args.gradient_accumulation_steps

            # Paso 2: Backward pass para obtener los gradientes iniciales
            self.accelerator.backward(loss)

            # Paso 3: Aplicar AWP si está habilitado
            if self.awp_enabled:
                if self.awp_adversary is None:
                    # Inicializar AWP con el optimizador actual del Trainer
                    # Accedemos al optimizador que el Trainer ha preparado
                    self.awp_adversary = AWP(
                        model=model,
                        optimizer=self.optimizer,
                        adv_lr=self.awp_config.get('adv_lr', 0.001),
                        adv_eps=self.awp_config.get('adv_eps', 0.001)
                    )
                
                # Guardar los pesos originales antes de la perturbación
                self.awp_adversary._save()
                
                # Calcular y aplicar la perturbación adversaria
                self.awp_adversary.attack_step()

                # Paso 4: Recalcular la pérdida con los pesos perturbados (AWP loss)
                with torch.autocast(self.args.device.type, dtype=torch.float16, enabled=self.args.fp16):
                    outputs_awp = model(**inputs)
                    loss_awp = self.compute_loss(model, outputs_awp, inputs)

                if self.args.n_gpu > 1:
                    loss_awp = loss_awp.mean()

                if self.args.gradient_accumulation_steps > 1:
                    loss_awp = loss_awp / self.args.gradient_accumulation_steps
                    
                self.accelerator.backward(loss_awp)
                self.awp_adversary.restore() # Restaurar pesos para el siguiente forward pass

            # El Trainer maneja automáticamente el paso del optimizador y el scheduler
            return loss.detach() # Retornar la pérdida normal, pero la optimización incluye AWP

def train_model():
    logger.info("Iniciando el proceso de entrenamiento...")

    # Cargar los datos preprocesados
    try:
        train_dataset = load_from_disk(os.path.join(settings.PROCESSED_DATA_DIR, "train_dataset"))
        eval_dataset = load_from_disk(os.path.join(settings.PROCESSED_DATA_DIR, "validation_dataset"))
        logger.info(f"Datasets cargados. Entrenamiento: {len(train_dataset)} muestras, Validación: {len(eval_dataset)} muestras.")
        # Asegurarse de que son de tipo Dataset (no DatasetDict si se guardó así)
        if isinstance(train_dataset, DatasetDict):
            train_dataset = train_dataset['train']
        if isinstance(eval_dataset, DatasetDict):
            eval_dataset = eval_dataset['validation'] # O 'test' si se usa así
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
        # Se carga el modelo inicial para Mixout (si está habilitado)
        original_pretrained_model = AutoModelForSequenceClassification.from_pretrained(
            settings.MODEL_NAME,
            num_labels=settings.NUM_LABELS
        )
        # El modelo que será entrenado
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
            # Asegurarse de que original_pretrained_model tenga el mismo estado inicial
            # Esto es clave para que Mixout funcione correctamente
            original_pretrained_model_copy = AutoModelForSequenceClassification.from_pretrained(
                settings.MODEL_NAME,
                num_labels=settings.NUM_LABELS
            )
            model = apply_mixout_to_model(model, original_pretrained_model_copy, settings.MIXOUT_PROB)
            logger.info(f"Mixout aplicado al modelo con probabilidad p={settings.MIXOUT_PROB}.")
        except Exception as e:
            logger.error(f"Error al aplicar Mixout: {e}")
            # Considerar si se debe detener el entrenamiento o continuar sin Mixout
            settings.USE_MIXOUT = False # Deshabilitar si falla
            logger.warning("El entrenamiento continuará sin Mixout debido al error anterior.")
            
    training_args = get_training_arguments(
        output_dir=settings.CHECKPOINT_DIR,
        logging_dir=settings.TENSORBOARD_LOG_DIR,
        per_device_train_batch_size=settings.TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=settings.EVAL_BATCH_SIZE,
        num_train_epochs=settings.NUM_TRAIN_EPOCHS,
        learning_rate=settings.LEARNING_RATE,
        weight_decay=settings.WEIGHT_DECAY,
        warmup_steps=settings.WARMUP_STEPS,
        save_strategy=settings.SAVE_STRATEGY,
        evaluation_strategy=settings.EVALUATION_STRATEGY,
        load_best_model_at_end=settings.LOAD_BEST_MODEL_AT_END,
        metric_for_best_model=settings.METRIC_FOR_BEST_MODEL,
        fp16=settings.USE_FP16,
        gradient_accumulation_steps=settings.GRADIENT_ACCUMULATION_STEPS,
        gradient_checkpointing=settings.USE_GRADIENT_CHECKPOINTING,
        logging_steps=settings.LOGGING_STEPS,
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
        'adv_lr': settings.AWP_ADV_LR,
        'adv_eps': settings.AWP_ADV_EPS
    }

    trainer = trainer_class(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics, # Función para calcular métricas
        callbacks=all_callbacks,
        awp_enabled=settings.USE_AWP, # Solo para CustomTrainer
        awp_config=awp_config,        # Solo para CustomTrainer
    )

    train_result = None
    try:
        # Reanudar desde un checkpoint si se especifica
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
        # Aquí podrías añadir lógica para guardar el estado, ajustar parámetros y reintentar
        # Esto es más complejo y a menudo se delega a frameworks como Accelerate
        # Por ahora, simplemente salimos o puedes añadir un sys.exit(1)
        return
    except Exception as e:
        logger.error(f"Ocurrió un error inesperado durante el entrenamiento: {e}")
        return

    if train_result:
        # Guarda el modelo final entrenado (que será el mejor si load_best_model_at_end=True)
        final_model_path = os.path.join(settings.CHECKPOINT_DIR, "final_model")
        trainer.save_model(final_model_path)
        tokenizer.save_pretrained(final_model_path)
        logger.info(f"Modelo y tokenizer finales guardados en: {final_model_path}")

        # Puedes también guardar el estado del entrenamiento (métricas, etc.) si no lo hizo el callback
        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

        logger.info("Proceso de entrenamiento finalizado.")
    else:
        logger.warning("El entrenamiento no se completó exitosamente o no se generó un resultado de entrenamiento.")

if __name__ == "__main__":
    train_model()
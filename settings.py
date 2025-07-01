# config/settings.py

import os
import torch

# Obtiene el directorio base del proyecto (una carpeta arriba de 'config')
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Directorios de datos
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")

# Directorios de logs y checkpoints
LOGS_DIR = os.path.join(PROJECT_ROOT, "logs")
CHECKPOINTS_DIR = os.path.join(LOGS_DIR, "checkpoints")
TENSORBOARD_LOG_DIR = os.path.join(LOGS_DIR, "tensorboard")
RUN_HISTORY_PATH = os.path.join(LOGS_DIR, "run_history.json")

# Directorio para modelos guardados (por ejemplo, el modelo final)
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
# Nombre del archivo de reseñas original en 'data/raw'
RAW_DATA_FILENAME = "product_reviews.csv"

# Checkpoint del modelo de Hugging Face a utilizar para fine-tuning
# Ejemplos: "bert-base-uncased", "roberta-base", "microsoft/mdeberta-v3-base"
MODEL_CHECKPOINT = "distilbert-base-uncased" # Un modelo más ligero para empezar

# Longitud máxima de secuencia para la tokenización
MAX_SEQ_LENGTH = 128

# Proporción para dividir el dataset de entrenamiento en (train, validation)
TRAIN_VALIDATION_SPLIT_RATIO = 0.9 # 90% para entrenamiento, 10% para validación

# Semilla aleatoria para reproducibilidad (data splitting, inicialización de pesos, etc.)
RANDOM_SEED = 42

# Número total de épocas para entrenar
NUM_TRAIN_EPOCHS = 3.0

# Tamaño de batch por dispositivo (GPU/CPU) para entrenamiento
PER_DEVICE_TRAIN_BATCH_SIZE = 16

# Tamaño de batch por dispositivo (GPU/CPU) para evaluación
PER_DEVICE_EVAL_BATCH_SIZE = 32

# Tasa de aprendizaje inicial
LEARNING_RATE = 2e-5

# Decaimiento de pesos para la regularización L2
WEIGHT_DECAY = 0.01

# Proporción de pasos de calentamiento (warmup) sobre el total de pasos
# Reduce la tasa de aprendizaje al inicio para estabilizar el entrenamiento.
WARMUP_RATIO = 0.1

# Frecuencia de loggeo (en pasos)
LOGGING_STEPS = 500

# Frecuencia para guardar checkpoints (en pasos)
SAVE_STEPS = 500

# Estrategia de evaluación (ej. "steps", "epoch", "no")
EVALUATION_STRATEGY = "steps"

# Estrategia para guardar el modelo (ej. "steps", "epoch", "no")
SAVE_STRATEGY = "steps"

# Cargar el mejor modelo (basado en 'metric_for_best_model') al final del entrenamiento
LOAD_BEST_MODEL_AT_END = True

# Métrica utilizada para determinar el "mejor" modelo
# Debe coincidir con una métrica computada en utils/metrics.py
METRIC_FOR_BEST_MODEL = "eval_f1"

# Modo de optimización (ej. "max" para métricas como F1, "min" para pérdida)
GREATER_IS_BETTER = True # True si una mayor 'eval_f1' es mejor
GRADIENT_ACCUMULATION_STEPS = 1

# Usar entrenamiento de precisión mixta (float16) si hay GPU disponible
FP16 = torch.cuda.is_available()
USE_AWP = False
AWP_LR = 1e-4             # Tasa de aprendizaje para la perturbación AWP
AWP_EPS = 1e-6            # Magnitud de la perturbación epsilon
AWP_START_STEP = 500      # Paso a partir del cual aplicar AWP
AWP_EMB_NAME = "distilbert.embeddings.word_embeddings.weight"

USE_MIXOUT = False
MIXOUT_PROBABILITY = 0.7 # Probabilidad 'p' para Mixout (similar a dropout)


def create_necessary_directories():
    os.makedirs(RAW_DATA_DIR, exist_ok=True)
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
    os.makedirs(TENSORBOARD_LOG_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(LOGS_DIR, exist_ok=True) # Asegura que el dir. de run_history.json exista
    print("Directorios del proyecto verificados/creados.")

# if __name__ == "__main__":
#     create_necessary_directories()
#     print(f"Ruta raíz del proyecto: {PROJECT_ROOT}")
#     print(f"Modelo de checkpoint: {MODEL_CHECKPOINT}")
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
# **NUEVO**: Ruta completa al archivo de datos raw
RAW_DATA_PATH = os.path.join(RAW_DATA_DIR, RAW_DATA_FILENAME)

# **NUEVO**: Nombres de las columnas de texto y etiqueta en tu CSV raw
TEXT_COLUMN = "text"  # ¡Asegúrate que coincida con la columna de texto de tu CSV!
LABEL_COLUMN = "label" # ¡Asegúrate que coincida con la columna de etiqueta de tu CSV!

# Checkpoint del modelo de Hugging Face a utilizar para fine-tuning
# Ejemplos: "bert-base-uncased", "roberta-base", "microsoft/mdeberta-v3-base"
MODEL_CHECKPOINT = "distilbert-base-uncased" # Un modelo más ligero para empezar
# **NUEVO**: Usamos MODEL_CHECKPOINT como MODEL_NAME para consistencia
MODEL_NAME = MODEL_CHECKPOINT

# Longitud máxima de secuencia para la tokenización
MAX_SEQ_LENGTH = 128
# **NUEVO**: Usamos MAX_SEQ_LENGTH como MAX_LENGTH para consistencia
MAX_LENGTH = MAX_SEQ_LENGTH

# **MODIFICADO/NUEVO**: Proporciones para dividir el dataset
# Estas son las que split_data en preprocess.py espera
TEST_SPLIT_SIZE = 0.2     # 20% para el conjunto de prueba
VALIDATION_SPLIT_SIZE = 0.1 # 10% para el conjunto de validación (del resto después de test)
# TRAIN_VALIDATION_SPLIT_RATIO (Tu variable original, ya no es usada directamente por split_data)
# TRAIN_VALIDATION_SPLIT_RATIO = 0.9 # (Puedes mantenerla o borrarla si no la usas más)

# Semilla aleatoria para reproducibilidad (data splitting, inicialización de pesos, etc.)
RANDOM_SEED = 42
# **NUEVO**: Usamos RANDOM_SEED como SEED para consistencia
SEED = RANDOM_SEED

# **NUEVO**: Número de etiquetas (clases) en tu dataset. Se actualizará dinámicamente en preprocess.py.
# Inicialízalo con un valor por defecto (ej. 2 para binario)
NUM_LABELS = 2 # Valor por defecto. preprocess.py lo detectará y actualizará.

# Número total de épocas para entrenar
NUM_TRAIN_EPOCHS = 2

# Tamaño de batch por dispositivo (GPU/CPU) para entrenamiento
PER_DEVICE_TRAIN_BATCH_SIZE = 8

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
SAVE_TOTAL_LIMIT = 1
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
USE_GRADIENT_CHECKPOINTING = False
GRADIENT_ACCUMULATION_STEPS = 2

# Usar entrenamiento de precisión mixta (float16) si hay GPU disponible
FP16 = torch.cuda.is_available()

# --- Configuración de Early Stopping ---
EARLY_STOPPING_PATIENCE = 3   # Número de épocas a esperar si no hay mejora
EARLY_STOPPING_THRESHOLD = 0.001 # Mejora mínima requerida para considerarse "mejora"

# --- Configuración de AWP (Adversarial Weight Perturbation) ---
USE_AWP = True
AWP_LR = 1e-4   
AWP_EPS = 0.01   
AWP_START_STEP = 0 
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
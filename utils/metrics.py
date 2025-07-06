# utils/metrics.py

import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import EvalPrediction
from typing import Dict
import logging

# Configuración de logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def compute_metrics(p: EvalPrediction) -> Dict[str, float]:
    logger.debug("Calculando métricas...")
    
    # Obtener las predicciones de clase (el índice del logit más alto)
    predictions = np.argmax(p.predictions, axis=1)
    
    # Obtener las etiquetas verdaderas
    labels = p.label_ids
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='weighted')

    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }
    
    logger.debug(f"Métricas calculadas: {metrics}")
    return metrics
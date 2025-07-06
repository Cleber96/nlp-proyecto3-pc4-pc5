# utils/metrics.py

import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from transformers import EvalPrediction
from typing import Dict
import logging
import matplotlib.pyplot as plt
import seaborn as sns 
import os 

# Configuración de logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, labels: list, output_path: str):
    # Calcula la matriz de confusión
    cm = confusion_matrix(y_true, y_pred)
    
    # Configura el plot
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    
    plt.title("Matriz de Confusión")
    plt.xlabel("Predicción")
    plt.ylabel("Verdadero")
    plt.tight_layout() # Ajusta el diseño para evitar recortes
    
    # Guarda la figura
    plt.savefig(output_path)
    plt.close() # Cierra la figura para liberar memoria
    logger.info(f"Matriz de confusión guardada en: {output_path}")

def compute_metrics(p: EvalPrediction) -> Dict[str, float]:
    logger.debug("Calculando métricas...")
    
    # Obtener las predicciones de clase (el índice del logit más alto)
    predictions = np.argmax(p.predictions, axis=1)
    
    # Obtener las etiquetas verdaderas
    labels = p.label_ids
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='weighted', zero_division=0) 

    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }
    
    logger.debug(f"Métricas calculadas: {metrics}")
    return metrics
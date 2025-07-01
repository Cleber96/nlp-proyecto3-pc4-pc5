# PROYECTO 03 :Fine-tuning de Transformers con Hugging Face y Técnicas Avanzadas

## Visión General del Proyecto

Este proyecto es una implementación completa y paso a paso para realizar **fine-tuning de modelos Transformers** utilizando la popular librería [Hugging Face Transformers](https://huggingface.co/docs/transformers/index) y [Hugging Face Datasets](https://huggingface.co/docs/datasets/index). Va más allá del fine-tuning básico, incorporando técnicas de regularización avanzadas implementadas manualmente, como **Adversarial Weight Perturbation (AWP)** y **Mixout**, junto con mecanismos robustos de manejo de entrenamiento y pruebas de integridad de código.

El objetivo principal es proporcionar un flujo de trabajo estructurado para el fine-tuning de modelos para tareas de clasificación de texto (como el análisis de reseñas de productos), demostrando un control granular sobre el proceso de entrenamiento, la configuración de `Trainer`, los callbacks personalizados y la gestión de recursos.

## Objetivos del Proyecto

Los objetivos clave de este proyecto son:

1.  **Preprocesamiento de Datos**: Tokenizar y preparar un corpus pequeño (ej. reseñas de productos) utilizando la librería `datasets` y los tokenizadores "fast" de Hugging Face.
2.  **Configuración Avanzada del `Trainer`**:
    * Parametrizar `TrainingArguments` (tamaño de batch, tasa de aprendizaje, épocas, etc.) desde un archivo de configuración central.
    * Implementar **callbacks personalizados** para funcionalidades como Early Stopping y logging detallado.
    * Integrar **métricas de evaluación personalizadas** (precisión, recall, F1-score) usando la función `compute_metrics`.
3.  **Regularización Manual Avanzada**:
    * Desarrollar e integrar **AWP (Adversarial Weight Perturbation)** de forma manual, sin depender de librerías externas pre-existentes para esta técnica.
    * Implementar **Mixout** también de forma manual, como una alternativa al dropout tradicional, dentro de las capas del modelo.
    * Garantizar que estas técnicas no interfieran con el `scheduler` de optimización.
4.  **Robustez en el Entrenamiento**: Añadir lógica para el manejo de **acumulación de gradientes** y detección de errores de **OOM (Out-Of-Memory)** con capacidades de reinicio suave del entrenamiento.
5.  **Pruebas de Integridad y Replicabilidad**: Incluir scripts de pruebas de integridad de código basados en AST para validar la "originalidad" de las implementaciones manuales, y asegurar que el proyecto sea fácilmente replicable.
6.  **Fingerprinting de Ejecución**: Añadir un hash único (HMAC) en los checkpoints o logs al guardar, vinculando la ejecución al equipo/entorno.

## Estructura del Repositorio
``` markdown
fine_tuning_transformers/
├── data/
│   ├── raw/
│   │   └── product_reviews.csv              # Corpus original de reseñas (o cualquier otro formato)
│   └── processed/
│       ├── train_dataset/                   # Dataset de entrenamiento en formato .arrow o similar
│       └── validation_dataset/              # Dataset de validación en formato .arrow o similar
│
├── training/
│   ├── __init__.py
│   ├── adversarial/
│   │   ├── __init__.py
│   │   ├── awp.py                           # Implementación manual de Adversarial Weight Perturbation (AWP)
│   │   └── mixout.py                        # Implementación manual de Mixout
│   │
│   ├── callbacks/
│   │   ├── __init__.py
│   │   ├── custom_callbacks.py              # Callbacks personalizados (EarlyStopping, logging, etc.)
│   │   └── fingerprint_callback.py          # Callback para añadir el hash de equipo a los checkpoints
│   │
│   ├── trainer_config.py                    # Funciones para configurar TrainingArguments y callbacks
│   └── train.py                             # Script principal para orquestar el entrenamiento
│
├── models/
│   ├── __init__.py
│   └── custom_model.py                      # (Opcional) Archivo para definir un modelo personalizado si es necesario
│
├── evaluation/
│   ├── __init__.py
│   └── eval.py                              # Script para evaluar el modelo en el test set
│
├── utils/
│   ├── __init__.py
│   ├── preprocess.py                        # Funciones de tokenización, limpieza y split del dataset
│   └── metrics.py                           # Función compute_metrics para calcular precisión y recall
│
├── logs/
│   ├── checkpoints/                         # Checkpoints del modelo guardados por el Trainer
│   ├── tensorboard/                         # Archivos de TensorBoard (.tfevents)
│   └── run_history.json                     # (Opcional) Log de la ejecución con parámetros y métricas finales
│
├── notebooks/
│   ├── 00_Project_Setup_and_Overview.ipynb  # Configuración del entorno, overview del proyecto y librerías
│   ├── 01_Data_Preprocessing_and_Tokenization.ipynb # Teoría y aplicación de `datasets` y tokenizadores "fast"
│   ├── 02_Trainer_Configuration_and_Basic_Training.ipynb # Teoría de `Trainer`, `TrainingArguments` y entrenamiento base
│   ├── 03_Custom_Callbacks_and_Metrics.ipynb # Teoría y aplicación de `EarlyStopping`, logging y `compute_metrics`
│   ├── 04_Adversarial_Weight_Perturbation_(AWP).ipynb # Teoría de AWP y su implementación manual
│   ├── 05_Mixout_Implementation.ipynb       # Teoría de Mixout y su implementación manual
│   ├── 06_Gradient_Accumulation_and_OOM_Handling.ipynb # Teoría y manejo de acumulación de gradientes y OOM
│   └── 07_Full_Training_Orchestration_and_Evaluation.ipynb # Integración de todos los componentes y script de evaluación
│
├── scripts/
│   ├── train_script.sh                      # Script para ejecutar training/train.py desde la terminal
│   └── evaluate_script.sh                   # Script para ejecutar evaluation/eval.py
│
├── tests/
│   ├── __init__.py
│   ├── check_originality.py                 # Script basado en AST para pruebas de integridad
│   ├── test_awp.py                          # Pruebas unitarias para la implementación de AWP
│   ├── test_mixout.py                       # Pruebas unitarias para la implementación de Mixout
│   └── test_callbacks.py                    # Pruebas unitarias para los callbacks personalizados
│
├── config/
│   └── settings.py                          # Archivo de configuración para parámetros (batch size, LR, etc.)
│
├── .gitignore
├── README.md
├── requirements.txt
└── run.py                                   # Script de entrada para ejecutar el proyecto
```
## REPLICAR proyecto
1. Crear un entorno virtual (si no tienes uno)
python -m venv venv

2. Activar el entorno virtual
- En Windows:
```bash
.\venv\Scripts\activate
```
- En macOS/Linux:
```bash
source venv/bin/activate
```

3. Instalar las dependencias
```bash
pip install -r requirements.txt
```
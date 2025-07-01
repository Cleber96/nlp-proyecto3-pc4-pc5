## Miniproyectos CC0C2

**Estructura de evaluación sugerida:**
* **Proyecto (código, cuaderno de resultados):** Hasta 8 puntos.
* **Presentación (claridad, análisis, demostración, vídeo):** Hasta 12 punto (este punto es crucial y refleja la priorización de la exposición).
* **Entrega la dirección del repositorio de desarrollo**.
* **Fecha de presentación:** 3 de julio desde las 16:00 hrs.

#### Controles generales para todos los proyectos

1. **Pruebas de integridad de código**: scripts basados en AST (`tests/check_originality.py`) que evalúan similitudes estructurales con soluciones externas.
2. **Presentación y defensa**: demo en vivo o video corto donde el equipo explica el flujo interno.
3. **Callbacks y tests personalizados**: cada proyecto incluye ejercicios teóricos (`.md`) y pruebas unitarias que exigen respuestas y comportamientos personalizados.
4. **Fingerprinting de ejecución**: al guardar checkpoints o logs se añade un hash único (HMAC) vinculado al equipo.

### Proyecto 3: Fine-tuning con Transformers y Hugging Face

**Contexto y motivación**
El ecosistema Hugging Face simplifica enormemente el fine-tuning de modelos pre-entrenados. Este proyecto guía al estudiante por todo el flujo: desde la tokenización hasta el entrenamiento supervisado con callbacks y métricas personalizadas.

**Objetivos**

1. Preprocesar un corpus pequeño (p. ej. reseñas de producto) usando `datasets` y el tokenizador "fast" elegido.
2. Configurar un `Trainer` con:

   * `TrainingArguments` parametrizables (batch size, learning rate, epochs).
   * Callbacks para early stopping y logging personalizado.
   * Métricas de precisión y recall usando `compute_metrics`.
3. Implementar AWP (Adversarial Weight Perturbation) y mixout de forma manual, sin librerías externas, dentro del loop de entrenamiento.
4. Añadir manejo de gradiente acumulado y detección de OOM con reinicio suave del entrenamiento.

**Entregables**

* `data/preprocess.py`: tokenización, limpieza y split del dataset.
* `training/trainer_config.py`: funciones que devuelven `TrainingArguments` y listas de callbacks.
* `training/train.py`: orquestador principal que carga data, modelo (`AutoModelForSequenceClassification`) y lanza `Trainer`.
* `evaluation/eval.py`: script para evaluar en test set, generar matriz de confusión y reportes de classification.
* `logs/`:

  * Checkpoints guardados automáticamente.
  * Archivos de TensorBoard (`.tfevents`).
* **Vídeo de demostración**: \~10 min mostrando la ejecución de `train.py`, visualización de métricas en TensorBoard y un ejemplo de inferencia con `pipeline`.

**Retos clave**

* Integrar AWP y mixout controlando que no interfieran con el scheduler de optimización.
* Diseñar un callback de early stopping sencillo.
* Balancear batch size y gradiente acumulado para evitar OOM en GPUs limitadas.
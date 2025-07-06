# **Práctica calificada 5 CC0C2**
## Examen 2: Fine-tuning de Transformers con Hugging Face y Técnicas Avanzadas  
**Duración:** 3 horas  
**Repositorio base:** [nlp-proyecto3-pc4-pc5](https://github.com/Cleber96/nlp-proyecto3-pc4-pc5.git)

### Tokenización y DataLoader  
- En `data/preprocess.py`, ajusta la tokenización para truncar a 128 tokens y pad dinámico.  
- Describe cómo habilitarías `gradient_accumulation` en un entorno con 4 GB de GPU.

### Configurar Trainer con adapters y LoRA  
En `training/trainer_config.py`:  
- Añade una confi### Tokenización y DataLoader  
- En `data/preprocess.py`, ajusta la tokenización para truncar a 128 tokens y pad dinámico.  
- Describe cómo habilitarías `gradient_accumulation` en un entorno con 4 GB de GPU.

### Configurar Trainer con adapters y LoRA  
En `training/trainer_config.py`:  
- Añade una configuración de LoRA (`rank=8`, `α=16`) aplicada a las capas de atención.  
- Un Adapter (`bottleneck=64`) en cada capa feed-forward.  
- Define cómo en el loop de entrenamiento se habilitan solo estos parámetros para optimización.

### Implementación de AWP y mixout manual  
En `training/train.py`, dentro de `compute_loss` o hook de backward:  
- Aplica AWP con `ε=0.01` durante la mitad de cada epoch.  
- Integra mixout con `p=0.1` en las capas lineales del classifier.

### Callbacks y métricas  
- Crea un callback sencillo para early stopping tras 3 epochs sin mejora en F1.  
- Ajusta `compute_metrics` para devolver precision, recall y F1, y muestra la matriz de confusión tras el entrenamiento.

### Evaluación final  
- Ejecuta `evaluation/eval.py` sobre el test set y captura el reporte de clasificación.  
- Redacta en 5 líneas cómo afectarían los adapters y LoRA a la latencia de inferencia.

#### Entrega

Cada estudiante presentará su propio repositorio con todos los scripts modificados, los resultados (tablas, gráficas, checkpoints) y un informe en Markdown que documente brevemente la instalación, la ejecución y un análisis de los resultados obtenidos.

#### Puntuaciones

#### Examen 2: Fine-tuning con Transformers y técnicas avanzadas (20 pt)  
- **Preprocesamiento y DataLoader (3 pt)**
  - 2 pt: Tokenización a 128 tokens y padding dinámico.  
  - 1 pt: Uso adecuado de `gradient_accumulation` en entorno limitado.
- **Configuración de LoRA y Adapters (5 pt)**
  - 3 pt: Parámetros de LoRA (rank, α) aplicados correctamente.  
  - 2 pt: Adapters integrados en capas feed-forward.
- **Implementación de AWP y mixout (4 pt)**
  - 2 pt: AWP en el loop de entrenamiento.  
  - 2 pt: Mixout incorporado en capas lineales.
- **Callbacks y métricas (4 pt)**
  - 2 pt: Callback de early stopping funcionando.  
  - 2 pt: Métricas de precision, recall y F1, y matriz de confusión.
- **Análisis de resultados (4 pt)**
  - 2 pt: Tabla comparativa de parámetros entrenables vs. full fine-tuning.  
  - 2 pt: Comentario sobre impacto en latencia y uso de GPU.
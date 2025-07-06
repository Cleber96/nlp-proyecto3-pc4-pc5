# SOLUCIÓN
## EJECUCIÓN DE PROYECTO (replicarlo localmente)
```bash
    python3 -m venv venv 
    source venv/bin/activate
    pip install -r requirements.txt
    python3 download_and_prepare_hf_data.py
    python3 run.py preprocess
    python3 run.py train
    python3 run.py evaluate
```
## PREGUNTA 01: Tokenización y DataLoader  
- En `utils/preprocess.py`, en la función `tokenize_funtion()` cambiamos el máximo de tokens y ṕonemos  un pad dinámico
```python
#ANTES (linea 68)
def tokenize_function(examples, tokenizer, text_column, max_length):
    return tokenizer(examples[text_column], truncation=True, max_length=max_length)
#AHORA
def tokenize_function(examples, tokenizer, text_column): 
    return tokenizer(examples[text_column], truncation=True, max_length=128, padding=False)
```
OBS: El `max_length` que se tiene en `config/setting.py` es 128 tokens por defecto, pero se hace explicito para responder
```python
# (linea 38)
MAX_SEQ_LENGTH = 128
MAX_LENGTH = MAX_SEQ_LENGTH
```
- para habilitar la a`gradient_accumulation` con una GPU de 4 GB, ajustaríamos el parámetro `GRADIENT_ACCUMULATION_STEPS` en el archivo `config/settings.py` para simular un tamaño de batch mayor al original. Se cambia para ir acumulando sus gradientes antes de realizar una única actualización de los pesos del modelo
```python
#ANTES
# (linea 63)
PER_DEVICE_TRAIN_BATCH_SIZE = 16
# (linea 100)
GRADIENT_ACCUMULATION_STEPS = 1
# AHORA
# (linea 63)
PER_DEVICE_TRAIN_BATCH_SIZE = 8 # 4gb gpu
# (linea 100)
GRADIENT_ACCUMULATION_STEPS = 2 # para mantener el original
``` 
## PREGUNTA 02: Configurar Trainer con adapters y LoRA  
En `training/trainer_config.py` añado  
- Añade una configuración de LoRA (`rank=8`, `α=16`) aplicada a las capas de atención.
```python
    #configuración lora
    lora_config = LoraConfig(
        r=8,                       # rank = 8
        lora_alpha=16,             # alpha = 16
        target_modules=[
            "q_lin", "k_lin", "v_lin", "out_lin" # Capas de atención 
        ],
        lora_dropout=0.1,  
        bias="none", 
        task_type=TaskType.SEQ_CLS
    )

    # Aplica LoRA
    model = get_peft_model(model, lora_config)    
    #parámetros entrenablesLoRA
    model.print_trainable_parameters() 
    logger.info("LoRA aplicado al modelo. Se han modificado los parámetros entrenables.")
```
- Un Adapter (`bottleneck=64`) en cada capa feed-forward.  
```python
adapter_config = PfeifferConfig(
        reduction_factor=16,
        non_linearity="relu",
        leave_out=["predictions"],
        bottleneck_dim=64 # cambio el bottleneck
    )
```
- Define cómo en el loop de entrenamiento se habilitan solo estos parámetros para optimización.
La optimización de solo los parámetros de LoRa durante el entrenamiento se gestiona automáticamente al utilizar la librería PEFT de Hugging Face. Al envolver el modelo base con get_peft_model(), esta función congela eficazmente las capas pre-entre nadas existentes, designando exclusivamente las nuevas matrices de bajo rango de LoRA como entrenables. Esto simplifica significativamente el proceso, ya que el optimizador del Trainer de Hugging Face solo verá y actualizará los gradientes asociados a estos parámetros eficientemente ajustados.

## PREGUNTA 03: Implementación de AWP y mixout manual  
En `training/train.py`, dentro de `compute_loss` o hook de backward:  
- Aplica AWP con `ε=0.01 durante la mitad de cada epoch
```python
#Antes
USE_AWP = False
AWP_LR = 1e-4             # Tasa de aprendizaje para la perturbación AWP
AWP_EPS = 1e-6            # Magnitud de la perturbación epsilon
AWP_START_STEP = 500      # Paso a partir del cual aplicar AWP
AWP_EMB_NAME = "distilbert.embeddings.word_embeddings.weight"
#ahora 
USE_AWP = True
AWP_LR = 1e-4   
AWP_EPS = 0.01   
AWP_START_STEP = 0 
``` 
OBS: mi estructura original para AWP está en la clase `customTrainer` y en la `función train_model`, por lo que solo se cambiará `AWP_EPS` en `settings.py` a 0.01 y que `AWP_START_STEP` se calcule dinámicamente para que AWP se active a la mitad de los pasos
```python
#antes
    awp_config = {
        'adv_lr': settings.AWP_LR,
        'adv_eps': settings.AWP_EPS,
        'awp_start_step': settings.AWP_START_STEP,
        'awp_emb_name': settings.AWP_EMB_NAME,
    }
#ahora
    awp_config = {
        'adv_lr': settings.AWP_LR,
        'adv_eps': settings.AWP_EPS,
        'awp_start_step': awp_start_step_calculated, #se aplica
        'awp_emb_name': settings.AWP_EMB_NAME
    }
```
- Integra mixout con `p=0.1` en las capas lineales del classifier.

## PREGUNTA 04: Callbacks y métricas  
- Crea un callback sencillo para early stopping tras 3 epochs sin mejora en F1.  
- Ajusta `compute_metrics` para devolver precision, recall y F1, y muestra la matriz de confusión tras el entrenamiento.

## PREGUNTA 05: Evaluación final  
- Ejecuta `evaluation/eval.py` sobre el test set y captura el reporte de clasificación.  
- Redacta en 5 líneas cómo afectarían los adapters y LoRA a la latencia de inferencia.

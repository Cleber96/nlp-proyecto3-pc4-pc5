# training/callbacks/fingerprint_callback.py

import os
import hashlib
import json
import socket
from datetime import datetime
from transformers import TrainerCallback, TrainerState, TrainingArguments, TrainerControl
import logging

# Configuración básica de logging para este módulo
logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def _get_environment_hash() -> str:
    hostname = socket.gethostname()
    current_time = datetime.now().strftime("%Y%m%d%H%M%S")
    # Para una mayor robustez, se podría incluir información de CPU/GPU, versiones de Python/PyTorch/Transformers
    # Ejemplo: f"{hostname}-{current_time}-{sys.version}-{torch.__version__}-{transformers.__version__}"
    fingerprint_string = f"{hostname}-{current_time}"
    return hashlib.sha256(fingerprint_string.encode('utf-8')).hexdigest()[:10] # Tomar los primeros 10 caracteres

class FingerprintCallback(TrainerCallback):
    
    def __init__(self):
        self.environment_fingerprint = _get_environment_hash()
        logger.info(f"FingerprintCallback inicializado. Huella digital del entorno: {self.environment_fingerprint}")

    def on_save(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        # La ruta del directorio del checkpoint se obtiene de state.global_step o args.output_dir
        # El Trainer ya guarda el checkpoint en args.output_dir/checkpoint-<step>
        checkpoint_dir = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")

        # Si es el "best_model_at_end" y la estrategia es "epoch" o "steps" que ya ha guardado,
        # puede que el directorio sea el principal de output. Hay que ser robustos.
        # Determinamos el directorio de guardado actual
        if state.global_step == state.max_steps: # Final del entrenamiento, suele guardar en output_dir directamente
             current_save_dir = args.output_dir
        elif state.is_world_process_zero: # Solo el proceso 0 guarda el checkpoint
             # Busca el último checkpoint guardado o usa el path de la época/paso actual
             # En general, el Trainer se encarga de crear el dir como checkpoint-step
             if not os.path.exists(checkpoint_dir):
                 # Esto puede ocurrir si on_save se llama por el save_strategy 'epoch'
                 # y no por un save_steps exacto. Buscamos el último directorio creado.
                 all_checkpoints = [d for d in os.listdir(args.output_dir) if d.startswith("checkpoint-")]
                 if all_checkpoints:
                     checkpoint_dir = os.path.join(args.output_dir, max(all_checkpoints, key=lambda d: int(d.split('-')[1])))
                 else: # Si no hay checkpoints, asumimos el directorio base
                     checkpoint_dir = args.output_dir
             current_save_dir = checkpoint_dir
        else: # Otros procesos no guardan ni modifican el estado
            return


        trainer_state_path = os.path.join(current_save_dir, "trainer_state.json")

        if os.path.exists(trainer_state_path):
            try:
                with open(trainer_state_path, 'r') as f:
                    trainer_state = json.load(f)
            except json.JSONDecodeError as e:
                logger.error(f"Error decodificando trainer_state.json en {trainer_state_path}: {e}")
                trainer_state = {} # Inicializar vacío si hay error
        else:
            trainer_state = {} # Archivo no encontrado, inicializar vacío

        # Añadir o actualizar la huella digital en el estado del trainer
        trainer_state['environment_fingerprint'] = self.environment_fingerprint
        trainer_state['fingerprint_timestamp'] = datetime.now().isoformat()

        # Guardar el estado actualizado
        try:
            with open(trainer_state_path, 'w') as f:
                json.dump(trainer_state, f, indent=4)
            logger.debug(f"Huella digital añadida a {trainer_state_path}")
        except Exception as e:
            logger.error(f"Error al escribir huella digital en {trainer_state_path}: {e}")
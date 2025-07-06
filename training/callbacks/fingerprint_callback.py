# training/callbacks/fingerprint_callback.py

import os
import json
import hashlib
import hmac
import datetime
import platform
import sys
import torch
import transformers
import datasets
import pandas
import numpy
import accelerate
import sklearn
from transformers import TrainerCallback, TrainingArguments, TrainerState, TrainerControl
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

class FingerprintCallback(TrainerCallback):
    """
    Un callback que añade una huella digital única (HMAC) al archivo trainer_state.json
    para cada checkpoint guardado.
    """
    def __init__(self, team_id: str = "EquipoCC0C2", secret_key: str = "MiClaveSecreta"):
        self.team_id = team_id
        self.secret_key = secret_key.encode('utf-8') # La clave secreta debe ser bytes

    def _generate_fingerprint_data(self, state: TrainerState) -> Dict[str, Any]:
        """Genera datos relevantes del entorno para la huella digital."""
        fingerprint_data = {
            "timestamp": datetime.datetime.now().isoformat(),
            "team_id": self.team_id,
            "python_version": sys.version,
            "platform": platform.platform(),
            "torch_version": torch.__version__,
            "transformers_version": transformers.__version__,
            "datasets_version": datasets.__version__,
            "pandas_version": pandas.__version__,
            "numpy_version": numpy.__version__,
            "accelerate_version": accelerate.__version__,
            "sklearn_version": sklearn.__version__,
            "current_epoch": state.epoch,
            "global_step": state.global_step,
            "learning_rate": state.learning_rate,
            "best_metric": state.best_metric,
            "best_model_checkpoint": state.best_model_checkpoint,
            # Puedes añadir más información relevante si lo deseas
        }
        # Verificar si CUDA está disponible y añadir información de GPU
        if torch.cuda.is_available():
            fingerprint_data["cuda_available"] = True
            fingerprint_data["cuda_version"] = torch.version.cuda
            fingerprint_data["gpu_name"] = torch.cuda.get_device_name(0)
        else:
            fingerprint_data["cuda_available"] = False

        return fingerprint_data

    def _generate_hmac(self, data: Dict[str, Any]) -> str:
        """Genera un HMAC para los datos proporcionados."""
        # Convertir el diccionario a una cadena JSON consistente
        data_str = json.dumps(data, sort_keys=True)
        return hmac.new(self.secret_key, data_str.encode('utf-8'), hashlib.sha256).hexdigest()

    def on_save(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """
        Se ejecuta cuando el Trainer guarda un checkpoint.
        Añade la huella digital al archivo trainer_state.json.
        """
        output_dir = args.output_dir
        if state.best_model_checkpoint:
            # Si se guarda un best model, el state se guarda en su directorio específico
            checkpoint_dir = state.best_model_checkpoint
        else:
            # Si no es un best model, se guarda en el último checkpoint normal
            checkpoint_dir = os.path.join(output_dir, f"checkpoint-{state.global_step}")

        # Construir la ruta al archivo trainer_state.json
        trainer_state_path = os.path.join(checkpoint_dir, "trainer_state.json")

        try:
            # Cargar el contenido existente del trainer_state.json
            if os.path.exists(trainer_state_path):
                with open(trainer_state_path, 'r', encoding='utf-8') as f:
                    trainer_state_content = json.load(f)
            else:
                trainer_state_content = {}
                logger.warning(f"trainer_state.json no encontrado en {checkpoint_dir}, se creará uno nuevo.")

            # Generar los datos de la huella digital
            fingerprint_data = self._generate_fingerprint_data(state)
            hmac_value = self._generate_hmac(fingerprint_data)

            # Añadir la huella digital y los datos al estado
            trainer_state_content['environment_fingerprint'] = hmac_value
            trainer_state_content['fingerprint_details'] = fingerprint_data

            # Guardar el archivo actualizado
            os.makedirs(os.path.dirname(trainer_state_path), exist_ok=True)
            with open(trainer_state_path, 'w', encoding='utf-8') as f:
                json.dump(trainer_state_content, f, indent=4)
            logger.info(f"Huella digital añadida a {trainer_state_path}")

        except Exception as e:
            logger.error(f"Error al añadir la huella digital al trainer_state.json en {checkpoint_dir}: {e}", exc_info=True)
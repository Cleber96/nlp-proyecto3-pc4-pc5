# models/custom_model.py

import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification, PreTrainedModel, AutoConfig
from transformers.modeling_outputs import SequenceClassifierOutput
from typing import Optional

class CustomSequenceClassificationModel(PreTrainedModel):

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.transformer = AutoModelForSequenceClassification.from_config(config)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> SequenceClassifierOutput:
        # Pasa las entradas al modelo Transformer base (AutoModelForSequenceClassification ya tiene una cabeza)
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            labels=labels, # Pasa las etiquetas directamente para que AutoModel calcule la pérdida
            **kwargs,
        )

        return outputs

def load_custom_model(model_name_or_path: str, num_labels: int) -> CustomSequenceClassificationModel:
    """
    Carga una instancia de CustomSequenceClassificationModel.
    En la práctica, AutoModelForSequenceClassification.from_pretrained()
    ya maneja esto para modelos estándar de HF.
    Este helper sería útil si 'CustomSequenceClassificationModel' realmente
    tuviera una arquitectura modificada compleja.
    """
    config = AutoConfig.from_pretrained(model_name_or_path, num_labels=num_labels)
    model = CustomSequenceClassificationModel(config)
    return model
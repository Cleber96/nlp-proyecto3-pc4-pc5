# training/adversarial/mixout.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)

class MixoutLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, p: float = 0.0):
        super().__init__()
        # Componemos una capa nn.Linear estándar
        self.linear = nn.Linear(in_features, out_features, bias)
        self.p = p # Probabilidad de mezcla
        self.original_weight = None 

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training and self.p > 0 and self.original_weight is not None:
            mixed_weight = (1 - self.p) * self.linear.weight + self.p * self.original_weight
            return F.linear(x, mixed_weight, self.linear.bias)
        else:
            return self.linear(x)

def apply_mixout_to_model(model: nn.Module, original_pretrained_model: nn.Module, p: float = 0.0) -> nn.Module:
    if p == 0:
        logger.info("Mixout probability is 0. No Mixout applied.")
        return model
    target_layer_names = ["pre_classifier", "classifier"] 
    model_modules = {name: module for name, module in model.named_modules()}
    original_model_modules = {name: module for name, module in original_pretrained_model.named_modules()}

    for target_name in target_layer_names:
        current_linear_module = model_modules.get(target_name)
        original_linear_module = original_model_modules.get(target_name)

        # Verificamos que ambas capas existan y sean instancias de nn.Linear.
        if (isinstance(current_linear_module, nn.Linear) and 
            isinstance(original_linear_module, nn.Linear)):
            
            # Creamos la nueva capa MixoutLinear.
            new_mixout_layer = MixoutLinear(
                current_linear_module.in_features,
                current_linear_module.out_features,
                bias=current_linear_module.bias is not None,
                p=p
            )
            
            # Copiamos los pesos y sesgos actuales a la capa lineal encapsulada.
            new_mixout_layer.linear.weight.data.copy_(current_linear_module.weight.data)
            if current_linear_module.bias is not None:
                new_mixout_layer.linear.bias.data.copy_(current_linear_module.bias.data)
        
            new_mixout_layer.original_weight = original_linear_module.weight.data.clone().detach()
            setattr(model, target_name, new_mixout_layer)
            logger.info(f"Capa '{target_name}' transformada a MixoutLinear con p={p}.")
        else:
            logger.warning(f"Advertencia: No se pudo aplicar Mixout a la capa '{target_name}'. Asegúrate que sea una capa lineal y esté presente en ambos modelos.")

    return model
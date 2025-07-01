# training/adversarial/mixout.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class MixoutLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, p: float = 0.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias
        self.p = p # Probabilidad de mezcla

        # Pesos y sesgos de la capa lineal normal
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)

        # Pesos y sesgos pre-entrenados originales, se copiarán más tarde
        self.weight_orig = nn.Parameter(self.weight.clone(), requires_grad=False)
        if bias:
            self.bias_orig = nn.Parameter(self.bias.clone(), requires_grad=False)
        else:
            self.register_parameter('bias_orig', None)

        self.reset_parameters()

    def reset_parameters(self):
        """
        Inicializa los pesos de la capa, siguiendo la inicialización de nn.Linear.
        """
        nn.init.kaiming_uniform_(self.weight, a=5)
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / (fan_in ** 0.5)
            nn.init.uniform_(self.bias, -bound, bound)
            
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.training and self.p > 0:
            # Crea una máscara binomial para decidir qué pesos mezclar
            # Los elementos de la máscara son 0 o 1, representando mantener o mezclar
            mask = torch.empty_like(self.weight).bernoulli_(1 - self.p)
            
            # Aplica la máscara: si mask[i] es 0, se usa el peso original, si es 1, se usa el peso actual
            # Luego se escala por 1/(1-p) para compensar los pesos "mezclados"
            mixed_weight = self.weight * mask + self.weight_orig * (1 - mask)
            
            # El factor de escala (1 / (1 - p)) se aplica como en Dropout,
            # para mantener la expectativa de la salida constante.
            mixed_weight = mixed_weight / (1 - self.p)

            # Realiza la multiplicación de la matriz con el peso mezclado
            return F.linear(input, mixed_weight, self.bias)
        else:
            # En modo de evaluación o si p=0, se comporta como una capa lineal normal
            return F.linear(input, self.weight, self.bias)

def apply_mixout_to_model(model: nn.Module, original_pretrained_model: nn.Module, p: float = 0.0) -> nn.Module:
    if p == 0:
        print("Mixout probability is 0. No Mixout applied.")
        return model

    # Recorre los módulos del modelo
    for name, module in model.named_children():
        # Si el módulo es una capa lineal...
        if isinstance(module, nn.Linear):
            # Crea una nueva capa MixoutLinear con los mismos parámetros
            new_linear = MixoutLinear(
                module.in_features,
                module.out_features,
                bias=module.bias is not None,
                p=p
            )
            # Copia los pesos y sesgos actuales a la nueva capa
            new_linear.weight.data.copy_(module.weight.data)
            if module.bias is not None:
                new_linear.bias.data.copy_(module.bias.data)

            # Encuentra el módulo correspondiente en el modelo pre-entrenado original
            # y copia sus pesos al 'weight_orig' de MixoutLinear
            orig_module = original_pretrained_model
            for sub_name in name.split('.'):
                orig_module = getattr(orig_module, sub_name)

            if isinstance(orig_module, nn.Linear):
                new_linear.weight_orig.data.copy_(orig_module.weight.data)
                if module.bias is not None:
                    new_linear.bias_orig.data.copy_(orig_module.bias.data)
                
                # Reemplaza la capa lineal original con la nueva capa MixoutLinear
                # Esto es crucial para modificar el modelo en su lugar
                setattr(model, name, new_linear)
                # print(f"  Reemplazada capa lineal: {name}")
            else:
                print(f"  Advertencia: Módulo '{name}' en el modelo original no es nn.Linear. No se aplicó Mixout a esta capa.")

        # Si el módulo tiene submódulos, llama recursivamente a la función
        elif len(list(module.children())) > 0:
            setattr(model, name, apply_mixout_to_model(module, getattr(original_pretrained_model, name), p))

    return model

def check_mixout_applied(model: nn.Module) -> bool:
    for module in model.modules():
        if isinstance(module, MixoutLinear):
            return True
    return False
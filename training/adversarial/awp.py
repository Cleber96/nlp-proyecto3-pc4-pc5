# training/adversarial/awp.py

import torch
import torch.nn as nn
from typing import Dict, Any

class AWP:
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        adv_param: str = 'weight',  # Parámetro a perturbar: 'weight' o 'bias'
        adv_lr: float = 0.001,      # Tasa de aprendizaje para la perturbación
        adv_eps: float = 0.001      # Epsilon (magnitud) máxima de la perturbación
    ):
        self.model = model
        self.optimizer = optimizer
        self.adv_param = adv_param
        self.adv_lr = adv_lr
        self.adv_eps = adv_eps
        self.backup = {}  # Para guardar los pesos originales
        self.backup_eps = {}  # Para guardar las perturbaciones aplicadas

    def _save(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None and self.adv_param in name:
                # Almacena el valor original del parámetro
                self.backup[name] = param.data.clone()

    def attack_step(self):
        e = 1e-6  # Pequeño valor para evitar división por cero
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None and self.adv_param in name:
                # Normaliza el gradiente y lo escala por la tasa de aprendizaje adversaria
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    r_adv = param.grad / (norm + e) * self.adv_lr
                    # Clampa la perturbación al rango [-adv_eps, adv_eps]
                    param.data.add_(r_adv, alpha=1) # param.data = param.data + r_adv
                    param.data = torch.min(param.data, self.backup[name] + self.adv_eps)
                    param.data = torch.max(param.data, self.backup[name] - self.adv_eps)
                # Guarda la perturbación aplicada para poder restaurar
                self.backup_eps[name] = param.data.clone()

    def restore(self):
        for name, param in self.model.named_parameters():
            if name in self.backup:
                param.data = self.backup[name] # Restaura al valor original
        self.backup = {} # Limpia el backup
        self.backup_eps = {} # Limpia el backup de perturbaciones (opcional, pero buena práctica)

    def _set_original_grad(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None and self.adv_param in name:
                param.grad.data = self.backup_eps[name]
# tests/test_mixout.py

import pytest
import torch
import torch.nn as nn
from training.adversarial.mixout import MixoutLinear

class TestMixoutLinearInitialization:

    def test_mixout_linear_replaces_nn_linear(self):
        in_features = 10
        out_features = 5
        mixout_p = 0.5
        layer = MixoutLinear(in_features, out_features, p=mixout_p)

        assert isinstance(layer, MixoutLinear)
        assert layer.in_features == in_features
        assert layer.out_features == out_features
        assert layer.p == mixout_p
        assert hasattr(layer, '_initial_weights')
        # _initial_weights deben tener la misma forma que weight
        assert layer._initial_weights.shape == layer.weight.shape
        # Los sesgos también deben ser manejados si existen
        assert hasattr(layer, 'bias') or layer.bias is None
        if layer.bias is not None:
            assert hasattr(layer, '_initial_bias')
            assert layer._initial_bias.shape == layer.bias.shape

    def test_mixout_linear_initial_weights_stored_correctly(self):
        in_features = 8
        out_features = 4
        layer = MixoutLinear(in_features, out_features, p=0.1)

        # Verificar que _initial_weights es una copia de weight.data
        # y no la misma referencia de memoria.
        assert torch.equal(layer._initial_weights, layer.weight.data)
        assert layer._initial_weights is not layer.weight.data

        if layer.bias is not None:
            assert torch.equal(layer._initial_bias, layer.bias.data)
            assert layer._initial_bias is not layer.bias.data

    def test_mixout_linear_p_value_validation(self):
        in_features, out_features = 5, 5
        with pytest.raises(ValueError, match="Mixout probability has to be between 0 and 1"):
            MixoutLinear(in_features, out_features, p=1.1)
        with pytest.raises(ValueError, match="Mixout probability has to be between 0 and 1"):
            MixoutLinear(in_features, out_features, p=-0.1)

class TestMixoutLinearForwardTraining:
    @pytest.fixture
    def setup_mixout_layer(self):
        in_features = 2
        out_features = 2
        # Pesos y sesgos iniciales y actuales para facilitar la verificación
        layer = MixoutLinear(in_features, out_features, p=0.5, bias=True)
        
        # Asignar valores conocidos a los pesos y sesgos
        layer.weight.data = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)
        layer.bias.data = torch.tensor([0.5, 1.5], dtype=torch.float32)
        
        # Asegurarse de que _initial_weights es una copia
        layer._initial_weights = layer.weight.data.clone()
        layer._initial_bias = layer.bias.data.clone() # Clona el sesgo también

        # Cambiar los pesos y sesgos actuales para que sean diferentes de los iniciales
        layer.weight.data = torch.tensor([[1.1, 2.1], [3.1, 4.1]], dtype=torch.float32)
        layer.bias.data = torch.tensor([0.6, 1.6], dtype=torch.float32)

        layer.train() # Asegurarse de que está en modo entrenamiento
        
        dummy_input = torch.tensor([[1.0, 1.0]], dtype=torch.float32)
        return layer, dummy_input

    def test_forward_pass_p_is_zero(self):
        in_features, out_features = 2, 2
        layer = MixoutLinear(in_features, out_features, p=0.0, bias=True)
        layer.train() # Asegurarse de que está en modo entrenamiento

        # Asignar valores fijos para predecir la salida
        layer.weight.data = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)
        layer.bias.data = torch.tensor([0.5, 1.5], dtype=torch.float32)
        # Asegurarse de que initial_weights también es una copia para la comparación
        layer._initial_weights = layer.weight.data.clone()
        layer._initial_bias = layer.bias.data.clone()

        dummy_input = torch.tensor([[1.0, 1.0]], dtype=torch.float32)

        # Calcular salida esperada sin Mixout (como un nn.Linear normal)
        expected_output = torch.matmul(dummy_input, layer.weight.T) + layer.bias
        
        actual_output = layer(dummy_input)
        assert torch.allclose(actual_output, expected_output)

    def test_forward_pass_p_is_one(self):
        in_features, out_features = 2, 2
        layer = MixoutLinear(in_features, out_features, p=1.0, bias=True)
        layer.train()

        # Asignar valores conocidos (diferentes)
        layer.weight.data = torch.tensor([[10.0, 20.0], [30.0, 40.0]], dtype=torch.float32)
        layer.bias.data = torch.tensor([10.5, 11.5], dtype=torch.float32)
        
        # Pesos iniciales que deben usarse
        layer._initial_weights = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)
        layer._initial_bias = torch.tensor([0.5, 1.5], dtype=torch.float32)

        dummy_input = torch.tensor([[1.0, 1.0]], dtype=torch.float32)

        # Calcular salida esperada usando SOLO los pesos y sesgos iniciales
        expected_output = torch.matmul(dummy_input, layer._initial_weights.T) + layer._initial_bias
        actual_output = layer(dummy_input)

        assert torch.allclose(actual_output, expected_output)

    def test_forward_pass_p_is_between_zero_and_one(self, setup_mixout_layer):
        layer, dummy_input = setup_mixout_layer
        
        torch.manual_seed(42)

        # Recordar los pesos actuales y iniciales
        current_weight = layer.weight.data.clone()
        initial_weight = layer._initial_weights.clone()
        current_bias = layer.bias.data.clone()
        initial_bias = layer._initial_bias.clone()

        # Calcular la máscara que Mixout usará
        mask_weight = torch.bernoulli(torch.full(current_weight.shape, 1 - layer.p))
        mask_bias = torch.bernoulli(torch.full(current_bias.shape, 1 - layer.p)) if layer.bias is not None else None

        # Pesos esperados después de la mezcla
        mixed_weight = current_weight * mask_weight + initial_weight * (1 - mask_weight)
        mixed_bias = current_bias * mask_bias + initial_bias * (1 - mask_bias) if layer.bias is not None else None

        # Calcular la salida esperada con los pesos y sesgos mezclados
        if layer.bias is not None:
            expected_output = torch.matmul(dummy_input, mixed_weight.T) + mixed_bias
        else:
            expected_output = torch.matmul(dummy_input, mixed_weight.T)

        # Realizar la pasada hacia adelante real
        actual_output = layer(dummy_input)
        
        assert torch.allclose(actual_output, expected_output)

class TestMixoutLinearForwardEvaluation:
    def test_forward_pass_eval_mode(self):
        in_features, out_features = 2, 2
        layer = MixoutLinear(in_features, out_features, p=0.5, bias=True)
        
        # Asignar valores conocidos (diferentes)
        layer.weight.data = torch.tensor([[10.0, 20.0], [30.0, 40.0]], dtype=torch.float32)
        layer.bias.data = torch.tensor([10.5, 11.5], dtype=torch.float32)
        
        # Pesos iniciales (deberían ser ignorados en modo eval)
        layer._initial_weights = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)
        layer._initial_bias = torch.tensor([0.5, 1.5], dtype=torch.float32)

        layer.eval() # Poner la capa en modo de evaluación

        dummy_input = torch.tensor([[1.0, 1.0]], dtype=torch.float32)

        # Calcular salida esperada usando SOLO los pesos y sesgos actuales
        expected_output = torch.matmul(dummy_input, layer.weight.T) + layer.bias
        actual_output = layer(dummy_input)

        assert torch.allclose(actual_output, expected_output)
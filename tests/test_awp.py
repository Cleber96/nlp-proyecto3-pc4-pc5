# tests/test_awp.py

import pytest
import torch
import torch.nn as nn
from training.adversarial.awp import AWP

class DummyModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_labels):
        super().__init__()
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear = nn.Linear(embedding_dim, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, num_labels)
        
        # Inicializar pesos de forma predecible para facilitar las pruebas
        nn.init.constant_(self.word_embeddings.weight, 0.5)
        nn.init.constant_(self.linear.weight, 0.1)
        nn.init.constant_(self.classifier.weight, 0.2)
        nn.init.constant_(self.linear.bias, 0.0)
        nn.init.constant_(self.classifier.bias, 0.0)

    def forward(self, input_ids):
        embedded = self.word_embeddings(input_ids)
        x = self.linear(embedded.mean(dim=1)) # Simular alguna agregación
        logits = self.classifier(x)
        return logits

@pytest.fixture
def awp_setup():
    vocab_size = 10
    embedding_dim = 5
    hidden_dim = 3
    num_labels = 2
    model = DummyModel(vocab_size, embedding_dim, hidden_dim, num_labels)
    
    awp_lr = 0.01
    awp_eps = 1e-3
    start_step = 0 # Iniciar AWP desde el primer paso
    emb_name = 'word_embeddings.weight' # Nombre del parámetro de embedding a perturbar

    awp_instance = AWP(model, awp_lr, awp_eps, start_step, emb_name)
    
    return model, awp_instance, awp_lr, awp_eps, emb_name
class TestAWPInitialization:

    def test_awp_initialization(self, awp_setup):
        model, awp_instance, awp_lr, awp_eps, emb_name = awp_setup

        assert awp_instance.model == model
        assert awp_instance.awp_lr == awp_lr
        assert awp_instance.awp_eps == awp_eps
        assert awp_instance.start_step == 0
        assert awp_instance.emb_name == emb_name
        assert awp_instance.backup is None # No hay backup al inicio
        assert awp_instance.backup_eps is None # No hay backup de epsilon al inicio

        # Asegurarse de que el parámetro a perturbar se encuentra
        found_param = False
        for name, _ in model.named_parameters():
            if name == emb_name:
                found_param = True
                break
        assert found_param, f"Parámetro de embedding '{emb_name}' no encontrado en el modelo dummy."

class TestAWPAttackStep:
    def test_attack_step_applies_perturbation(self, awp_setup):
        model, awp_instance, awp_lr, awp_eps, emb_name = awp_setup

        original_emb_weight = model.word_embeddings.weight.data.clone()
        
        # Simular un gradiente para el embedding
        dummy_grad = torch.randn_like(original_emb_weight)
        model.word_embeddings.weight.grad = dummy_grad

        # Ejecutar el paso de ataque
        awp_instance.attack_step()

        # Verificar que se ha creado un backup de los pesos originales
        assert awp_instance.backup is not None
        assert emb_name in awp_instance.backup
        assert torch.equal(awp_instance.backup[emb_name], original_emb_weight)

        # Verificar que los pesos del embedding han sido modificados
        perturbed_emb_weight = model.word_embeddings.weight.data
        assert not torch.equal(perturbed_emb_weight, original_emb_weight)

        norm_grad = dummy_grad / (torch.norm(dummy_grad) + 1e-10) # Añadir epsilon para estabilidad
        expected_perturbation = awp_eps * norm_grad
        
        # Verifiquemos que `adv_param` se calcula correctamente
        norm_grad_calc = dummy_grad / (torch.norm(dummy_grad) + 1e-10)
        expected_adv_param = awp_eps * norm_grad_calc
        assert torch.allclose(awp_instance.adv_param[emb_name], expected_adv_param)

        # Verificar que la perturbación aplicada es correcta
        # model.word_embeddings.weight.data debería ser original_emb_weight + awp_lr * expected_adv_param
        expected_perturbed_weight = original_emb_weight + awp_lr * expected_adv_param
        assert torch.allclose(perturbed_emb_weight, expected_perturbed_weight)


    def test_attack_step_no_gradient(self, awp_setup):
        model, awp_instance, _, _, emb_name = awp_setup

        original_emb_weight = model.word_embeddings.weight.data.clone()
        # No asignar gradiente
        model.word_embeddings.weight.grad = None 

        awp_instance.attack_step()

        # Los pesos no deben cambiar si no hay gradiente
        assert torch.equal(model.word_embeddings.weight.data, original_emb_weight)
        assert awp_instance.backup is None # No debería haber backup si no se aplicó perturbación

class TestAWPRestore:
    def test_restore_reverts_weights(self, awp_setup):
        model, awp_instance, awp_lr, awp_eps, emb_name = awp_setup

        original_emb_weight = model.word_embeddings.weight.data.clone()
        dummy_grad = torch.randn_like(original_emb_weight)
        model.word_embeddings.weight.grad = dummy_grad

        # Aplicar perturbación
        awp_instance.attack_step()
        perturbed_weight = model.word_embeddings.weight.data.clone()
        assert not torch.equal(perturbed_weight, original_emb_weight)

        # Restaurar
        awp_instance.restore()

        # Verificar que los pesos han vuelto a su estado original
        assert torch.equal(model.word_embeddings.weight.data, original_emb_weight)
        assert awp_instance.backup is None # El backup debe borrarse después de restaurar
        assert awp_instance.backup_eps is None # Y también el backup_eps

    def test_restore_no_backup(self, awp_setup):
        model, awp_instance, _, _, _ = awp_setup
        
        # Asegurarse de que no hay backup
        awp_instance.backup = None
        awp_instance.backup_eps = None

        # Llamar a restore; no debería levantar una excepción
        try:
            awp_instance.restore()
        except Exception as e:
            pytest.fail(f"restore() levantó una excepción inesperada: {e}")
# tests/test_callbacks.py

import pytest
import os
import json
from unittest.mock import MagicMock, patch

from training.callbacks.custom_callbacks import CustomEarlyStoppingCallback, CustomLoggingCallback
class TestCustomEarlyStoppingCallback:
    @pytest.fixture
    def mock_trainer_components(self):
        args = MagicMock()
        state = MagicMock()
        control = MagicMock()
        # Valores iniciales para simular el estado del entrenamiento
        state.epoch = 0.0
        state.global_step = 0
        state.best_metric = None
        state.best_model_checkpoint = None
        control.should_training_stop = False # Inicialmente no debería parar
        return args, state, control

    def test_early_stopping_stops_training_on_patience(self, mock_trainer_components):
        args, state, control = mock_trainer_components
        # Configurar EarlyStoppingCallback con paciencia=2
        callback = CustomEarlyStoppingCallback(early_stopping_patience=2)
        state.best_metric = 0.90
        callback.on_evaluate(args, state, control, metrics={"eval_accuracy": 0.90})
        assert not control.should_training_stop

        # Época 1: No mejora (paciencia = 1)
        state.epoch = 1.0
        callback.on_evaluate(args, state, control, metrics={"eval_accuracy": 0.89})
        assert not control.should_training_stop

        # Época 2: No mejora (paciencia = 2, debería parar ahora)
        state.epoch = 2.0
        callback.on_evaluate(args, state, control, metrics={"eval_accuracy": 0.88})
        # Verifica que el control de parada se activa
        assert control.should_training_stop
        assert state.best_metric == 0.90 # La mejor métrica debe ser la original

    def test_early_stopping_does_not_stop_on_improvement(self, mock_trainer_components):
        args, state, control = mock_trainer_components
        callback = CustomEarlyStoppingCallback(early_stopping_patience=2)

        # Época 0: Mejor métrica inicial
        state.best_metric = 0.85
        callback.on_evaluate(args, state, control, metrics={"eval_accuracy": 0.85})
        assert not control.should_training_stop

        # Época 1: Mejora
        state.epoch = 1.0
        callback.on_evaluate(args, state, control, metrics={"eval_accuracy": 0.90})
        assert not control.should_training_stop
        assert state.best_metric == 0.90 # Mejor métrica actualizada

        # Época 2: No mejora, pero la paciencia se reinicia con la última mejora
        state.epoch = 2.0
        callback.on_evaluate(args, state, control, metrics={"eval_accuracy": 0.89})
        assert not control.should_training_stop
        assert state.best_metric == 0.90 # Mejor métrica sigue siendo 0.90

    def test_early_stopping_with_threshold(self, mock_trainer_components):
        args, state, control = mock_trainer_components
        # Umbral de 0.01: mejora debe ser > 0.01
        callback = CustomEarlyStoppingCallback(early_stopping_patience=1, early_stopping_threshold=0.01)

        state.best_metric = 0.90
        callback.on_evaluate(args, state, control, metrics={"eval_accuracy": 0.90})
        assert not control.should_training_stop

        # Mejora de 0.005 (0.905 - 0.90), que no supera el umbral de 0.01 -> debería detenerse
        state.epoch = 1.0
        callback.on_evaluate(args, state, control, metrics={"eval_accuracy": 0.905})
        assert control.should_training_stop # Se detiene porque 0.005 < 0.01

        # Resetear y probar con mejora suficiente
        args, state, control = mock_trainer_components # Resetear mocks
        callback = CustomEarlyStoppingCallback(early_stopping_patience=1, early_stopping_threshold=0.01)
        state.best_metric = 0.90
        callback.on_evaluate(args, state, control, metrics={"eval_accuracy": 0.90})
        
        # Mejora de 0.02 (0.92 - 0.90), que sí supera el umbral de 0.01 -> NO debería detenerse
        state.epoch = 1.0
        control.should_training_stop = False # Resetear para esta prueba específica
        callback.on_evaluate(args, state, control, metrics={"eval_accuracy": 0.92})
        assert not control.should_training_stop
        assert state.best_metric == 0.92 # Mejor métrica actualizada


class TestCustomLoggingCallback:

    @pytest.fixture
    def mock_trainer_components_for_logging(self):
        args = MagicMock()
        state = MagicMock()
        control = MagicMock()
        state.epoch = 0.0
        state.global_step = 0
        return args, state, control

    @patch('builtins.open', new_callable=MagicMock)
    @patch('os.makedirs', new_callable=MagicMock)
    def test_logging_callback_on_log(self, mock_makedirs, mock_open, mock_trainer_components_for_logging):

        args, state, control = mock_trainer_components_for_logging
        log_path = "mock/path/run_history.json"
        callback = CustomLoggingCallback(log_path=log_path)

        # Simular un log de entrenamiento
        state.epoch = 0.5
        state.global_step = 100
        logs_train = {"loss": 0.5, "learning_rate": 1e-5}
        callback.on_log(args, state, control, logs_train)

        # Simular un log de evaluación
        state.epoch = 1.0
        state.global_step = 200
        logs_eval = {"eval_loss": 0.4, "eval_accuracy": 0.9}
        callback.on_log(args, state, control, logs_eval)
        
        # Verificar que el historial de ejecución se ha llenado
        assert len(callback.run_history) == 2
        assert callback.run_history[0]["epoch"] == 0.5
        assert callback.run_history[0]["global_step"] == 100
        assert callback.run_history[0]["loss"] == 0.5
        assert callback.run_history[1]["epoch"] == 1.0
        assert callback.run_history[1]["eval_accuracy"] == 0.9

        # Asegurarse de que `open` y `makedirs` no fueron llamadas todavía
        mock_open.assert_not_called()
        mock_makedirs.assert_not_called()

    @patch('builtins.open', new_callable=MagicMock)
    @patch('os.makedirs', new_callable=MagicMock)
    @patch('json.dump', new_callable=MagicMock)
    def test_logging_callback_on_train_end(self, mock_json_dump, mock_makedirs, mock_open, mock_trainer_components_for_logging):
        args, state, control = mock_trainer_components_for_logging
        log_path = "mock/path/run_history.json"
        callback = CustomLoggingCallback(log_path=log_path)

        # Simular algunos logs
        callback.on_log(args, state, control, {"loss": 0.6})
        state.epoch = 1.0
        state.global_step = 50
        callback.on_log(args, state, control, {"eval_accuracy": 0.85})

        # Llamar a on_train_end
        callback.on_train_end(args, state, control)

        # Verificar que se intentó crear el directorio
        mock_makedirs.assert_called_once_with(os.path.dirname(log_path), exist_ok=True)
        
        # Verificar que `open` fue llamado con la ruta correcta y en modo escritura
        mock_open.assert_called_once_with(log_path, 'w')
        
        # Verificar que `json.dump` fue llamado con el historial correcto
        # Aquí, mock_open().return_value es el mock del manejador de archivo
        mock_json_dump.assert_called_once_with(callback.run_history, mock_open.return_value, indent=4)
        
        # Verificar que el historial de ejecución no está vacío
        assert len(callback.run_history) > 0

    @patch('builtins.open', side_effect=IOError("Error de escritura"))
    @patch('os.makedirs', new_callable=MagicMock)
    @patch('json.dump', new_callable=MagicMock)
    def test_logging_callback_on_train_end_error_handling(self, mock_json_dump, mock_makedirs, mock_open, mock_trainer_components_for_logging, caplog):
        """
        Verifica que el callback de logging maneja errores al intentar guardar el archivo.
        """
        args, state, control = mock_trainer_components_for_logging
        log_path = "mock/path/run_history.json"
        callback = CustomLoggingCallback(log_path=log_path)

        # Aseguramos que el logger esté en INFO para capturar el error
        with caplog.at_level(logging.ERROR):
            callback.on_train_end(args, state, control)
            
            # Verifica que se registró un mensaje de error
            assert "Error al guardar el historial de ejecución" in caplog.text
            assert "Error de escritura" in caplog.text
            
            # Verifica que json.dump no se llamó porque open falló
            mock_json_dump.assert_not_called()
# scripts/evaluate_script.sh

MODEL_PATH=""
DATASET_SPLIT="validation" # Valor por defecto

# Parsear argumentos
while [[ "$#" -gt 0 ]]; do
    case "$1" in
        --model_path)
            MODEL_PATH="$2"
            shift
            ;;
        --dataset_split)
            DATASET_SPLIT="$2"
            shift
            ;;
        -h|--help)
            echo "Uso: ./scripts/evaluate_script.sh [--model_path <PATH>] [--dataset_split <SPLIT>]"
            echo ""
            echo "Opciones:"
            echo "  --model_path <PATH>      Ruta al directorio del modelo entrenado y tokenizer."
            echo "                           (ej. logs/checkpoints/final_model o logs/checkpoints/checkpoint-XXXX)"
            echo "                           Si no se especifica, usa el modelo final por defecto."
            echo "  --dataset_split <SPLIT>  El split del dataset a usar para la evaluación ('validation' o 'test')."
            echo "                           Por defecto: 'validation'."
            echo "  -h, --help               Muestra esta ayuda."
            exit 0
            ;;
        *)
            echo "Error: Argumento desconocido '$1'"
            exit 1
            ;;
    esac
    shift
done

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT=$(dirname "$SCRIPT_DIR")
cd "$PROJECT_ROOT" || { echo "Error: No se pudo navegar a $PROJECT_ROOT"; exit 1; }

echo "Cambiando directorio de trabajo a: $(pwd)"

if [ -d ".venv" ]; then
    echo "Activando entorno virtual..."
    source .venv/bin/activate
else
    echo "No se encontró un entorno virtual '.venv'. Continuando sin activación."
fi

echo "Iniciando la evaluación del modelo..."

PYTHON_COMMAND="python evaluation/eval.py --dataset_split $DATASET_SPLIT"

if [ -n "$MODEL_PATH" ]; then
    PYTHON_COMMAND="$PYTHON_COMMAND --model_path $MODEL_PATH"
fi

echo "Ejecutando: $PYTHON_COMMAND"
eval $PYTHON_COMMAND


if [ -d ".venv" ]; then
    echo "Desactivando entorno virtual."
    deactivate
fi

echo "Script de evaluación finalizado."
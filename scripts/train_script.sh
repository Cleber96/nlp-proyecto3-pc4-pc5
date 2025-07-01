# scripts/train_script.sh

RESUME_FROM_CHECKPOINT=""

# Parsear argumentos
while [[ "$#" -gt 0 ]]; do
    case "$1" in
        --resume_from_checkpoint)
            RESUME_FROM_CHECKPOINT="$2"
            shift
            ;;
        -h|--help)
            echo "Uso: ./scripts/train_script.sh [--resume_from_checkpoint <PATH>]"
            echo ""
            echo "Opciones:"
            echo "  --resume_from_checkpoint <PATH>  Ruta a un checkpoint para reanudar el entrenamiento."
            echo "                                   Ej: logs/checkpoints/checkpoint-1234"
            echo "  -h, --help                       Muestra esta ayuda."
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

echo "Iniciando el entrenamiento del modelo..."

PYTHON_COMMAND="python training/train.py"

if [ -n "$RESUME_FROM_CHECKPOINT" ]; then
    PYTHON_COMMAND="$PYTHON_COMMAND --resume_from_checkpoint $RESUME_FROM_CHECKPOINT"
    echo "Argumento de reanudación: $RESUME_FROM_CHECKPOINT"
fi

echo "Ejecutando: $PYTHON_COMMAND"
eval $PYTHON_COMMAND

if [ -d ".venv" ]; then
    echo "Desactivando entorno virtual."
    deactivate
fi

echo "Script de entrenamiento finalizado."
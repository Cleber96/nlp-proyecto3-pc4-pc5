# tests/check_originality.py

import ast
import os
import sys

FILES_TO_CHECK = [
    os.path.join("training", "adversarial", "awp.py"),
    os.path.join("training", "adversarial", "mixout.py"),
]

FORBIDDEN_MIXOUT_IMPORTS = ["torch.nn.Dropout", "nn.Dropout", "Dropout"]
FORBIDDEN_MIXOUT_CALLS_IN_FORWARD = ["torch.nn.functional.dropout"] # Podría ser un call a F.dropout
FORBIDDEN_AWP_IMPORTS = [] # No hay implementaciones de AWP comunes en librerías que queramos evitar aquí.


def check_file_originality(filepath, forbidden_imports, forbidden_calls_in_forward=None):

    errors = []
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            tree = ast.parse(f.read(), filename=filepath)

        # Recorrer el AST para buscar patrones prohibidos
        for node in ast.walk(tree):
            # Check 1: Imports prohibidos
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                for alias in node.names:
                    imported_name = alias.name
                    if isinstance(node, ast.ImportFrom) and node.module:
                        imported_name = f"{node.module}.{imported_name}"
                    
                    for forbidden in forbidden_imports:
                        if forbidden in imported_name:
                            errors.append(
                                f"Error en {filepath}, línea {node.lineno}: "
                                f"Uso prohibido de '{forbidden}' en importación: '{imported_name}'."
                            )
            
            # Check 2: Llamadas prohibidas dentro de métodos 'forward' (si aplica)
            if forbidden_calls_in_forward and isinstance(node, ast.ClassDef):
                for method_node in node.body:
                    if isinstance(method_node, ast.FunctionDef) and method_node.name == 'forward':
                        for sub_node in ast.walk(method_node):
                            if isinstance(sub_node, ast.Call):
                                # Intenta obtener el nombre completo de la función llamada
                                func_name = ""
                                if isinstance(sub_node.func, ast.Name):
                                    func_name = sub_node.func.id
                                elif isinstance(sub_node.func, ast.Attribute):
                                    # Esto maneja casos como 'torch.nn.functional.dropout'
                                    # Recursivamente construye el nombre completo
                                    current_attr = sub_node.func
                                    parts = []
                                    while isinstance(current_attr, ast.Attribute):
                                        parts.append(current_attr.attr)
                                        current_attr = current_attr.value
                                    if isinstance(current_attr, ast.Name):
                                        parts.append(current_attr.id)
                                    func_name = ".".join(reversed(parts))
                                
                                for forbidden_call in forbidden_calls_in_forward:
                                    if forbidden_call in func_name:
                                        errors.append(
                                            f"Error en {filepath}, línea {sub_node.lineno}: "
                                            f"Uso prohibido de '{forbidden_call}' dentro del método 'forward'."
                                        )

    except FileNotFoundError:
        errors.append(f"Error: Archivo no encontrado en la ruta: {filepath}")
    except Exception as e:
        errors.append(f"Error al parsear el archivo {filepath}: {e}")
    
    return errors

if __name__ == "__main__":
    all_errors = []
    
    # Asegurarse de que el script se ejecuta desde la raíz del proyecto
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    os.chdir(project_root)

    print(f"Iniciando verificación de originalidad desde: {os.getcwd()}")

    # Verificar mixout.py
    mixout_filepath = FILES_TO_CHECK[1] # 'training/adversarial/mixout.py'
    print(f"\nVerificando {mixout_filepath}...")
    mixout_errors = check_file_originality(
        mixout_filepath,
        forbidden_imports=FORBIDDEN_MIXOUT_IMPORTS,
        forbidden_calls_in_forward=FORBIDDEN_MIXOUT_CALLS_IN_FORWARD
    )
    if mixout_errors:
        print("Errores encontrados en Mixout:")
        for error in mixout_errors:
            print(f"  - {error}")
        all_errors.extend(mixout_errors)
    else:
        print("  - OK: No se encontraron patrones prohibidos en Mixout.")

    # Verificar awp.py
    awp_filepath = FILES_TO_CHECK[0] # 'training/adversarial/awp.py'
    print(f"\nVerificando {awp_filepath}...")
    awp_errors = check_file_originality(
        awp_filepath,
        forbidden_imports=FORBIDDEN_AWP_IMPORTS # Actualmente vacío, pero extensible
    )
    if awp_errors:
        print("Errores encontrados en AWP:")
        for error in awp_errors:
            print(f"  - {error}")
        all_errors.extend(awp_errors)
    else:
        print("  - OK: No se encontraron patrones prohibidos en AWP.")

    if all_errors:
        print("\n--- Verificación de originalidad: FALLIDA ---")
        print(f"Se encontraron {len(all_errors)} problemas que indican falta de originalidad.")
        sys.exit(1) # Salir con código de error
    else:
        print("\n--- Verificación de originalidad: EXITOSA ---")
        print("Todos los archivos cumplen con los criterios de originalidad esperados.")
        sys.exit(0) # Salir con éxito
import pandas as pd
import os

def compare_matrices(small_matrix, large_matrix):
    """
    Compara la matriz pequeña con la matriz grande y verifica si los elementos de la pequeña coinciden en la misma posición.
    """
    rows_small, cols_small = small_matrix.shape
    
    # Verificar que la matriz pequeña quepa dentro de la grande
    if rows_small > large_matrix.shape[0] or cols_small > large_matrix.shape[1]:
        return "Error: La matriz pequeña es más grande que la matriz grande en alguna dimensión."
    
    # Iterar sobre la matriz pequeña y comparar con la matriz grande
    for i in range(rows_small):
        for j in range(cols_small):
            if small_matrix[i, j] != large_matrix[i, j]:
                return f"No coinciden en la posición ({i}, {j})"
    
    return "Coinciden"

def compare_all_matrices(small_folder, large_folder):
    """
    Compara todas las matrices en dos carpetas. Se asume que los nombres de los archivos coinciden.
    """
    small_files = sorted(os.listdir(small_folder))
    large_files = sorted(os.listdir(large_folder))
    
    if len(small_files) != len(large_files):
        return "Error: El número de archivos en ambas carpetas no coincide."
    
    # Comparar cada par de archivos
    results = {}
    for small_file, large_file in zip(small_files, large_files):
        # Leer las matrices
        small_matrix = pd.read_csv(os.path.join(small_folder, small_file), header=None).values
        large_matrix = pd.read_csv(os.path.join(large_folder, large_file), header=None).values
        
        # Comparar las matrices
        result = compare_matrices(small_matrix, large_matrix)
        results[small_file] = result
    
    return results

# Usar la función para comparar todas las matrices (sustituye las carpetas con tus rutas)
small_folder = 'C:/TEG/denseMatrix'
large_folder = 'C:/TEG/sparseMatrix'
comparison_results = compare_all_matrices(small_folder, large_folder)

# Imprimir los resultados
for file, result in comparison_results.items():
    print(f"Comparación para {file}: {result}")

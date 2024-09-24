import numpy as np
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

def sum_small_matrices(small_folder):
    """
    Suma todas las matrices pequeñas dentro de una matriz de tamaño 32x32.
    """
    small_files = sorted(os.listdir(small_folder))
    
    # Inicializar una matriz vacía de tamaño 32x32
    matrix_sum = np.zeros((32, 32))
    
    for small_file in small_files:
        # Leer cada archivo CSV y convertirlo en una matriz (array de numpy)
        small_matrix = pd.read_csv(os.path.join(small_folder, small_file), header=None).values
        
        # Obtener las dimensiones de la matriz pequeña
        rows_small, cols_small = small_matrix.shape
        
        # Sumar la matriz pequeña dentro de la matriz de 32x32
        matrix_sum[:rows_small, :cols_small] += small_matrix
    
    return matrix_sum

def sum_large_matrices(large_folder):
    """
    Suma todas las matrices grandes contenidas en los archivos CSV en la carpeta especificada.
    """
    large_files = sorted(os.listdir(large_folder))
    
    # Inicializar la suma con None (a la espera de cargar la primera matriz)
    matrix_sum = None
    
    for large_file in large_files:
        # Leer cada archivo CSV y convertirlo en una matriz (array de numpy)
        large_matrix = pd.read_csv(os.path.join(large_folder, large_file), header=None).values
        
        # Si es la primera matriz, inicializamos la suma con ella
        if matrix_sum is None:
            matrix_sum = large_matrix
        else:
            # Acumular la suma de las matrices
            matrix_sum = np.add(matrix_sum, large_matrix)
    
    return matrix_sum

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

total_sum_matrix = sum_large_matrices(large_folder)

# Mostrar la matriz suma total
print("La suma de todas las matrices grandes es:")
print(total_sum_matrix)

matrix_df = pd.DataFrame(total_sum_matrix)

# Ruta del archivo CSV
file_path = 'genGS.csv'

# Verificar si el archivo ya existe
if not os.path.exists(file_path):
    # Guardar el DataFrame en un archivo CSV si no existe
    matrix_df.to_csv(file_path, index=False)
else:
    print(f"El archivo {file_path} ya existe. No se sobrescribió.")

total_sum_small_matrix = sum_small_matrices(small_folder)
matrix_df = pd.DataFrame(total_sum_matrix)

# Ruta del archivo CSV
file_path = 'genGD.csv'

# Verificar si el archivo ya existe
if not os.path.exists(file_path):
    # Guardar el DataFrame en un archivo CSV si no existe
    matrix_df.to_csv(file_path, index=False)
else:
    print(f"El archivo {file_path} ya existe. No se sobrescribió.")

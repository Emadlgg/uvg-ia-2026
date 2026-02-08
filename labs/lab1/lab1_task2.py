import pandas as pd
import numpy as np

# 1. GENERACIÓN DE DATASET SUCIO

np.random.seed(48)  # Para que los resultados sean reproducibles
n_filas = 100

# Generar datos
edades = np.random.randint(18, 70, size=n_filas)
salarios = np.random.randint(20000, 120000, size=n_filas)

# Crear desbalance: 90 ceros, 10 unos
compro_producto = np.array([0] * 90 + [1] * 10)
np.random.shuffle(compro_producto)

# Crear DataFrame
df = pd.DataFrame({
    'Edad': edades,
    'Salario': salarios,
    'Compró_Producto': compro_producto
})

# Introducir 10% de valores NaN en Edad
indices_nulos = np.random.choice(df.index, size=10, replace=False)
df.loc[indices_nulos, 'Edad'] = np.nan

print("DATASET ORIGINAL:")
print(df.head(15))
print(f"\nNulos en Edad: {df['Edad'].isna().sum()}")
print(f"Clase 0: {(df['Compró_Producto']==0).sum()}, Clase 1: {(df['Compró_Producto']==1).sum()}\n")

# 2. IMPUTACIÓN DE DATOS FALTANTES

# Calcular promedio de edades (ignora NaN automáticamente)
promedio_edad = df['Edad'].mean()

# Recorrer y rellenar NaN con el promedio
for i in range(len(df)):
    if pd.isna(df.loc[i, 'Edad']):
        df.loc[i, 'Edad'] = promedio_edad

print("DESPUÉS DE IMPUTACIÓN:")
print(f"Nulos en Edad: {df['Edad'].isna().sum()}")
print(f"Promedio usado: {promedio_edad:.2f}\n")

"""
¿En qué situación usar el promedio sería una mala idea y sería mejor usar la mediana?

RESPUESTA:
El promedio sería una mala idea cuando hay OUTLIERS (valores extremos) en los datos.

Por ejemplo:
- Si tenemos edades: [25, 28, 30, 32, 35, 90]
- Promedio = 40 años (afectado por el 90)
- Mediana = 31 años (más representativa del grupo)

El outlier de 90 años distorsiona el promedio hacia arriba. Si usamos 40 para 
rellenar datos faltantes de personas que probablemente tienen alrededor de 30 años,
estaríamos introduciendo un sesgo.

La mediana es más robusta ante outliers porque representa el valor central de los
datos ordenados, sin importar cuán extremos sean los valores en los extremos.

CASOS PRÁCTICOS:
- Salarios (hay CEOs con salarios extremos)
- Precios de casas (hay mansiones que distorsionan el promedio)
- Tiempos de respuesta en sistemas (puede haber casos extremos de lentitud)
"""

# 3. UNDERSAMPLING MANUAL

def undersampling_manual(dataframe, columna_clase):
    # Separar por clases
    clase_minoritaria = dataframe[dataframe[columna_clase] == 1]
    clase_mayoritaria = dataframe[dataframe[columna_clase] == 0]
    
    # Contar elementos de clase minoritaria
    n_minoritaria = len(clase_minoritaria)
    
    # Tomar muestra aleatoria de la clase mayoritaria del mismo tamaño
    clase_mayoritaria_reducida = clase_mayoritaria.sample(n=n_minoritaria, random_state=42)
    
    # Combinar y mezclar
    df_balanceado = pd.concat([clase_minoritaria, clase_mayoritaria_reducida])
    df_balanceado = df_balanceado.sample(frac=1, random_state=42).reset_index(drop=True)
    
    return df_balanceado

df_balanceado = undersampling_manual(df, 'Compró_Producto')

print("DESPUÉS DE UNDERSAMPLING:")
print(f"Total filas: {len(df_balanceado)}")
print(f"Clase 0: {(df_balanceado['Compró_Producto']==0).sum()}, Clase 1: {(df_balanceado['Compró_Producto']==1).sum()}")
print(f"\n{df_balanceado.head(10)}")
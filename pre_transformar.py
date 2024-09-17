import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import LabelEncoder
from scipy import stats

data = pd.read_csv('train.csv')

# Exploración inicial del conjunto de datos
print(data.head())
print(data.shape)
print(data.info())
print(data.describe())

# Seleccionar las variables importantes para el análisis
selected_columns = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 
                    'TotalBsmtSF', 'YearBuilt', 'FullBath', 'LotArea', 'MoSold', 'YrSold']

# Mostrar todos los valores faltantes
valores_faltantes = data.isnull().sum()
print('\n Valores faltantes \n')
print(valores_faltantes[valores_faltantes > 0])

# Identificar todas las variables que contienen valores faltantes
print(data.columns[data.isnull().any()])
print('\n Proporcion en % \n')
print(data.isnull().sum()/data.shape[0]*100)

# Imputar valores faltantes para las columnas numéricas con la mediana
data['SalePrice'].fillna(data['SalePrice'].median(), inplace=True)
data['OverallQual'].fillna(data['OverallQual'].median(), inplace=True)
data['GrLivArea'].fillna(data['GrLivArea'].median(), inplace=True)
data['GarageCars'].fillna(data['GarageCars'].median(), inplace=True)
data['TotalBsmtSF'].fillna(data['TotalBsmtSF'].median(), inplace=True)
data['YearBuilt'].fillna(data['YearBuilt'].median(), inplace=True)
data['FullBath'].fillna(data['FullBath'].median(), inplace=True)
data['LotArea'].fillna(data['LotArea'].median(), inplace=True)
data['MoSold'].fillna(data['MoSold'].median(), inplace=True)
data['YrSold'].fillna(data['YrSold'].median(), inplace=True)

# Imputar valores faltantes para las columnas categóricas con la moda
data['BsmtQual'].fillna(data['BsmtQual'].mode()[0], inplace=True)
data['KitchenQual'].fillna(data['KitchenQual'].mode()[0], inplace=True)
data['Neighborhood'].fillna(data['Neighborhood'].mode()[0], inplace=True)

# Verificar que los valores faltantes hayan sido imputados correctamente
print("\nValores faltantes después de la mediana y moda\n")
print(data[['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 
            'TotalBsmtSF', 'YearBuilt', 'FullBath', 'LotArea', 
            'BsmtQual', 'KitchenQual', 'Neighborhood', 'MoSold', 
            'YrSold']].isnull().sum())

# Z-score para detección de outliers
numerical_columns = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'YearBuilt', 'FullBath', 'LotArea', 'MoSold', 'YrSold']
numerical_data = data[numerical_columns]
z_scores = np.abs(stats.zscore(numerical_data))
outlier_amplitud = 3
outliers = (z_scores > outlier_amplitud)

# Función para comparar los gráficos de las columnas importantes antes y después de eliminar los outliers
def plot_outliers_comparison(original_data, cleaned_data, columns):
    for column in columns:
        plt.figure(figsize=(14, 6))

        # Gráfico de la variable original con outliers
        plt.subplot(1, 2, 1)
        sns.histplot(original_data[column], kde=True)
        plt.title(f'{column}: Con Outliers')

        # Gráfico de la variable sin outliers
        plt.subplot(1, 2, 2)
        sns.histplot(cleaned_data[column], kde=True)
        plt.title(f'{column}: Sin Outliers')

        plt.show()

# Identificar y eliminar los outliers usando Z-score
threshold = 3
outliers_condition = (np.abs(stats.zscore(data[numerical_columns])) < threshold).all(axis=1)

# Crear una copia de los datos sin outliers
data_no_outliers = data[outliers_condition]

# Comparar los gráficos de todas las columnas importantes antes y después de eliminar los outliers
plot_outliers_comparison(data, data_no_outliers, selected_columns)

# Mostrar la cantidad de outliers por columna
outliers_count = np.sum(outliers, axis=0)

print('\n Cantidad de outliers\n')
print(outliers_count)

# Aplicar Min-Max Scaling y Z-score Estandarización
scaler_minmax = MinMaxScaler()
scaler_standard = StandardScaler() 

# Aplicar Min-Max Scaling a las columnas numéricas
minmax_scaled_data = scaler_minmax.fit_transform(data[numerical_columns])
minmax_scaled_df = pd.DataFrame(minmax_scaled_data, columns=numerical_columns)

# Aplicar Z-score Estandarización a las columnas numéricas
zscore_scaled_data = scaler_standard.fit_transform(data[numerical_columns])
zscore_scaled_df = pd.DataFrame(zscore_scaled_data, columns=numerical_columns)

# Agregar los gráficos de comparación Min-Max Scaling y Z-Score
def plot_scaling_comparison(minmax_data, zscore_data, columns):
    for column in columns:
        plt.figure(figsize=(14, 6))

        # Gráfico de Min-Max Scaling
        plt.subplot(1, 2, 1)
        sns.histplot(minmax_data[column], kde=True)
        plt.title(f'{column}: Min-Max Scaling')

        # Gráfico de Z-score Estandarización
        plt.subplot(1, 2, 2)
        sns.histplot(zscore_data[column], kde=True)
        plt.title(f'{column}: Estandarización')

        plt.show()

# Seleccionar las columnas para comparar
scaled_columns = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF']

# Crear DataFrames de los datos escalados para usar en los gráficos
minmax_scaled_df = pd.DataFrame(minmax_scaled_data, columns=numerical_columns)
zscore_scaled_df = pd.DataFrame(zscore_scaled_data, columns=numerical_columns)

# Mostrar los gráficos que comparan Min-Max Scaling y Z-score Estandarización
plot_scaling_comparison(minmax_scaled_df, zscore_scaled_df, scaled_columns)

# Mostrar los primeros resultados de Min-Max Scaling y Z-score
print('\nMin-Max:')
print(minmax_scaled_df.head())

print("\nZ-Score:")
print(zscore_scaled_df.head())

# Evaluar el sesgo de las columnas numéricas
skewness = data[selected_columns].apply(lambda x: stats.skew(x))
print(f'\nSesgo de las columnas seleccionadas:\n{skewness}')

# Identificar las columnas con sesgo alto (skewness > 0.5 o skewness < -0.5)
skewed_columns = skewness[skewness.abs() > 0.5].index
print(f'\nColumnas con sesgo alto:\n{skewed_columns}')

# Aplicar la transformación logarítmica a las columnas sesgadas
for col in skewed_columns:
    data[col + '_log'] = np.log1p(data[col])  # log1p maneja ceros

# Función para comparar la distribución original y transformada
def plot_distribution(original, transformed, column):
    plt.figure(figsize=(14, 6))

    # Gráfico de la variable original
    plt.subplot(1, 2, 1)
    sns.histplot(original, kde=True)
    plt.title(f'Distribución original de {column}')

    # Gráfico de la variable transformada
    plt.subplot(1, 2, 2)
    sns.histplot(transformed, kde=True)
    plt.title(f'Distribución logarítmica de {column}')
    
    plt.show()

# Mostrar los gráficos de las columnas con sesgo antes y después de la transformación logarítmica
for col in skewed_columns:
    plot_distribution(data[col], data[col + '_log'], col)

transformed_skewness = data[[col + '_log' for col in skewed_columns]].apply(lambda x: stats.skew(x))
print(f'\nSesgo de las columnas seleccionadas después de la transformación:\n{transformed_skewness}')

# Creacion de la variable House_Age
data['House_Age'] = data['YrSold'] - data['YearBuilt']
print("\n Variable House Age \n")
print(data[['House_Age']].head())

# Creacion de la variable TotalOusideArea
data['Total_Outside_Area'] = data['OpenPorchSF'] + data['EnclosedPorch'] + data['3SsnPorch'] + data['ScreenPorch']
print("\n Variable Total Outside Area \n")
print(data[['Total_Outside_Area']].head())

# Codificacion label encoding

# Crear el codificador
label_encoder = LabelEncoder()

# Codificar 'House_Age_Category'
data['House_Age_Category'] = pd.cut(data['House_Age'], bins=[0, 10, 30, 100], 
                                    labels=['Nuevo', 'Moderno', 'Antiguo'], include_lowest=True)
data['House_Age_Category_Encoded'] = label_encoder.fit_transform(data['House_Age_Category'])

# Codificar 'Total_Outside_Area_Category'
data['Total_Outside_Area_Category'] = pd.cut(data['Total_Outside_Area'], bins=[0, 100, 300, 1000], 
                                             labels=['Pequeño', 'Mediano', 'Grande'], include_lowest=True)
data['Total_Outside_Area_Category_Encoded'] = label_encoder.fit_transform(data['Total_Outside_Area_Category'])

# Mostrar las variables codificadas
print(data[['House_Age_Category']].head())
print(data[['House_Age_Category_Encoded']].head())
print(data[['Total_Outside_Area_Category']].head())
print(data[['Total_Outside_Area_Category_Encoded']].head())
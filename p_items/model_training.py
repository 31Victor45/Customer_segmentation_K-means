# model_training.py
# Este script ahora se ejecutará con scikit-learn==1.5.1
# para crear un modelo compatible con tu app de Streamlit.

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from joblib import dump

# Cargar el conjunto de datos
# Asegúrate de que este archivo esté en la misma carpeta
df = pd.read_csv("segmented_customers.csv")

# Identificar las características (X) y la variable objetivo (y)
X = df.drop('Cluster', axis=1)
y = df['Cluster']

# Separar columnas categóricas y numéricas
categorical_features = X.select_dtypes(include=['object']).columns
numerical_features = X.select_dtypes(include=['float64', 'int64']).columns

# Crear las tuberías de preprocesamiento
numerical_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown='ignore')

# Combinar ambas tuberías
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Crear la tubería completa con preprocesamiento y el modelo
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

# Dividir los datos
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entrenar el modelo
print("Entrenando el modelo...")
model_pipeline.fit(X_train, y_train)
print("Entrenamiento del modelo completado.")

# Evaluar el modelo (opcional, pero buena práctica)
accuracy = model_pipeline.score(X_test, y_test)
print(f"Precisión del modelo en el conjunto de prueba: {accuracy:.2%}")

# Guardar la tubería entrenada en un nuevo archivo
# Esto reemplazará el archivo antiguo y solucionará el problema de compatibilidad
dump(model_pipeline, 'customer_cluster_model.joblib')
print("La tubería del modelo se ha guardado como 'customer_cluster_model.joblib'")

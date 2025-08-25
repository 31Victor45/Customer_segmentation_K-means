import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer

# --- Configuración de la página ---
st.set_page_config(
    page_title="Análisis de Segmentación de Clientes",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Título de la aplicación ---
st.title("🛍️ Segmentación de Clientes")
st.markdown("Un análisis de segmentación de clientes utilizando el algoritmo K-Means. La aplicación preprocesa los datos, entrena un modelo y lo utiliza para predecir el segmento de un nuevo cliente.")

# --- Funciones para entrenamiento y predicción ---

@st.cache_data
def load_and_preprocess_data():
    """
    Carga el conjunto de datos de clientes, selecciona las columnas
    relevantes y preprocesa los datos.
    
    Returns:
        tuple: (DataFrame de datos, ColumnTransformer preajustado,
                 DataFrame de datos preprocesados)
    """
    try:
        df = pd.read_csv("segmented_customers.csv")
    except FileNotFoundError:
        st.error("Error: Archivo 'segmented_customers.csv' no encontrado. Por favor, asegúrese de que el archivo existe en la misma carpeta.")
        st.stop()
    
    # Seleccionamos las características relevantes para la segmentación
    features = ['Purchase Amount (USD)', 'Review Rating', 'Previous Purchases']
    df_features = df[features]
    
    # Definimos el preprocesador: un ColumnTransformer con MinMaxScaler
    preprocessor = ColumnTransformer(
        transformers=[
            ('scaler', MinMaxScaler(), features)
        ],
        remainder='passthrough'
    )
    
    # Ajustamos y transformamos los datos con el preprocesador
    preprocessed_data = preprocessor.fit_transform(df_features)
    
    return df, preprocessor, preprocessed_data

@st.cache_resource
def train_and_save_model(data, preprocessor):
    """
    Entrena el modelo K-Means, lo guarda junto con el preprocesador
    y retorna el modelo ajustado.
    
    Args:
        data (DataFrame): Datos preprocesados para el entrenamiento.
        preprocessor (ColumnTransformer): El objeto de preprocesamiento ajustado.
    
    Returns:
        KMeans: El modelo K-Means ajustado.
    """
    # Usamos K-Means para segmentar a los clientes en 4 clusters
    # El número de clusters (n_clusters) puede ser ajustado
    kmeans_model = KMeans(n_clusters=4, random_state=42, n_init=10)
    kmeans_model.fit(data)
    
    # Guardamos el preprocesador y el modelo ajustado
    # Esto es crucial para la implementación en Streamlit Cloud
    joblib.dump(preprocessor, 'preprocessor.pkl')
    joblib.dump(kmeans_model, 'kmeans_model.pkl')
    
    return kmeans_model

def load_saved_model_and_preprocessor():
    """
    Carga el preprocesador y el modelo K-Means desde los archivos .pkl.
    """
    try:
        preprocessor = joblib.load('preprocessor.pkl')
        kmeans_model = joblib.load('kmeans_model.pkl')
        return preprocessor, kmeans_model
    except FileNotFoundError:
        st.error("Error al cargar los modelos. Por favor, asegúrese de que 'preprocessor.pkl' y 'kmeans_model.pkl' están en el mismo directorio.")
        st.info("Para generar los archivos .pkl, ejecute este script en su entorno local.")
        return None, None
        
# --- Carga y entrenamiento de los modelos ---
df, preprocessor, preprocessed_data = load_and_preprocess_data()
kmeans_model = train_and_save_model(preprocessed_data, preprocessor)

# Añadimos la columna del cluster al DataFrame original
df['Cluster'] = kmeans_model.labels_

st.header("📊 Resumen de los Clusters")
st.dataframe(df.groupby('Cluster')[['Purchase Amount (USD)', 'Review Rating', 'Previous Purchases']].mean())

# --- Interfaz para predecir un nuevo cliente ---
st.sidebar.header("Predice el Segmento de un Nuevo Cliente")

# Creamos las entradas para el usuario en la barra lateral
purchase_amount = st.sidebar.number_input("Monto de la Compra (USD)", min_value=0.0, max_value=100.0, value=50.0)
review_rating = st.sidebar.slider("Calificación de la Reseña", min_value=1.0, max_value=5.0, value=3.5, step=0.1)
previous_purchases = st.sidebar.number_input("Compras Anteriores", min_value=0, value=10)

# Botón para predecir
if st.sidebar.button("Predecir Cluster"):
    # Cargamos el preprocesador y el modelo guardados
    preprocessor, loaded_model = load_saved_model_and_preprocessor()
    
    if preprocessor and loaded_model:
        # Creamos un DataFrame con los datos del nuevo cliente
        new_customer_data = pd.DataFrame([[purchase_amount, review_rating, previous_purchases]],
                                         columns=['Purchase Amount (USD)', 'Review Rating', 'Previous Purchases'])
        
        # Preprocesamos los datos del nuevo cliente
        preprocessed_new_customer = preprocessor.transform(new_customer_data)
        
        # Predecimos el cluster
        predicted_cluster = loaded_model.predict(preprocessed_new_customer)[0]
        
        # Mostramos el resultado
        st.sidebar.success(f"El nuevo cliente pertenece al **Cluster {predicted_cluster}**.")

st.markdown("""
---
### Pasos para Solucionar el Error:
1.  **Revisa el `requirements.txt`** para asegurarte de que `joblib` está incluido.
2.  **Guarda este código** como `app.py` en tu máquina local.
3.  **Ejecuta el script** en tu entorno. Esto creará los archivos `preprocessor.pkl` y `kmeans_model.pkl`.
4.  **Sube los tres archivos** (`app.py`, `requirements.txt` y los dos archivos `.pkl`) a tu repositorio de GitHub.
5.  **Despliega de nuevo** tu aplicación en Streamlit Cloud.
""")

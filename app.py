# app.py

import streamlit as st
import pandas as pd
from joblib import load

# Cargar el modelo pre-entrenado
# Asegúrate de que el archivo 'customer_cluster_model.joblib' esté en el mismo directorio
try:
    model_pipeline = load('p_items/customer_cluster_model.joblib')
    # st.success("¡Modelo cargado exitosamente!")
except FileNotFoundError:
    st.error("Error: No se encontró el archivo 'customer_cluster_model.joblib'. Por favor, ejecuta el script de entrenamiento primero.")
    st.stop() # Detiene la aplicación si el archivo del modelo no se encuentra

# Obtener una lista de todos los valores posibles para las características categóricas
# Esta es una buena práctica para asegurar que la app use entradas válidas
try:
    df = pd.read_csv("p_items/segmented_customers.csv")
except FileNotFoundError:
    st.error("Error: No se encontró el archivo 'segmented_customers.csv'. Asegúrate de que esté en el mismo directorio.")
    st.stop()

# Obtener una lista de todos los valores únicos de cada columna categórica
unique_values = {col: df[col].unique().tolist() for col in df.select_dtypes(include='object').columns}

# --- BARRA LATERAL (SIDEBAR) FIJA ---
with st.sidebar:
    # Se ha reemplazado use_column_width por use_container_width para evitar el aviso
    st.image("p_items/img/info.png", use_container_width=True)
    st.title("Acerca de esta aplicación")
    st.markdown(
        """
        Esta aplicación de segmentación de clientes utiliza un modelo de machine learning para 
        predecir a qué clúster de clientes pertenece un nuevo individuo, basándose en sus 
        características demográficas y de compra. 
        
        La clasificación ayuda a las empresas a personalizar estrategias de marketing y 
        a ofrecer una mejor experiencia al cliente.
        """
    )
    
    # Botón para navegar a la landing page
    st.markdown("---")
    st.markdown("### Navegación")
    
    # Se ha reemplazado el st.markdown con un botón de enlace nativo
    # El usuario debe reemplazar #url_de_la_web con la URL real de su landing page
    st.link_button("Ver perfiles de clientes", url="#url_de_la_web")
    
    # Aclaración para el usuario sobre el enlace
    st.info("Reemplaza '#url_de_la_web' con la URL real de tu landing page.")
    
# --- INTERFAZ DE USUARIO PRINCIPAL ---
st.title("Aplicación de Segmentación de Clientes")
st.markdown("## Predice el clúster de tu cliente")
st.write("Ingresa las características del cliente a continuación para predecir su segmento.")

# Crear formularios de entrada para cada característica
with st.form("customer_form"):
    st.header("Información del Cliente")

    # Características numéricas
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        age = st.number_input("Edad", min_value=18, max_value=80, value=35)
    with col2:
        purchase_amount = st.number_input("Monto de Compra (USD)", min_value=0, max_value=200, value=50)
    with col3:
        review_rating = st.slider("Calificación de Reseña", min_value=1.0, max_value=5.0, value=3.5, step=0.1)
    with col4:
        previous_purchases = st.number_input("Compras Previas", min_value=0, max_value=100, value=10)

    # Características categóricas
    col5, col6 = st.columns(2)
    with col5:
        gender = st.selectbox("Género", options=unique_values['Gender'])
        item_purchased = st.selectbox("Artículo Comprado", options=unique_values['Item Purchased'])
        category = st.selectbox("Categoría", options=unique_values['Category'])
        location = st.selectbox("Ubicación", options=unique_values['Location'])
        size = st.selectbox("Talla", options=unique_values['Size'])
        color = st.selectbox("Color", options=unique_values['Color'])
        season = st.selectbox("Estación", options=unique_values['Season'])
    with col6:
        subscription_status = st.selectbox("Estado de Suscripción", options=unique_values['Subscription Status'])
        payment_method = st.selectbox("Método de Pago", options=unique_values['Payment Method'])
        shipping_type = st.selectbox("Tipo de Envío", options=unique_values['Shipping Type'])
        discount_applied = st.selectbox("Descuento Aplicado", options=unique_values['Discount Applied'])
        promo_code_used = st.selectbox("Código Promocional Usado", options=unique_values['Promo Code Used'])
        preferred_payment = st.selectbox("Método de Pago Preferido", options=unique_values['Preferred Payment Method'])
        frequency = st.selectbox("Frecuencia de Compras", options=unique_values['Frequency of Purchases'])

    submitted = st.form_submit_button("Predecir Clúster")

# Cuando el usuario hace clic en el botón de predecir
if submitted:
    # Crear un DataFrame con las entradas del usuario
    input_data = pd.DataFrame([{
        'Age': age,
        'Gender': gender,
        'Item Purchased': item_purchased,
        'Category': category,
        'Purchase Amount (USD)': purchase_amount,
        'Location': location,
        'Size': size,
        'Color': color,
        'Season': season,
        'Review Rating': review_rating,
        'Subscription Status': subscription_status,
        'Payment Method': payment_method,
        'Shipping Type': shipping_type,
        'Discount Applied': discount_applied,
        'Promo Code Used': promo_code_used,
        'Previous Purchases': previous_purchases,
        'Preferred Payment Method': preferred_payment,
        'Frequency of Purchases': frequency,
    }])
    
    # Hacer la predicción
    predicted_cluster = model_pipeline.predict(input_data)[0]

    # Mostrar el resultado
    st.markdown(f"### El clúster predicho para este cliente es: **Clúster {predicted_cluster}**")
    st.info("Recuerda que las características del clúster se determinaron a través de análisis de datos previos. Esta predicción ayuda a asignar nuevos individuos a esos grupos.")
    # Nota agregada para guiar al usuario a la barra lateral
    st.info("¡No olvides! En la barra lateral (sidebar) encontrarás un botón que te lleva a la web para consultar las características detalladas de cada clúster.")

import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Cargar modelo, encoders y scaler
model = joblib.load('obesity_model.pkl')
le_dict = joblib.load('label_encoders.pkl')
scaler = joblib.load('scaler.pkl')

# Columnas
cat_cols = [
    'Gender',
    'family_history_with_overweight',
    'FAVC',
    'CAEC',
    'SMOKE',
    'SCC',
    'CALC',
    'MTRANS'
]

num_cols = [
    'Age',
    'Height',
    'Weight',
    'FCVC',
    'NCP',
    'CH2O',
    'FAF',
    'TUE'
]

model_cols = [
    'Gender',
    'Age',
    'Height',
    'Weight',
    'family_history_with_overweight',
    'FAVC',
    'FCVC',
    'NCP',
    'CAEC',
    'SMOKE',
    'CH2O',
    'SCC',
    'FAF',
    'TUE',
    'CALC',
    'MTRANS'
]

# Opciones
options = {
    'Gender': ['Male', 'Female'],
    'family_history_with_overweight': ['yes', 'no'],
    'FAVC': ['yes', 'no'],
    'CAEC': ['no', 'Sometimes', 'Frequently', 'Always'],
    'SMOKE': ['yes', 'no'],
    'SCC': ['yes', 'no'],
    'CALC': ['no', 'Sometimes', 'Frequently', 'Always'],
    'MTRANS': ['Automobile', 'Motorbike', 'Bike', 'Public_Transportation', 'Walking']
}

# Interfaz
st.title('Evalúa tu nivel de obesidad')

# Formulario
with st.form(key='input_form'):
    st.header('Datos personales y hábitos')
    
    features = {}
    
    # Campos categóricos
    cols = st.columns(2)
    for i, (field, vals) in enumerate(options.items()):
        with cols[i % 2]:
            raw_val = st.selectbox(field.replace('_', ' ').title(), vals)
            features[field] = raw_val
    
    st.header('Mediciones físicas')
    
    cols = st.columns(3)
    age = st.number_input('Edad (años)', min_value=14, max_value=90, value=25)
    height = st.number_input('Altura (m)', min_value=1.3, max_value=2.1, value=1.70, step=0.01)
    weight = st.number_input('Peso (kg)', min_value=30, max_value=250, value=70)
    fcvc = st.number_input('Verduras al día (0-3)', min_value=0.0, max_value=3.0, value=2.0, step=0.1)
    ncp = st.number_input('Comidas al día (1-4)', min_value=1.0, max_value=4.0, value=3.0, step=0.1)
    ch2o = st.number_input('Agua al día (1-3)', min_value=1.0, max_value=3.0, value=2.0, step=0.1)
    faf = st.number_input('Actividad física (0-3)', min_value=0.0, max_value=3.0, value=1.0, step=0.1)
    tue = st.number_input('Tiempo en pantalla (0-2)', min_value=0.0, max_value=2.0, value=1.0, step=0.1)
    
    features.update({
        'Age': age,
        'Height': height,
        'Weight': weight,
        'FCVC': fcvc,
        'NCP': ncp,
        'CH2O': ch2o,
        'FAF': faf,
        'TUE': tue
    })
    
    submit = st.form_submit_button('Predecir nivel de obesidad')

if submit:
    # Procesar inputs
    input_dict = {}
    for col in cat_cols:
        raw_val = features[col]
        input_dict[col] = le_dict[col].transform([raw_val])[0]
    
    for col in num_cols:
        input_dict[col] = features[col]
    
    df_input = pd.DataFrame([input_dict])
    
    # Escalar numéricas
    df_input[num_cols] = scaler.transform(df_input[num_cols])
    
    # Ordenar columnas
    df_input = df_input[model_cols]
    
    # Predicción
    pred = model.predict(df_input)[0]
    prediction = le_dict['NObeyesdad'].inverse_transform([pred])[0]
    
    st.success(f'Resultado estimado: {prediction}')
    st.info('Consulta a un profesional para recomendaciones personalizadas.')
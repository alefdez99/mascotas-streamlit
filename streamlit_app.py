import streamlit as st
import joblib
import pandas as pd
import json

st.title("Classipet")
st.write("Clasificador de mascotas")
st.image("img/mascotas.jpg", use_container_width=True)

# Carga el modelo entrenado y las asignaciones para el color de ojos y el largo del pelo
model = joblib.load("model/pets_model.joblib")
with open("model/category_mapping.json", "r") as f:
    category_mapping = json.load(f)

# Extrae los valores categóricos de las asignaciones
eye_color_values = category_mapping["eye_color"]
fur_length_values = category_mapping["fur_length"]

# Crea los inputs para la entrada de datos
weight = st.number_input("Peso (kg)", min_value=0.0, max_value=100.0, value=10.0)
height = st.number_input("Altura (cm)", min_value=10.0, max_value=100.0, value=10.0)

eye_color = st.selectbox("Color de ojos", ['Azul', 'Marrón', 'Gris', 'Verde'])
fur_length = st.selectbox("Largo del pelo", ['Corto', 'Mediano', 'Largo'])

# Mapea la selección de color de ojos y el largo del pelo al español
eye_color_map = {"Marrón": "brown", "Azul": "blue", "Gris": "gray", "Verde": "green"}
fur_length_map = {"Corto": "short", "Mediano": "medium", "Largo": "long"}

selected_eye_color = eye_color_map[eye_color]
selected_fur_length = fur_length_map[fur_length]

# Genera las columnas binarias para eye_color y fur_length
#eye_color_binary = [int(color == selected_eye_color) for color in eye_color_values]
eye_color_binary = [(color == selected_eye_color) for color in eye_color_values]
fur_length_binary = [(length == selected_fur_length) for length in fur_length_values]

# Crea un dataframe con los datos de la mascota
input_data = [weight, height] + eye_color_binary + fur_length_binary
columns = ["weight_kg", "height_cm"] + [f"eye_color_{color}" for color in eye_color_values] + [f"fur_length_{length}" for length in fur_length_values]
input_df = pd.DataFrame([input_data], columns=columns)

if st.button("Clasifica la mascota"):
    # Realiza la predición
    prediction = model.predict(input_df)[0]

    # Mapea la predicción
    prediction_map = {"dog": "Perro", "cat": "Gato", "rabbit": "Conejo"}

    # Muestra la predicción
    st.success(f"La mascota es un {prediction_map[prediction]}", icon = "✅")


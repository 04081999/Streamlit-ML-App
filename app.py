import streamlit as st
import pickle
import numpy as np

st.title("Application Machine Learning - Streamlit")

# Charger le modèle
model = pickle.load(open("model.pkl", "rb"))

st.header("Entrer les informations")

age = st.number_input("Âge", 18, 100)
revenu = st.number_input("Revenu mensuel")
niveau_etude = st.selectbox(
    "Niveau d'étude",
    ["Aucun", "Secondaire", "Supérieur"]
)

# Encoder niveau d'étude
niveau_map = {"Aucun": 0, "Secondaire": 1, "Supérieur": 2}
niveau_encoded = niveau_map[niveau_etude]

if st.button("Prédire"):
    prediction = model.predict([[age, revenu, niveau_encoded]])
    result = "Oui (Chômeur)" if prediction[0] == 1 else "Non (Employé)"

    st.success(f"Résultat de la prédiction : {result}")
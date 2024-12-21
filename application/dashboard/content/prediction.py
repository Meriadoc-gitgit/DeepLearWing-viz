import streamlit as st
import pandas as pd 
import os
import numpy as np

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from copy import deepcopy
from src.improved_aerodynamic_model import *

import joblib

# Configuration générale pour tous les plots
layout_config = dict(
    width=750,  # Largeur augmentée
    height=500,  # Hauteur augmentée
    font=dict(size=14)  # Taille de police augmentée
)

def prediction(df_sample):
    """
    Predict the Lift-to-Drag ratio using the trained XGBoost model.
    """
    print(os.getcwd())
    model = joblib.load('dashboard/content/xgboost_model.joblib')

    st.markdown("## Modélisation & Prédiction")
    st.markdown("""
    Dans cette section, nous présentons les performances du modèle XGBoost, l'importance des variables, et proposons un outil de prédiction interactif.
    """)

    # Affichage des performances du modèle
    st.markdown("### Performances du Modèle XGBoost")
    metrics_df = display_model_performance(model)
    st.dataframe(metrics_df)

    # Importance des Features
    st.markdown("### Importance des Caractéristiques par Régime")
    regime_choice = st.selectbox("Choisissez un régime d'écoulement", ["laminar", "transition", "turbulent"])
    if regime_choice in model.feature_importances:
        importance_df = model.feature_importances[regime_choice].copy()
        importance_df = importance_df.head(15)  
        
        fig_importance = px.bar(
            importance_df,
            x='Importance',
            y='Feature',
            orientation='h',
            title=f"Importance des Features - Régime {regime_choice.capitalize()}",
            color='Importance', 
            color_continuous_scale='magma',
        )
        fig_importance.update_layout(yaxis={'categoryorder':'total ascending'}, **layout_config)
        st.plotly_chart(fig_importance, use_container_width=True)

    st.markdown("### Prédiction du Ratio L/D")
    st.markdown("""
    Exemple de prédiction pour des conditions de vol typiques :
    - Angle d'attaque de 5° (écoulement attaché)
    - Nombre de Reynolds de 500,000 (représentatif d'un petit avion)
    """)
    
    # Valeurs par défaut pour l'exemple
    angle_input = st.slider("Angle d'attaque (°)", -10, 20, 5)
    reynolds_input = st.slider("Nombre de Reynolds", 100000, 1000000, 500000)
    
    # Préparer les données d'entrée
    input_data = model.prepare_input_data(angle_input, reynolds_input)
    
    if st.button("Prédire le Ratio L/D"):
        try:
            regime = 'transition'
            predicted_ratio = model.regime_models[regime]['model'].predict(input_data)
            
            # Zones de performance et couleurs
            if predicted_ratio[0] > 60:
                performance_level = "exceptionnel"
                message_color = "blue"
                message_function = st.info
            elif predicted_ratio[0] > 45:
                performance_level = "très performant"
                message_color = "green"
                message_function = st.success
            elif predicted_ratio[0] > 30:
                performance_level = "moyen"
                message_color = "orange"
                message_function = st.warning
            elif predicted_ratio[0] > 20:
                performance_level = "peu performant"
                message_color = "red"
                message_function = st.error
            elif predicted_ratio[0] > 0:
                performance_level = "critique"
                message_color = "darkred"
                message_function = st.error
            else:
                performance_level = "négatif - décrochage sévère"
                message_color = "purple"
                message_function = st.error
            
            message = f"""
            **Prédiction pour les conditions spécifiées :**
            - Angle d'attaque : {angle_input}°
            - Nombre de Reynolds : {reynolds_input:,}
            - **Ratio L/D prédit : {predicted_ratio[0]:.2f}**
            
            Ce ratio est considéré comme **{performance_level}** pour un profil d'aile.
            Plus le ratio est élevé, meilleure est la performance (plus de portance pour moins de traînée).
            
            Repères de performance :
            - Négatif : < 0 (décrochage sévère)
            - Critique : 0-20
            - Peu performant : 20-30
            - Moyen : 30-45
            - Très performant : 45-60
            - Exceptionnel : >60
            """
            
            message_function(message)
            
            fig = go.Figure()
            
            # Zones de performance (incluant les valeurs négatives)
            fig.add_hrect(y0=-20, y1=0, fillcolor="purple", opacity=0.1, line_width=0,
                         annotation_text="Décrochage sévère", annotation_position="right")
            fig.add_hrect(y0=0, y1=20, fillcolor="darkred", opacity=0.1, line_width=0,
                         annotation_text="Critique", annotation_position="right")
            fig.add_hrect(y0=20, y1=30, fillcolor="red", opacity=0.1, line_width=0,
                         annotation_text="Peu performant", annotation_position="right")
            fig.add_hrect(y0=30, y1=45, fillcolor="yellow", opacity=0.1, line_width=0,
                         annotation_text="Moyen", annotation_position="right")
            fig.add_hrect(y0=45, y1=60, fillcolor="green", opacity=0.1, line_width=0,
                         annotation_text="Très performant", annotation_position="right")
            fig.add_hrect(y0=60, y1=100, fillcolor="blue", opacity=0.1, line_width=0,
                         annotation_text="Exceptionnel", annotation_position="right")
            
            # Point actuel uniquement
            fig.add_trace(go.Scatter(
                x=[angle_input],
                y=[predicted_ratio[0]],
                mode='markers',
                name='Point actuel',
                marker=dict(color='red', size=15, symbol='circle')
            ))
            
            fig.update_layout(
                title=f"Ratio L/D pour l'angle choisi (Re = {reynolds_input:,})",
                xaxis_title="Angle d'attaque (°)",
                yaxis_title="Ratio L/D prédit",
                height=500,
                showlegend=True
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.error(f"Erreur lors de la prédiction : {str(e)}")
            st.write("Détails de l'erreur pour le débogage :", e.__class__.__name__)

    st.markdown("""
    Les résultats montrent que certaines variables géométriques et l'angle ont une grande influence sur la performance (L/D), 
    tandis que le régime d'écoulement (lié au Reynolds) joue un rôle crucial dans la distribution de ces performances.
    """)
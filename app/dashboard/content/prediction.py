import streamlit as st
import pandas as pd 
import os

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from copy import deepcopy

# Configuration générale pour tous les plots
layout_config = dict(
    width=750,  # Largeur augmentée
    height=500,  # Hauteur augmentée
    font=dict(size=14)  # Taille de police augmentée
)
from src.improved_aerodynamic_model import *

def prediction(df_sample):
    """"""
    # df_sample = pd.read_csv('data/echantillon_stratifie.csv')

    df_model = df_sample.copy()
    model = ImprovedAerodynamicModel(df_model)
    model.add_geometric_features()
    model.create_advanced_features()
    model.segment_flow_regimes()
    model.prepare_features()
    model.train()
    
    # if not os.path.exists('model/xgboost_model.joblib'):
    #     model = train_and_save_model()
    # else:
    #     model = ImprovedAerodynamicModel.load_model('model/xgboost_model.joblib')

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
    st.markdown("Ajustez l'angle et le nombre de Reynolds pour prédire le ratio L/D.")
    angle_input = st.slider("Angle d'attaque (°)", -10, 20, 5)
    reynolds_input = st.slider("Nombre de Reynolds", 100000, 1000000, 500000)
    input_data = model.prepare_input_data(angle_input, reynolds_input)

    if st.button("Prédire le Ratio L/D"):
        regime = 'transition'
        predicted_ratio = model.regime_models[regime]['model'].predict(input_data)
        st.write(f"**Ratio Lift-to-Drag prédit** : {predicted_ratio[0]:.2f}")

    st.markdown("""
    Les résultats montrent que certaines variables géométriques et l'angle ont une grande influence sur la performance (L/D), 
    tandis que le régime d'écoulement (lié au Reynolds) joue un rôle crucial dans la distribution de ces performances.
    """)
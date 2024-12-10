import streamlit as st
import os
import pandas as pd 
import copy

from src.preprocess import engineer_aerodynamic_features, create_stratified_sample

def app_layout(df, featured_df, stratified_df):
    from .content import introduction, data_management, dataviz, optimisation, prediction, conclusion

    st.set_page_config(
        page_title="SDA 2024 - Marius Ayrault",
        page_icon=":shark:",
        layout='wide',
        initial_sidebar_state="auto",
        menu_items={
            'About': "#Github Repository :\n\nhttps://github.com/Meriadoc-gitgit/DeepLearWing-viz"
        }
    )


    print("Chargement avec succès")

    page = st.sidebar.radio("Summary", ["Introduction", "Exploration de données", "Visualisation de données", "Optimisation", "Modélisation & Prédiction", "Conclusion"])

    if page == "Introduction":
        introduction()
    elif page == "Exploration de données":
        data_management(df, featured_df, stratified_df)
    elif page == "Visualisation de données":
        dataviz(stratified_df)
    elif page == "Optimisation":
        optimisation(stratified_df)
    elif page == "Modélisation & Prédiction":
        prediction(stratified_df)
    else:
        conclusion()

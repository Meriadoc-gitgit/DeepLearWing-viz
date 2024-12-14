import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from aquarel import load_theme

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from copy import deepcopy
import os

from src.wordcloud import *

# Configuration générale pour tous les plots
layout_config = dict(
    width=750,  # Largeur augmentée
    height=500,  # Hauteur augmentée
    font=dict(size=14)  # Taille de police augmentée
)

def dataviz(df_sample):
    """
    Data visualization page of the dashboard
    """
    # df_sample = pd.read_csv('data/echantillon_stratifie.csv')
    
    st.markdown("# 📈 Visualisation de données 📈")

    st.markdown("---")

    col1, col2 = st.columns([3, 2])


    print(os.getcwd())
    pdf_text = read_pdf_pdfplumber("data/541.pdf")
    sections = remove_stop_words(extract_sections(pdf_text))


    with col1:
        
        # Scatter plot angle vs cl
        df_tmp = deepcopy(df_sample[:2000])
        fig1 = px.scatter(df_tmp, 
                          x='angle', 
                          y='cl',
                          color='reynolds',
                          trendline='ols',  # Ajoute une régression linéaire
                          title='Relation entre angle d\'attaque et coefficient de portance',
                          labels={'angle': 'Angle d\'attaque (°)',
                                  'cl': 'Coefficient de portance',
                                  'reynolds': 'Nombre de Reynolds'},
                          opacity=0.6)

        fig1.update_layout(**layout_config)
        st.plotly_chart(fig1)

        # Matrice de corrélation
        corr_matrix = df_sample[['angle', 'reynolds', 'cl', 'cd', 'cm']].corr()

        # Histogramme du ratio L/D
        fig2 = px.histogram(df_sample, 
                        x='Lift-to-Drag Ratio',
                        nbins=100,
                        title='Distribution des ratios portance/traînée',
                        labels={'LD_ratio': 'Ratio L/D'},
                        marginal='box')
        fig2.update_layout(**layout_config)
        st.plotly_chart(fig2)

    

    with col2:
        fig3 = go.Figure(data=go.Heatmap(
                            z=corr_matrix,
                            x=corr_matrix.columns,
                            y=corr_matrix.columns,
                            colorscale='RdBu',
                            zmin=-1,
                            zmax=1,
                            text=corr_matrix.round(3),
                            texttemplate='%{text}'))

        fig3.update_layout(
            title='Matrice de corrélation',
            width=500,    # Taille spécifique pour la matrice de corrélation
            height=400,
            font=dict(size=14)
        )
        st.plotly_chart(fig3)



        # Scatter plot cd vs cl
        # Calculer le ratio portance/traînée (L/D)
        df_sample['LD_ratio'] = df_sample['cl'] / df_sample['cd']

        # Trouver les meilleurs profils basés sur le ratio L/D maximal
        best_profiles = df_sample.groupby('name')['LD_ratio'].max().sort_values(ascending=False).head(10)

        # Create the Plotly bar plot
        fig = px.bar(
            best_profiles,
            title='Top 10 des profils avec le meilleur ratio portance/traînée maximal',
            opacity=0.7
        )
        fig.update_xaxes(title_text='Nom du profil', title_font_size=14)
        fig.update_yaxes(title_text='Ratio portance/traînée maximal', title_font_size=14)

        # Customize layout for font sizes and title
        fig.update_layout(
            title_font_size=16,
            xaxis_title_font_size=14,
            yaxis_title_font_size=14,
            showlegend=False
        )

        # Display the Plotly figure in Streamlit
        st.plotly_chart(fig)

        
        





    # Wordcloud
    # Streamlit app layout

    all_text = " ".join(sections.values())

    # General Word Cloud
    st.header("General Word Cloud")
    wordcloud = WordCloud(width=800, height=600, background_color="black").generate(all_text)

    # Display the general word cloud
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(wordcloud, interpolation="bilinear")
    ax.axis("off")
    st.pyplot(fig)

    # Section-specific Word Clouds
    st.header("Section-Specific Word Clouds")
    cols = st.columns(2)  # Two columns for layout


    # Loop through sections and generate word clouds
    for i, (section_title, section_text) in enumerate(sections.items()):
        wordcloud_section = WordCloud(width=800, height=600, background_color="black").generate(section_text)

        # Display the word cloud in the appropriate column
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.imshow(wordcloud_section, interpolation="bilinear")
        ax.axis("off")
        ax.set_title(section_title, fontsize=16)
        
        # Alternate between columns
        cols[i % 2].pyplot(fig)
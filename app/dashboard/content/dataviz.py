import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from aquarel import load_theme

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

def dataviz(df_sample):
    """
    Data visualization page of the dashboard
    """
    # df_sample = pd.read_csv('data/echantillon_stratifie.csv')
    
    st.markdown("# 📈 Visualisation de données 📈")

    st.markdown("---")

    col1, col2 = st.columns([3, 2])

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



        # df_tmp = deepcopy(df_sample[:2000])
        # # Scatter plot cd vs cl
        # df_sample2 = df_tmp.sort_values('reynolds')  # Trier d'abord par Reynolds

        # # Créer les bins avec des labels numériques ordonnés
        # reynolds_bins = pd.qcut(df_sample2['reynolds'], 
        #                     q=4,  # 4 bins pour plus de clarté
        #                     labels=['1', '2', '3', '4'],  # Labels numériques ordonnés
        #                     duplicates='drop')

        # fig5 = px.scatter(df_sample2, 
        #                 x='cd', 
        #                 y='cl',
        #                 color='angle',
        #                 title='Polaire de traînée',
        #                 labels={'cd': 'Coefficient de traînée',
        #                         'cl': 'Coefficient de portance',
        #                         'angle': 'Angle d\'attaque (°)'},
        #                 animation_frame=reynolds_bins,
        #                 range_x=[0, df_sample['cd'].quantile(0.99)],
        #                 range_y=[df_sample['cl'].min(), df_sample['cl'].max()])

        # # Calculer les plages de Reynolds pour chaque bin
        # reynolds_ranges = df_sample2.groupby(reynolds_bins)['reynolds'].agg("mean")
        # frame_titles = [f"Re: {int(reynolds_ranges[row]):,}" 
        #                 for row in range(len(reynolds_ranges))]

        # fig5.update_layout(
        #     **layout_config,
        #     updatemenus=[{
        #         'buttons': [
        #             {
        #                 'args': [None, {'frame': {'duration': 1000, 'redraw': True},
        #                             'fromcurrent': True,
        #                             'transition': {'duration': 300}}],
        #                 'label': 'Play',
        #                 'method': 'animate'
        #             },
        #             {
        #                 'args': [[None], {'frame': {'duration': 0, 'redraw': False},
        #                                 'mode': 'immediate',
        #                                 'transition': {'duration': 0}}],
        #                 'label': 'Pause',
        #                 'method': 'animate'
        #             }
        #         ],
        #         'direction': 'left',
        #         'pad': {'r': 10, 't': 87},
        #         'showactive': False,
        #         'type': 'buttons',
        #         'x': 0.1,
        #         'xanchor': 'right',
        #         'y': 0,
        #         'yanchor': 'top'
        #     }],
        #     sliders=[{
        #         'currentvalue': {
        #             'font': {'size': 12},
        #             'prefix': 'Nombre de Reynolds: ',
        #             'visible': True,
        #             'xanchor': 'right'
        #         },
        #         'steps': [
        #             {
        #                 'args': [[str(i+1)], {'frame': {'duration': 0, 'redraw': True},
        #                                     'mode': 'immediate',
        #                                     'transition': {'duration': 0}}],
        #                 'label': title,
        #                 'method': 'animate'
        #             } for i, title in enumerate(frame_titles)
        #         ]
        #     }]
        # )
        # st.plotly_chart(fig5)

    

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

        
        
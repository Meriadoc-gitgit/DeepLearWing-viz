import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd 
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly import figure_factory as ff

from copy import deepcopy
import os
import numpy as np
from scipy.stats import gaussian_kde
from scipy import stats

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
    st.markdown("# 📈 Visualisation de données 📈")

    st.markdown("---")

    col1, col2 = st.columns([3, 2])


    print(os.getcwd())
    pdf_text = read_pdf_pdfplumber("data/541.pdf")
    sections = remove_stop_words(extract_sections(pdf_text))


    with col1:
        
        # Réduction du nombre de points par échantillonnage intelligent
        sample_size = st.slider('Nombre de points à afficher', 100, 1000, 500, step=100)
        
        # Création de bins personnalisés pour l'angle
        angle_bins = pd.cut(df_sample['angle'], bins=20)
        
        # Création de bins personnalisés pour reynolds (en utilisant les valeurs uniques)
        reynolds_unique = sorted(df_sample['reynolds'].unique())
        
        # Échantillonnage stratifié
        df_plot = pd.DataFrame()
        points_per_group = max(1, sample_size // (20 * len(reynolds_unique)))
        
        for angle_bin in angle_bins.unique():
            for reynolds_val in reynolds_unique:
                temp_df = df_sample[
                    (df_sample['angle'].between(angle_bin.left, angle_bin.right)) & 
                    (df_sample['reynolds'] == reynolds_val)
                ]
                if not temp_df.empty:
                    df_plot = pd.concat([
                        df_plot, 
                        temp_df.sample(min(len(temp_df), points_per_group))
                    ])

        # Options d'affichage
        col_options1, col_options2 = st.columns(2)
        with col_options1:
            show_trend = st.checkbox('Afficher la tendance', value=True)
        with col_options2:
            show_avg = st.checkbox('Afficher la moyenne mobile', value=False)

        # Création du scatter plot principal
        fig = px.scatter(df_plot, 
                        x='angle', 
                        y='cl',
                        color='reynolds',
                        color_continuous_scale='viridis',
                        trendline='ols' if show_trend else None,
                        labels={
                            'angle': 'Angle d\'attaque (°)',
                            'cl': 'Coefficient de portance (CL)',
                            'reynolds': 'Nombre de Reynolds'
                        })

        # Ajout de la moyenne mobile si demandée
        if show_avg:
            df_avg = df_plot.groupby('angle')['cl'].mean().reset_index()
            df_avg = df_avg.sort_values('angle')
            # Calcul de la moyenne mobile
            df_avg['cl_smooth'] = df_avg['cl'].rolling(window=5, center=True).mean()
            fig.add_trace(go.Scatter(
                x=df_avg['angle'],
                y=df_avg['cl_smooth'],
                mode='lines',
                name='Moyenne mobile',
                line=dict(color='red', width=3),
                showlegend=True
            ))

        # Mise à jour des marqueurs
        fig.update_traces(
            selector=dict(mode='markers'),
            marker=dict(
                size=8,
                opacity=0.6,
                line=dict(width=1, color='white')
            )
        )

        # Mise en page améliorée
        fig.update_layout(
            plot_bgcolor='white',
            title=dict(
                text='Relation entre l\'Angle d\'Attaque et le Coefficient de Portance',
                x=0.0,
                font=dict(size=20)
            ),
            height=500,
            xaxis=dict(
                showgrid=True,
                gridwidth=1,
                gridcolor='rgba(128,128,128,0.2)',
                zeroline=True,
                zerolinewidth=1,
                zerolinecolor='black',
                range=[df_plot['angle'].min()-1, df_plot['angle'].max()+1]
            ),
            yaxis=dict(
                showgrid=True,
                gridwidth=1,
                gridcolor='rgba(128,128,128,0.2)',
                zeroline=True,
                zerolinewidth=1,
                zerolinecolor='black',
                range=[df_plot['cl'].min()-0.1, df_plot['cl'].max()+0.1]
            ),
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="right",
                x=0.99,
                bgcolor="rgba(255, 255, 255, 0.8)"
            )
        )

        st.plotly_chart(fig, use_container_width=True)

        st.markdown("""
        #### Analyse de la Relation Angle-Portance

        L'analyse du graphique met en évidence plusieurs aspects significatifs :

        1. **Corrélation Linéaire**
        - Une relation linéaire positive est observée dans la plage -10° à +10°
        - Le coefficient directeur de la droite de régression indique une augmentation proportionnelle du CL avec l'angle

        2. **Effet Reynolds**
        - La dispersion des points diminue à Reynolds élevé (points foncés)
        - Les mesures à faible Reynolds présentent une plus grande variabilité

        3. **Limites de Linéarité**
        - Décrochage observé au-delà de 15°
        - Non-linéarité croissante aux angles extrêmes

        **Implications pour l'Analyse Aérodynamique** :
        La plage d'angle d'attaque optimale se situe dans la zone de linéarité, où le comportement est le plus prévisible et stable.
        """)

        # Distribution des ratios L/D
        fig = go.Figure()

        # Ajout de l'histogramme avec séparations
        fig.add_trace(go.Histogram(
            x=df_sample['Lift-to-Drag Ratio'],
            nbinsx=50,
            name='Distribution',
            marker=dict(
                color='rgb(55, 83, 109)',
                line=dict(
                    color='white',
                    width=1
                )
            ),
            opacity=0.8
        ))

        # Ajout de la courbe de densité
        hist_data = np.histogram(df_sample['Lift-to-Drag Ratio'], bins=50)
        hist_x = (hist_data[1][:-1] + hist_data[1][1:]) / 2
        density = gaussian_kde(df_sample['Lift-to-Drag Ratio'])
        
        fig.add_trace(go.Scatter(
            x=hist_x,
            y=density(hist_x) * len(df_sample['Lift-to-Drag Ratio']) * (hist_data[1][1] - hist_data[1][0]),
            mode='lines',
            name='Densité',
            line=dict(color='#E74C3C', width=2)
        ))

        # Mise en page
        fig.update_layout(
            title=dict(
                text='Distribution des Ratios Portance/Traînée',
                x=0.0,
                font=dict(size=20)
            ),
            xaxis_title='Ratio L/D',
            yaxis_title='Fréquence',
            height=500,
            showlegend=True,
            plot_bgcolor='white',
            xaxis=dict(
                showgrid=True,
                gridwidth=1,
                gridcolor='rgba(128,128,128,0.2)',
            ),
            yaxis=dict(
                showgrid=True,
                gridwidth=1,
                gridcolor='rgba(128,128,128,0.2)',
            ),
            legend=dict(
                yanchor="top",
                y=0.95,
                xanchor="right",
                x=0.95
            ),
            bargap=0.1
        )

        st.plotly_chart(fig, use_container_width=True)

        st.markdown("""
        #### Analyse de la Distribution L/D

        La distribution des ratios portance/traînée présente les caractéristiques suivantes :

        1. **Caractéristiques de la Distribution**
        - Distribution asymétrique positive
        - Mode principal dans l'intervalle 20-30
        - Queue étendue vers les valeurs élevées

        2. **Analyse Quantitative**
        - Concentration majoritaire des observations entre 0 et 50
        - Présence de performances exceptionnelles > 100
        - Distribution caractéristique des phénomènes aérodynamiques optimisés

        **Implications pour l'Analyse** :
        La distribution observée suggère un potentiel d'optimisation significatif, avec une plage de performance typique bien définie et des cas exceptionnels méritant une analyse approfondie.
        """)


    

    with col2:
        # Matrice de corrélation
        corr_matrix = df_sample[['angle', 'reynolds', 'cl', 'cd', 'cm']].corr()

        # Création de la heatmap
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            text=corr_matrix.round(3),
            texttemplate='%{text}',
            textfont={"size": 12},
            colorscale='RdBu',
            zmid=0,
            colorbar=dict(
                title='Coefficient de<br>corrélation',
                titleside='right',
                thickness=15,
                len=0.7
            )
        ))

        fig.update_layout(
            title=dict(
                text='Matrice de Corrélation',
                x=0.0,
                font=dict(size=20)
            ),
            width=600,
            height=500,
            xaxis=dict(title='', tickangle=0),
            yaxis=dict(title=''),
            plot_bgcolor='white'
        )

        st.plotly_chart(fig, use_container_width=True)

        st.markdown("""
        #### Analyse des Corrélations

        L'étude des corrélations entre paramètres révèle plusieurs relations significatives :

        1. **Corrélation Angle-CL (0.919)**
        - Forte corrélation positive
        - Confirme la dépendance directe entre l'angle d'attaque et la génération de portance

        2. **Influence du Nombre de Reynolds**
        - Corrélation négative modérée avec CD (-0.311)
        - Corrélation faible avec CL (0.084)
        - Indique une réduction de la traînée à Reynolds élevé

        3. **Coefficient de Moment (CM)**
        - Corrélations faibles avec tous les paramètres
        - Suggère une relative indépendance des effets de moment

        **Implications pour la Modélisation** :
        Ces corrélations permettent d'identifier les paramètres clés pour l'optimisation des performances aérodynamiques.
        """)

        
        # Top 10 des profils
        top_10_profiles = df_sample.groupby('name')['Lift-to-Drag Ratio'].max().nlargest(10)

        fig = go.Figure(data=[
            go.Bar(
                x=top_10_profiles.index,
                y=top_10_profiles.values,
                text=top_10_profiles.values.round(1),
                textposition='auto',
                marker=dict(
                    color='rgb(55, 83, 109)',
                    line=dict(
                        color='white',
                        width=1
                    )
                ),
                opacity=0.8
            )
        ])

        fig.update_layout(
            title=dict(
                text='Classement des Profils par Performance Aérodynamique Maximale',
                x=0,
                font=dict(size=20)
            ),
            xaxis_title='Désignation du Profil',
            yaxis_title='Ratio L/D Maximal',
            height=500,
            plot_bgcolor='white',
            xaxis=dict(
                tickangle=45,
                showgrid=True,
                gridwidth=1,
                gridcolor='rgba(128,128,128,0.2)',
            ),
            yaxis=dict(
                showgrid=True,
                gridwidth=1,
                gridcolor='rgba(128,128,128,0.2)',
                zeroline=True,
                zerolinewidth=1,
                zerolinecolor='black'
            )
        )

        st.plotly_chart(fig, use_container_width=True)

        st.markdown(f"""
        #### Analyse des Performances Maximales

        L'analyse des dix profils les plus performants révèle une hiérarchie distincte :

        1. **Stratification des Performances**
        - Performance maximale de {top_10_profiles.values[0]:.1f} pour le profil {top_10_profiles.index[0]}
        - Écart significatif de {(top_10_profiles.values[0] - top_10_profiles.values[-1]):.1f} entre premier et dixième
        - Formation de trois strates de performance distinctes

        2. **Distribution des Performances**
        - Groupe supérieur : > {top_10_profiles.values[2]:.0f}
        - Groupe intermédiaire : {top_10_profiles.values[6]:.0f}-{top_10_profiles.values[3]:.0f}
        - Groupe inférieur : < {top_10_profiles.values[6]:.0f}

        **Note Méthodologique** :
        Ces performances représentent des maxima théoriques obtenus dans des conditions optimales. Les performances opérationnelles attendues seront typiquement inférieures de 40-50%.
        """)

    

    st.markdown("### 📊 Performance des Profils selon le Régime d'Écoulement")

    # Définition des seuils de Reynolds
    reynolds_thresholds = df_sample['reynolds'].quantile([0.33, 0.66])
    
    # Calcul des performances moyennes par régime
    reynolds_performance = df_sample.groupby('name').apply(
        lambda x: pd.Series({
            f'Bas Reynolds\n(<{reynolds_thresholds[0.33]:.0f})': 
                x[x['reynolds'] < reynolds_thresholds[0.33]]['Lift-to-Drag Ratio'].mean(),
            f'Reynolds Moyen\n({reynolds_thresholds[0.33]:.0f}-{reynolds_thresholds[0.66]:.0f})': 
                x[(x['reynolds'] >= reynolds_thresholds[0.33]) & 
                (x['reynolds'] < reynolds_thresholds[0.66])]['Lift-to-Drag Ratio'].mean(),
            f'Haut Reynolds\n(>{reynolds_thresholds[0.66]:.0f})': 
                x[x['reynolds'] >= reynolds_thresholds[0.66]]['Lift-to-Drag Ratio'].mean()
        })
    ).round(2)

    # Calcul du score de polyvalence (écart-type entre régimes)
    versatility_score = reynolds_performance.std(axis=1)
    
    # Tri des profils par performance moyenne
    reynolds_performance['Moyenne'] = reynolds_performance.mean(axis=1)
    top_profiles = reynolds_performance.nlargest(10, 'Moyenne').index

    # Préparation des données pour le graphique
    plot_data = reynolds_performance.loc[top_profiles].drop('Moyenne', axis=1)

    # Création du graphique
    fig = go.Figure()

    # Ajout des barres pour chaque régime
    for col in plot_data.columns:
        fig.add_trace(go.Bar(
            name=col,
            x=plot_data.index,
            y=plot_data[col],
            text=plot_data[col].round(1),
        ))

    # Mise en page
    fig.update_layout(
        title=dict(
            text="Top 10 des Profils : Performance selon le Régime de Reynolds",
            x=0.5,
            xanchor='center'
        ),
        xaxis_title="Profil",
        yaxis_title="Ratio L/D Moyen",
        barmode='group',
        height=500,
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=1.15,  # Déplacé plus haut
            xanchor="center",
            x=0.5,   # Centré horizontalement
            orientation="h"  # Orientation horizontale
        ),
        plot_bgcolor='white',
        xaxis=dict(
            tickangle=45,
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(128,128,128,0.2)',
        ),
        yaxis=dict(
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(128,128,128,0.2)',
            zeroline=False,
        )
    )

    st.plotly_chart(fig, use_container_width=True)

    # Analyse des résultats
    best_overall = plot_data.mean(axis=1).idxmax()
    most_versatile = versatility_score.loc[top_profiles].idxmin()
    best_high_re = plot_data.iloc[:, 2].idxmax()

    st.markdown(f"""
    #### 💡 Analyse des Performances

    Ce graphique montre comment les meilleurs profils se comportent dans différents régimes d'écoulement :
    
    - **{best_overall}** montre la meilleure performance globale
    - **{most_versatile}** est le plus polyvalent (performances stables à travers les régimes)
    - **{best_high_re}** excelle particulièrement à haut Reynolds

    #### 🎯 Implications pour le Choix du Profil
    - Pour des applications à Reynolds variable, privilégier les profils polyvalents
    - Pour des conditions spécifiques, choisir un profil optimisé pour ce régime
    - Les performances à haut Reynolds sont généralement meilleures
    """)







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




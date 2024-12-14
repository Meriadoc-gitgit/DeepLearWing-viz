import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from aquarel import load_theme

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from copy import deepcopy

from src.plot_airfoil_performance import plot_airfoil_performance

# Configuration générale pour tous les plots
layout_config = dict(
    width=750,  # Largeur augmentée
    height=500,  # Hauteur augmentée
    font=dict(size=14)  # Taille de police augmentée
)
def optimisation(df_sample):
    """
    Optimisation page of the dashboard
    """
    # df_sample = pd.read_csv('data/echantillon_stratifie.csv')
    
    st.markdown("# 🛩️ Optimisation des Profils d'Aile : La Quête du Meilleur Ratio Portance/Traînée 🛩️")

    st.markdown("---")
    col1, col2 = st.columns([2, 2])


    with col1:
        

        # Visualisation de la distribution globale des performances
        fig_intro = px.histogram(df_sample, 
                                x='Lift-to-Drag Ratio',
                                title='La Distribution des Performances Aérodynamiques',
                                color_discrete_sequence=['skyblue'],
                                marginal='box')
        fig_intro.update_layout(**layout_config)
        
        st.plotly_chart(fig_intro)

        st.markdown("""
        Dans le monde de l'aéronautique, un défi constant persiste : comment faire voler plus efficacement ? Le ratio portance/traînée (L/D) est la clé de cette efficacité. C'est le "rendement" de l'aile, déterminant la consommation de carburant, l'autonomie et les performances globales.""")

        st.markdown("""###### Les Champions de l'Efficacité""")

        
        # Identification des meilleurs profils
        top_performers = df_sample.groupby('name').agg({
            'Lift-to-Drag Ratio': ['max', 'mean', 'std']
        }).round(2)

        top_10 = top_performers.sort_values(('Lift-to-Drag Ratio', 'max'), ascending=False).head(10)

        st.write(top_10)


        st.markdown("""
        ##### Question : Les meilleures profils sont-ils aussi les plus stables ?""")
        # 5. Performance vs Stabilité
        # Recalculons les statistiques de manière plus directe
        stability_data = df_sample.groupby('name').agg({
            'Lift-to-Drag Ratio': ['max', 'std']
        }).reset_index()

        # Renommons les colonnes pour plus de clarté
        stability_data.columns = ['name', 'max_performance', 'stability']

        # Création du graphique
        fig_stability = px.scatter(stability_data,
                                x='max_performance',
                                y='stability',
                                hover_data=['name'],
                                title='Compromis Performance-Stabilité',
                                labels={'max_performance': 'Performance maximale (L/D)',
                                        'stability': 'Variabilité (écart-type)',
                                        'name': 'Profil'})

        # Ajout d'annotations pour les cas extrêmes
        top_performers = stability_data.nlargest(3, 'max_performance')
        most_stable = stability_data.nsmallest(3, 'stability')

        for _, row in top_performers.iterrows():
            fig_stability.add_annotation(
                x=row['max_performance'],
                y=row['stability'],
                text=row['name'],
                showarrow=True,
                arrowhead=1
            )

        for _, row in most_stable.iterrows():
            fig_stability.add_annotation(
                x=row['max_performance'],
                y=row['stability'],
                text=row['name'],
                showarrow=True,
                arrowhead=1
            )

        fig_stability.update_layout(
            **layout_config,
            showlegend=True,
            xaxis_title="Performance maximale (L/D)",
            yaxis_title="Variabilité (écart-type)",
            hovermode='closest'
        )

        
        st.plotly_chart(fig_stability)

        st.markdown("""###### Profils avec les meilleures performances maximales""")
        st.write(top_performers[['name', 'max_performance', 'stability']].round(2))

        st.markdown("###### Profils les plus stables:")
        st.write(most_stable[['name', 'max_performance', 'stability']].round(2))

        

    with col2:

        # Visualisation de l'interaction entre ces facteurs
        fig_factors = px.scatter(df_sample,
                                x='angle',
                                y='Lift-to-Drag Ratio',
                                color='Log Reynolds',
                                title='L\'Impact de l\'Angle d\'Attaque et du Reynolds',
                                labels={'Lift-to-Drag Ratio': 'Ratio L/D',
                                    'angle': 'Angle d\'attaque (°)',
                                    'Log Reynolds': 'Log(Reynolds)'},
                                opacity=0.6)
        fig_factors.update_layout(**layout_config)
        st.plotly_chart(fig_factors)

        st.markdown("""
        Deux paramètres critiques influencent ce ratio :
        - L'angle d'attaque : l'inclinaison de l'aile par rapport à l'écoulement
        - Le nombre de Reynolds : caractérisant le régime d'écoulement""")


        # Cartographie des conditions optimales
        optimal_conditions = df_sample.loc[df_sample.groupby('name')['Lift-to-Drag Ratio'].idxmax()]
        fig_optimal = px.scatter(optimal_conditions,
                                x='angle',
                                y='reynolds',
                                color='Lift-to-Drag Ratio',
                                title='La "Zone d\'Or" des Performances',
                                hover_data=['name'])
        fig_optimal.update_layout(**layout_config)
        st.plotly_chart(fig_optimal)



        # Identification des profils avec bon compromis performance/stabilité
        # Normalisation des métriques pour créer un score composite
        stability_data['performance_norm'] = (stability_data['max_performance'] - stability_data['max_performance'].mean()) / stability_data['max_performance'].std()
        stability_data['stability_norm'] = -(stability_data['stability'] - stability_data['stability'].mean()) / stability_data['stability'].std()
        stability_data['score_composite'] = stability_data['performance_norm'] + stability_data['stability_norm']

        # Top 10 des profils équilibrés
        best_balanced = stability_data.nlargest(10, 'score_composite')
        st.markdown("###### Top 10 des profils avec le meilleur compromis performance/stabilité:")
        st.write(best_balanced[['name', 'max_performance', 'stability', 'score_composite']].round(3))



        # 1. D'abord, identifions les meilleurs profils
        # Calculer la moyenne du ratio portance/traînée pour chaque profil
        profile_performance = df_sample.groupby('name')['Lift-to-Drag Ratio'].mean().sort_values(ascending=False)

        # Sélectionner les 5 meilleurs profils
        top_profiles = profile_performance.head(5).index.tolist()

        st.markdown("###### Les 5 meilleurs profils sont ")
        for i, (profile, score) in enumerate(profile_performance.head(5).items(), 1):
            strng = f"{i}. {profile}: {score:.2f}"
            st.markdown(strng)


        # 2. Maintenant, analysons les conditions optimales pour ces profils
        if not df_sample.empty and len(top_profiles) > 0:
            # Filtrer les données pour les profils optimaux
            optimal_data = df_sample[df_sample['name'].isin(top_profiles)].copy()
            
            if not optimal_data.empty:
                # Analyser chaque profil
                results = []
                for name in top_profiles:
                    profile_data = optimal_data[optimal_data['name'] == name]
                    max_idx = profile_data['Lift-to-Drag Ratio'].idxmax()
                    
                    results.append({
                        'name': name,
                        'optimal_angle': profile_data.loc[max_idx, 'angle'],
                        'optimal_reynolds': profile_data.loc[max_idx, 'reynolds'],
                        'max_LD': profile_data['Lift-to-Drag Ratio'].max(),
                        'angle_range': profile_data['angle'].max() - profile_data['angle'].min(),
                        'reynolds_range': profile_data['reynolds'].max() - profile_data['reynolds'].min()
                    })
                
                # Créer le DataFrame final
                optimal_conditions = pd.DataFrame(results).set_index('name')
                
                st.markdown("###### Conditions optimales pour les meilleurs profils")
                st.write(optimal_conditions.round(2))
            else:
                print("Aucune donnée trouvée pour les profils sélectionnés dans df_sample")
        else:
            print("Données manquantes : vérifiez df_sample et top_profiles")



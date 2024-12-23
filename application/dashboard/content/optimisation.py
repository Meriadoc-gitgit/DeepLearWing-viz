# =============================================================================
# 1. IMPORTS ET CONFIGURATION
# =============================================================================

# 1.1 Imports standards et scientifiques
import numpy as np
import pandas as pd
from scipy import stats
from scipy.interpolate import griddata

# 1.2 Interface utilisateur
import streamlit as st

# 1.3 Visualisation
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# 1.4 Utilitaires
from copy import deepcopy

# 1.5 Modules locaux
from src.plot_airfoil_performance import plot_airfoil_performance

# =============================================================================
# 2. CONFIGURATION GLOBALE
# =============================================================================

LAYOUT_CONFIG = dict(
    width=750,
    height=500,
    font=dict(size=14)
)

# Configuration des graphiques
PLOT_COLORS = {
    'surface_scale': [
        [0, '#053061'],      # Bleu foncé
        [0.2, '#2166ac'],    # Bleu
        [0.4, '#4393c3'],    # Bleu clair
        [0.6, '#92c5de'],    # Très bleu clair
        [0.8, '#f4a582'],    # Orange clair
        [1, '#67001f']       # Rouge foncé
    ],
    'scatter_scale': [
        [0, '#e5f5e0'],      # Vert très clair
        [0.5, '#74c476'],    # Vert moyen
        [1, '#238b45']       # Vert foncé
    ]
}

# Paramètres d'analyse
ANALYSIS_PARAMS = {
    'performance_weight': 0.6,
    'stability_weight': 0.4,
    'top_performers_threshold': 0.97,
    'optimal_angle_threshold': 0.8
}

# =============================================================================
# 3. FONCTION PRINCIPALE
# =============================================================================

def _display_performance_overview(df_sample):
    """
    Affiche la vue globale des performances et l'optimisation
    
    Args:
        df_sample (pd.DataFrame): DataFrame source
    """
    st.markdown("""
    # 🎯 Vue Globale des Performances
    
    Cette section présente une vue d'ensemble des performances aérodynamiques et des zones optimales d'utilisation.
    """)

    # Surface 3D des performances
    st.markdown("""
    ### 📊 Cartographie 3D des Performances
    
    La surface 3D ci-dessous représente l'espace complet des performances aérodynamiques :
    - **Axe X** : Angle d'attaque (inclinaison de l'aile)
    - **Axe Y** : Nombre de Reynolds (régime d'écoulement)
    - **Axe Z** : Ratio portance/traînée (L/D)
    """)
    
    # Création de la surface 3D
    angle_range = np.linspace(df_sample['angle'].min(), df_sample['angle'].max(), 50)
    reynolds_range = np.linspace(df_sample['reynolds'].min(), df_sample['reynolds'].max(), 50)
    X, Y = np.meshgrid(angle_range, reynolds_range)
    
    Z = griddata(
        (df_sample['angle'], df_sample['reynolds']), 
        df_sample['Lift-to-Drag Ratio'],
        (X, Y),
        method='cubic',
        fill_value=np.nan
    )

    fig_3d = go.Figure()
    
    # Surface principale
    surface = go.Surface(
        x=X, y=Y, z=Z,
        colorscale=PLOT_COLORS['surface_scale'],
        colorbar=dict(
            title=dict(text='Ratio L/D', font=dict(size=14)),
            tickfont=dict(size=12),
            len=0.8
        ),
        name='Surface L/D',
        contours=dict(
            z=dict(show=True, usecolormap=True, highlightcolor="white", project=dict(z=True))
        ),
        lighting=dict(
            ambient=0.6,
            diffuse=0.5,
            fresnel=0.2,
            specular=0.1,
            roughness=0.5
        )
    )
    fig_3d.add_trace(surface)

    # Points optimaux
    best_performers = df_sample[df_sample['Lift-to-Drag Ratio'] > df_sample['Lift-to-Drag Ratio'].quantile(ANALYSIS_PARAMS['top_performers_threshold'])]
    optimal_points = go.Scatter3d(
        x=best_performers['angle'],
        y=best_performers['reynolds'],
        z=best_performers['Lift-to-Drag Ratio'],
        mode='markers+text',
        marker=dict(
            size=4,
            color='#FFD700',
            symbol='diamond',
            line=dict(color='#B8860B', width=1)
        ),
        text=["Zone Optimale" if i == len(best_performers)//2 else "" for i in range(len(best_performers))],
        textposition="top center",
        name='Points Optimaux'
    )
    fig_3d.add_trace(optimal_points)

    fig_3d.update_layout(
        scene=dict(
            xaxis_title=dict(text='Angle d\'attaque (°)', font=dict(size=12)),
            yaxis_title=dict(text='Nombre de Reynolds', font=dict(size=12)),
            zaxis_title=dict(text='Ratio L/D', font=dict(size=12)),
            camera=dict(
                up=dict(x=0, y=0, z=1),
                center=dict(x=0, y=0, z=-0.1),
                eye=dict(x=1.8, y=1.8, z=1.5)
            ),
            aspectratio=dict(x=1.2, y=1, z=0.7)
        ),
        width=800,
        height=600,
        margin=dict(t=0, b=0, l=0, r=0),
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor="rgba(255, 255, 255, 0.8)"
        )
    )
    st.plotly_chart(fig_3d, use_container_width=True)

    st.markdown("""
    #### 🎯 Interprétation de la Surface 3D
    
    La cartographie 3D révèle plusieurs zones caractéristiques :

    - **Zone de Performance Maximale** (Rouge) :
        - Pic de performance à 3° d'angle d'attaque
        - Reynolds optimal autour de 1M (10⁶)
        - Ratio L/D atteignant 55.1 pour les meilleurs profils
    
    - **Zone de Transition** (Bleu clair à Rouge) :
        - Plage utilisable entre 1° et 5°
        - Performance décroissant graduellement en s'éloignant de l'optimal
        - Bonne stabilité dans cette région
    
    - **Zone Critique** (Bleu foncé) :
        - Angles > 10° : décrochage aérodynamique
        - Angles < -5° : portance négative
        - Performances fortement dégradées
    
    - **Points Remarquables** (Dorés) :
        - Configurations exceptionnelles (top 3%)
        - Concentrés autour de 3° ± 1°
        - Principalement observés sur les profils e63 et ua79sfm
    """)

    # Préparation des données optimales
    optimal_conditions = df_sample.loc[df_sample.groupby('name')['Lift-to-Drag Ratio'].idxmax()].copy()
    top_performers = optimal_conditions.nlargest(5, 'Lift-to-Drag Ratio')
    other_profiles = optimal_conditions[~optimal_conditions['name'].isin(top_performers['name'])]

    
    # Analyse des paramètres de vol
    st.markdown("""
    ### 📈 Influence des Paramètres de Vol
    
    Visualisation de l'effet de l'angle d'attaque et du nombre de Reynolds :
    """)

    df_sample['Log Reynolds'] = np.log10(df_sample['reynolds'])
    
    fig_params = px.scatter(df_sample,
                         x='angle',
                         y='Lift-to-Drag Ratio',
                         color='Log Reynolds',
                         labels={'Lift-to-Drag Ratio': 'Ratio L/D',
                                'angle': 'Angle d\'attaque (°)',
                                'Log Reynolds': 'Log(Reynolds)'},
                         opacity=0.6,
                         color_continuous_scale=PLOT_COLORS['scatter_scale'])
    
    fig_params.update_traces(marker=dict(size=8))
    fig_params.update_layout(
        height=400,
        margin=dict(t=0, b=0, l=0, r=0),
        coloraxis_colorbar=dict(
            title='Log(Reynolds)',
            titleside='right',
            thickness=15,
            len=0.7
        )
    )
    
    st.plotly_chart(fig_params, use_container_width=True)

    st.markdown("""
    ##### 🔑 Points Clés
    - **Zone Optimale** : Angle d'attaque entre 2° et 4°
    - **Reynolds Optimal** : Autour de 1M (log₁₀ = 6)
    - **Comportement** :
      - Performance maximale dans la zone dorée
      - Dégradation rapide hors de la plage optimale
      - Meilleure stabilité à Reynolds élevé
    """)

    # Analyse de la zone optimale
    st.markdown("""
    ### 🎯 Zone Optimale d'Utilisation
    
    Le graphique ci-dessous montre l'évolution du ratio L/D en fonction de l'angle d'attaque :
    """)
    
    # Calcul des statistiques par angle
    angle_stats = df_sample.groupby('angle')['Lift-to-Drag Ratio'].agg(['mean', 'std']).reset_index()
    optimal_mask = angle_stats['mean'] > angle_stats['mean'].max() * ANALYSIS_PARAMS['optimal_angle_threshold']
    optimal_angles = angle_stats[optimal_mask]['angle']

    fig = go.Figure()

    # Courbe principale
    fig.add_trace(go.Scatter(
        x=angle_stats['angle'],
        y=angle_stats['mean'],
        mode='lines',
        name='L/D Moyen',
        line=dict(color='blue', width=2)
    ))

    # Zone d'incertitude
    fig.add_trace(go.Scatter(
        x=angle_stats['angle'],
        y=angle_stats['mean'] + angle_stats['std'],
        mode='lines',
        line=dict(width=0),
        showlegend=False
    ))

    fig.add_trace(go.Scatter(
        x=angle_stats['angle'],
        y=angle_stats['mean'] - angle_stats['std'],
        mode='lines',
        fill='tonexty',
        fillcolor='rgba(0,0,255,0.1)',
        line=dict(width=0),
        name='Plage de Variation'
    ))

    # Zone optimale
    fig.add_vrect(
        x0=optimal_angles.min(),
        x1=optimal_angles.max(),
        fillcolor="rgba(0,255,0,0.1)",
        layer="below",
        line_width=0,
        annotation_text="Zone Optimale",
        annotation_position="top left"
    )

    fig.update_layout(
        xaxis_title="Angle d'Attaque (degrés)",
        yaxis_title="Ratio L/D",
        height=400,
        margin=dict(t=0, b=0, l=0, r=0)
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown(f"""
    #### 📊 Points Clés
    - **Zone Optimale** : Entre {optimal_angles.min():.1f}° et {optimal_angles.max():.1f}°
    - **Performance Maximale** : L/D ratio de {angle_stats['mean'].max():.1f}
    - **Stabilité** : Meilleure stabilité dans la zone optimale
    """)

    # Nouvelle analyse focalisée sur le compromis performance-stabilité des profils
    st.markdown("""
    ### 📊 Compromis Performance-Stabilité
    
    Cette section présente une analyse détaillée du compromis entre performance et stabilité des profils.
    """)

    # Calcul de la stabilité (variation de L/D)
    stability = df_sample.groupby('name')['Lift-to-Drag Ratio'].std()
    performance = df_sample.groupby('name')['Lift-to-Drag Ratio'].max()
    
    # Création du dataframe pour l'analyse
    analysis_df = pd.DataFrame({
        'Stabilité': stability,  # On garde les valeurs positives
        'Performance': performance,
        'Profil': performance.index
    }).reset_index(drop=True)
    
    # Identification des profils remarquables
    best_performers = analysis_df.nlargest(5, 'Performance')
    
    # Calcul du meilleur compromis en favorisant haute performance et faible écart-type
    analysis_df['Score'] = (analysis_df['Performance'] ** 2) / analysis_df['Stabilité']
    
    # Sélection des meilleurs profils selon le score
    top_profiles = analysis_df.nlargest(5, 'Score')
    
    # Définition de la zone basée sur les profils remarquables (ua79sfm, e61, e63, goe804)
    target_profiles = ['ua79sfm', 'e61', 'e63', 'goe804']
    target_data = analysis_df[analysis_df['Profil'].isin(target_profiles)]
    
    # Calcul des limites de la zone
    perf_min = target_data['Performance'].min() * 0.95
    perf_max = target_data['Performance'].max() * 1.05
    stab_min = target_data['Stabilité'].min() * 0.95
    stab_max = target_data['Stabilité'].max() * 1.05
    
    # Création du graphique
    fig = go.Figure()
    
    # Scatter plot principal
    fig.add_trace(
        go.Scatter(
            x=analysis_df['Performance'],
            y=analysis_df['Stabilité'],
            mode='markers',
            marker=dict(
                size=10,
                color=analysis_df['Performance'],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(
                    title='Performance (L/D max)',
                    thickness=15
                )
            ),
            text=analysis_df['Profil'],
            hovertemplate="<b>%{text}</b><br>" +
                         "Performance: %{x:.1f}<br>" +
                         "Stabilité: %{y:.1f}<br>" +
                         "<extra></extra>",
            name='Tous les Profils'
        )
    )

    # Ajout de la zone de meilleur compromis
    fig.add_shape(
        type="rect",
        x0=perf_min,
        y0=stab_min,
        x1=perf_max,
        y1=stab_max,
        fillcolor="rgba(0,255,0,0.1)",
        line=dict(color="green", width=2, dash="dash"),
        name="Zone de compromis optimal"
    )
    
    # Ajouter une annotation pour expliquer la zone
    fig.add_annotation(
        x=(perf_min + perf_max)/2,
        y=(stab_min + stab_max)/2,
        text="Zone de meilleur compromis<br>Performance-Stabilité",
        showarrow=False,
        font=dict(size=10, color="green"),
        bgcolor="rgba(255,255,255,0.8)"
    )

    # Ajout des annotations pour les profils remarquables
    for _, row in best_performers.iterrows():
        fig.add_annotation(
            x=row['Performance'],
            y=row['Stabilité'],
            text=row['Profil'],
            showarrow=True,
            arrowhead=2,
            arrowsize=1,
            arrowwidth=2,
            ax=40,
            ay=-40
        )
    
    # Ajout d'une annotation spéciale pour le meilleur compromis
    best_compromise = analysis_df.loc[analysis_df['Score'].idxmax()]
    fig.add_annotation(
        x=best_compromise['Performance'],
        y=best_compromise['Stabilité'],
        text=f"{best_compromise['Profil']} (Meilleur compromis)",
        showarrow=True,
        arrowhead=2,
        arrowsize=1,
        arrowwidth=2,
        ax=60,
        ay=40,
        font=dict(size=12, color='red'),
        arrowcolor='red'
    )
    
    # Mise en page
    fig.update_layout(
        title='Compromis Performance-Stabilité des Profils',
        xaxis_title='Performance (L/D maximum)',
        yaxis_title='Stabilité (écart-type)',
        height=600,
        showlegend=False,
        plot_bgcolor='white'
    )
    
    # Ajout des lignes de référence
    fig.add_hline(y=analysis_df['Stabilité'].median(), 
                 line_dash="dash", line_color="gray",
                 annotation_text="Stabilité médiane")
    fig.add_vline(x=analysis_df['Performance'].median(),
                 line_dash="dash", line_color="gray",
                 annotation_text="Performance médiane")
    
    # Affichage du graphique
    st.plotly_chart(fig, use_container_width=True)
    
    # Analyse des résultats
    st.markdown(f"""
    # 📊 Analyse Performance-Stabilité
    Cette visualisation révèle les caractéristiques clés des profils :

    ## 🏆 Profils Remarquables
    - **Meilleure Performance** : {best_performers.iloc[0]['Profil']} (L/D = {best_performers.iloc[0]['Performance']:.1f})
    - **Meilleur Compromis** : {best_compromise['Profil']} (L/D = {best_compromise['Performance']:.1f}, Écart-type = {best_compromise['Stabilité']:.2f})

    ## 🎯 Distribution
    - **Performance Médiane** : L/D = {analysis_df['Performance'].median():.1f}
    - **Stabilité Médiane** : Écart-type = {analysis_df['Stabilité'].median():.2f}

    ## 💡 Observations
    - Les profils les plus performants tendent à avoir une stabilité plus faible
    - Les profils très stables ont généralement des performances plus modestes
    - Le profil {best_compromise['Profil']} se distingue par son excellent rapport performance/stabilité
    """)

def _display_detailed_profiles(df_sample, stability_data):
    """
    Affiche l'analyse détaillée des profils
    
    Args:
        df_sample (pd.DataFrame): DataFrame source
        stability_data (pd.DataFrame): DataFrame avec les métriques de stabilité
    """
    st.markdown("""
    # 🔍 Analyse Détaillée des Profils
    
    Cette section présente une analyse approfondie des caractéristiques de chaque profil.
    """)

    # Analyse du compromis performance-stabilité
    st.markdown("""
    ### 📊 Compromis Performance-Stabilité
    
    Ce graphique met en évidence la relation entre performance maximale et stabilité :
    """)
    
    fig_stability = px.scatter(stability_data,
                           x='max_performance',
                           y='stability',
                           hover_data=['name', 'mean_performance'],
                           title='Performance vs Stabilité',
                           labels={'max_performance': 'Performance maximale (L/D)',
                                 'stability': 'Variabilité (écart-type)',
                                 'name': 'Profil'})
    
    # Points des cas extrêmes
    top_5 = stability_data.nlargest(5, 'max_performance')
    
    for _, row in top_5.head(3).iterrows():
        fig_stability.add_annotation(
            x=row['max_performance'],
            y=row['stability'],
            text=row['name'],
            showarrow=True,
            arrowhead=1
        )

    fig_stability.update_traces(marker=dict(color='#90EE90'))
    fig_stability.update_layout(
        height=400,
        margin=dict(t=30, b=0, l=0, r=0),
        showlegend=False
    )
    
    st.plotly_chart(fig_stability, use_container_width=True)

    # Profils les plus performants
    st.markdown("""
    ### 🏆 Top Performers
    Les profils ayant démontré les meilleures performances maximales :
    """)
    
    top_5 = stability_data.nlargest(5, 'max_performance')
    st.write(top_5[['name', 'max_performance', 'mean_performance', 'stability']].round(2))

    # Profils équilibrés
    st.markdown("""
    ### 🎖️ Profils Équilibrés
    Ces profils offrent le meilleur compromis entre performance et stabilité :
    """)
    
    best_balanced = stability_data.nlargest(5, 'score_composite')
    st.write(best_balanced[['name', 'max_performance', 'stability', 'score_composite']].round(3))

    # Analyse comparative
    st.markdown("""
    ### 📊 Analyse Comparative
    """)
    
    # Sélecteur de profils
    selected_profiles = st.multiselect(
        "Sélectionnez les profils à comparer",
        sorted(df_sample['name'].unique()),
        default=sorted(df_sample['name'].unique())[:3]
    )

    if selected_profiles:
        param_options = {
            'Ratio L/D': 'Lift-to-Drag Ratio',
            'Coefficient de portance (CL)': 'cl',
            'Coefficient de traînée (CD)': 'cd',
            'Coefficient de moment (CM)': 'cm'
        }
        selected_param = st.selectbox(
            "Paramètre à analyser",
            list(param_options.keys())
        )

        fig_compare = go.Figure()

        # Définition d'une palette de couleurs harmonieuses
        colors = [
            '#4B89DC',  # Bleu clair élégant
            '#37BC9B',  # Vert menthe
            '#967ADC',  # Violet doux
            '#F6BB42',  # Jaune doré
            '#E9573F',  # Corail
            '#3BAFDA',  # Bleu turquoise
            '#8CC152',  # Vert printemps
            '#D770AD',  # Rose poudré
            '#5D9CEC',  # Bleu ciel
            '#48CFAD'   # Vert d'eau
        ]
        
        for i, profile in enumerate(selected_profiles):
            profile_data = df_sample[df_sample['name'] == profile]
            angle_stats = profile_data.groupby('angle')[param_options[selected_param]].agg(['mean', 'std']).reset_index()
            
            # Courbe principale avec une couleur unique pour chaque profil
            fig_compare.add_trace(go.Scatter(
                x=angle_stats['angle'],
                y=angle_stats['mean'],
                mode='lines',
                name=profile,             
                line=dict(width=2.5, color=colors[i % len(colors)])  # Légèrement plus épais pour une meilleure visibilité
            ))

        fig_compare.update_layout(
            xaxis_title="Angle d'Attaque (degrés)",
            yaxis_title=selected_param,
            height=400,
            margin=dict(t=0, b=0, l=0, r=0),
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="right",
                x=0.99,
                bgcolor="rgba(255, 255, 255, 0.8)"
            )
        )

        st.plotly_chart(fig_compare, use_container_width=True)

        # Analyse automatique pour le ratio L/D
        if selected_param == 'Ratio L/D':
            best_profile = None
            max_ld = -float('inf')
            best_angle = None
            
            for profile in selected_profiles:
                profile_data = df_sample[df_sample['name'] == profile]
                max_profile_ld = profile_data['Lift-to-Drag Ratio'].max()
                if max_profile_ld > max_ld:
                    max_ld = max_profile_ld
                    best_profile = profile
                    best_angle = profile_data.loc[profile_data['Lift-to-Drag Ratio'].idxmax(), 'angle']

            st.markdown(f"""
            #### 💡 Analyse des Performances
            
            Parmi les profils sélectionnés :
            - **{best_profile}** montre la meilleure performance
            - Ratio L/D maximal : {max_ld:.1f}
            - Angle optimal : {best_angle:.1f}°
            """)

def _display_header():
    """Affiche l'en-tête de la page avec introduction"""
    st.markdown("# 🛩️ Optimisation des Profils d'Aile")
    
    st.markdown("""
    ## Introduction
    
    Cette interface vous permet d'explorer et d'optimiser les performances aérodynamiques des différents profils à travers plusieurs aspects clés :
    
    - **Performance Globale** : Visualisation 3D des ratios portance/traînée (L/D)
    - **Zones Optimales** : Identification des meilleures configurations
    - **Analyse Comparative** : Comparaison entre profils
    - **Stabilité** : Robustesse des performances
    
    ---
    """)

def _display_conclusion():
    """Affiche la conclusion de l'analyse"""
    st.markdown("""
    ---
    ## 🎯 Conclusion et Recommandations

    ### Points Clés
    - **Zone Optimale** : Les profils atteignent leur meilleur ratio L/D entre 2° et 4° d'angle d'attaque
    - **Performance Maximale** : Le profil e63 montre les meilleures performances avec un L/D de 55.1
    - **Reynolds Optimal** : Performances optimales autour de 1M (10⁶), offrant un bon compromis stabilité/performance

    ### 💡 Recommandations Pratiques
    - **Profil Optimal** : Le e63 est recommandé pour les applications nécessitant une performance maximale
    - **Alternative Stable** : Le profil ua79sfm offre un excellent compromis entre performance (L/D = 52.3) et stabilité
    - **Plage d'Utilisation** : Maintenir l'angle d'attaque autour de 3° (±1°) pour des performances optimales

    ### 🔄 Synthèse
    Notre analyse montre que les profils modernes comme le e63 et ua79sfm surpassent significativement les designs classiques. 
    La zone de performance optimale est bien définie et reproductible, offrant une base solide pour les applications pratiques.
    """)

def optimisation(df_sample):
    """
    Page d'optimisation du dashboard pour l'analyse des performances des profils d'aile.
    
    Args:
        df_sample (pd.DataFrame): DataFrame contenant les données des profils d'aile
    """
    # En-tête et introduction
    _display_header()
    
    # Préparation des données
    stability_data = _prepare_stability_data(df_sample)
    
    # Affichage des colonnes principales
    col1, col2 = st.columns([1, 1])
    
    with col1:
        _display_performance_overview(df_sample)
    
    with col2:
        _display_detailed_profiles(df_sample, stability_data)
    
    # Conclusion
    _display_conclusion()

def _prepare_stability_data(df_sample):
    """
    Prépare les données de stabilité pour l'analyse
    
    Args:
        df_sample (pd.DataFrame): DataFrame source
        
    Returns:
        pd.DataFrame: DataFrame avec les métriques de stabilité calculées
    """
    stability_data = df_sample.groupby('name').agg({
        'Lift-to-Drag Ratio': ['max', 'mean', 'std']
    }).round(2)
    
    stability_data.columns = ['max_performance', 'mean_performance', 'stability']
    stability_data = stability_data.reset_index()
    
    # Normalisation pour le score composite
    stability_data['norm_performance'] = (stability_data['max_performance'] - stability_data['max_performance'].min()) / (stability_data['max_performance'].max() - stability_data['max_performance'].min())
    stability_data['norm_stability'] = 1 - (stability_data['stability'] - stability_data['stability'].min()) / (stability_data['stability'].max() - stability_data['stability'].min())
    stability_data['score_composite'] = ANALYSIS_PARAMS['performance_weight'] * stability_data['norm_performance'] + ANALYSIS_PARAMS['stability_weight'] * stability_data['norm_stability']
    
    return stability_data

# =============================================================================
# 5. LANCEMENT DE L'APPLICATION
# =============================================================================

if __name__ == "__main__":
    df_sample = pd.read_csv("data.csv")  # Remplacez par votre fichier de données
    optimisation(df_sample)
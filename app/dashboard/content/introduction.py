import streamlit as st
import os 
import base64
from src.naca0012 import *

def load_gif(gif_path):
    file_ = open(gif_path, "rb")
    contents = file_.read()
    data_url = base64.b64encode(contents).decode("utf-8")
    file_.close()
    return data_url

def introduction():
    """
    Introduction page of the dashboard
    """
    # Titre principal
    st.markdown("# 🚀 Exploration Aérodynamique des Profils Airfoils 🚀")
    # Petit séparateur
    st.markdown("---")


    st.markdown("""
        ## 🛩️ Introduction
        
        Bienvenue dans cet espace d'analyse et de modélisation dédié à l'aérodynamique des profils d'aile.  

        Notre **objectif** est de **comprendre, explorer et prédire les performances aérodynamiques d'une aile** en fonction de paramètres tels que l'angle d'attaque, le nombre de Reynolds, ou encore la géométrie du profil.


        Les profils d'aile influencent le flux d'air, conditionnant ainsi la **portance** ($C_L$), la **traînée** ($C_D$) et le **moment** ($C_M$).  
        
        Le ratio $\\frac{L}{D} = \\frac{C_L}{C_D}$ est un indicateur clé de l'**efficacité aérodynamique** : Un ratio élevé signifie plus de portance pour moins de traînée, ce qui est optimal pour le vol.

        
        Nous utiliserons une modélisation de machine learning en utilisant le modèle XGBoost afin d'anticiper 
        les performances aérodynamiques sans recourir à des simulations CFD coûteuses.
        </div>
        """, unsafe_allow_html=True)


    # Séparateur horizontal
    st.markdown("---")

    col1, col2 = st.columns([2,3])

    with col1:
        # Section Formules
        st.markdown("## Concepts Clés")
        st.markdown("""
            - Portance : $$C_L = \\frac{L}{\\tfrac{1}{2}\\rho V^2 S}$$  
            - Traînée : $$C_D = \\frac{D}{\\tfrac{1}{2}\\rho V^2 S}$$  
            - Moment : $$C_M = \\frac{M}{\\tfrac{1}{2}\\rho V^2 S c}$$  
            - Nombre de Reynolds : $$Re = \\frac{\\rho V L}{\\mu}$$

            Où :  
            - $\\rho$ : densité de l'air  
            - $V$ : vitesse  
            - $S$ : surface alaire  
            - $c$ : corde du profil  
            - $\\mu$ : viscosité dynamique
            """)

        st.markdown("""
            L'étude de ces paramètres permet d'anticiper les performances aérodynamiques dans un large éventail de conditions.
        """)
        # Paramètres du profil NACA 0012
        m = 0.0    # Cambrure maximale (0% de la corde)
        p = 0.0    # Position de la cambrure maximale (0% de la corde)
        t = 0.12   # Épaisseur relative (12% de la corde)

        # Génération des coordonnées
        x_airfoil, y_airfoil = naca4_profile(m, p, t)

        panels = define_panels(x_airfoil, y_airfoil)
        freestream = Freestream(u_inf=1.0, alpha=5.0)
        A = source_contribution_normal(panels)
        b = build_rhs(panels, freestream)
        # 6. Résolution du système linéaire pour obtenir les intensités des sources
        sigma = np.linalg.solve(A, b)
        for i, panel in enumerate(panels):
            panel.sigma = sigma[i]
        get_tangential_velocity(panels, freestream)


        # 8. Calcul du champ de vitesse
        Nx, Ny = 200, 200  # Nombre de points dans la grille
        X_start, X_end = -0.5, 1.5
        Y_start, Y_end = -0.5, 0.5
        X, Y = np.meshgrid(np.linspace(X_start, X_end, Nx), np.linspace(Y_start, Y_end, Ny))

        # Initialisation des composantes de vitesse
        u = freestream.u_inf * np.cos(freestream.alpha) * np.ones_like(X)
        v = freestream.u_inf * np.sin(freestream.alpha) * np.ones_like(X)

        # Calcul des vitesses induites par les sources
        for panel in panels:
            dx = X - panel.xc
            dy = Y - panel.yc
            r_squared = dx**2 + dy**2
            u += panel.sigma / (2 * np.pi) * dx / r_squared
            v += panel.sigma / (2 * np.pi) * dy / r_squared



    with col2:
        # Contexte et scénario
        st.markdown("## Contexte Historique & Scénario")
        st.markdown("""
            Les profils d'aile ont évolué des profils épais historiques (Clark-Y, NACA) aux profils supercritiques modernes, 
            offrant de meilleures performances à grande vitesse ou des qualités de vol spécifiques.  
            Aujourd'hui, la modélisation numérique et l'intelligence artificielle permettent d'**explorer virtuellement** des milliers 
            de profils, d'identifier les plus prometteurs, et d'**accélérer l'innovation** en aéronautique.

            **Scénario d'application** :  
            Imaginons un ingénieur concevant une aile de drone plus efficace. Plutôt que d'essayer chaque profil en soufflerie, 
            il utilise ce modèle pour estimer rapidement le ratio L/D, filtrer les profils moins performants, 
            et ainsi **réduire le temps et les coûts** de R&D.
        """)
        st.markdown(f"##### Écoulement autour du profil d'aile NACA0012")
    
        gif_path = "dashboard/content/naca0012_vorticity.gif"
        data_url = load_gif(gif_path)
        st.markdown(f"<div style='text-align:center;'><img src='data:image/gif;base64,{data_url}' alt='NACA0012 Flow'></div>",
                    unsafe_allow_html=True)


    st.markdown("---")
    st.markdown("""L'ensemble du code réalísé par Johan Ghré et Hoang Thuy Duong Vu""")
        
import streamlit as st
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import kagglehub
import plotly.graph_objects as go

# Set a Seaborn style
sns.set_theme(style="darkgrid")  # Options: "darkgrid", "whitegrid", "dark", "white", "ticks"

# Optional: Set a specific context for the plots (adjusts font size and other elements)
sns.set_context("notebook")  # Options: "paper", "notebook", "talk", "poster"

def data_management(featured_df, stratified_df):
    """
    Data exploration page of the dashboard
    """

    # data_path = "victorienmichel/deeplearwing"
    # dataset_path = kagglehub.dataset_download(data_path)
    # files = os.listdir(dataset_path)
    # csv_file = os.path.join(dataset_path, files[0])
    # df = pd.read_csv(csv_file)

    st.markdown("# 🔍 Gestion des Données 🔍")
    st.markdown("---")

    st.markdown("## Exploration de données")


    # print("Chargement des données")

    # Download data
    dataset_path = kagglehub.dataset_download("victorienmichel/deeplearwing")
    print("Fin du téléchargement")

    # Verify that the data has been downloaded
    files = os.listdir(dataset_path)
    print("Data path:",dataset_path)
    print("Files in dataset directory:",files)

    # Load data
    csv_file = os.path.join(dataset_path, files[0])
    df = pd.read_csv(csv_file)
    


    st.markdown("""
    ### Description de l'ensemble des données de DeepLearWing.csv 
    **Note**: Base de données en input, avant tout traitement de données
    #### Origine des données
        URL du dataset: "https://www.kaggle.com/datasets/victorienmichel/deeplearwing?select=DeepLearWing.csv"

        Data path sauvegardé dans l'environement de chacun: "/Users/{user}/.cache/kagglehub/datasets/victorienmichel/deeplearwing/versions/1"

    #### Nombre d'observations
        819191

    #### Nombre de variables
        8

    #### Types de variables avec leurs significations
    Cet ensemble de données contient des informations liées à l'aérodynamique des profils d'ailes, avec les colonnes suivantes :

    - **`name`** -- `object` : Le nom ou l'identifiant unique du profil d'aile.
    - **`angle`** -- `float64` : L'angle d'attaque (en degrés) du profil d'aile.
    - **`reynolds`** -- `int64` : Le nombre de Reynolds, qui caractérise le régime d'écoulement de l'air autour de l'aile.
    - **`x_coords`** -- 'object' : Coordonnées en X des points du profil d'aile (dans un repère 2D).
    - **`y_coords`** -- 'object' : Coordonnées en Y des points du profil d'aile (dans un repère 2D).
    - **`cd`** -- `float64` : Coefficient de traînée (Drag coefficient), une mesure de la résistance à l'écoulement.
    - **`cl`** -- `float64` : Coefficient de portance (Lift coefficient), une mesure de la force de portance générée.
    - **`cm`** -- `float64` : Coefficient de moment (Moment coefficient), lié au moment aérodynamique autour d'un point donné.

    #### Colonnes de données (total : 8 colonnes)

    | #  | Colonne   | Nombre de valeurs non nulles | Type de données |Nombre de valeurs manquantes|
    |----|-----------|------------------------------|-----------------|-|
    | 0  | `name`    | 819191                       | `object`        |0|
    | 1  | `angle`   | 819191                       | `float64`       |0|
    | 2  | `reynolds`| 819191                       | `int64`         |0|
    | 3  | `x_coords`| 819191                       | `object`        |0|
    | 4  | `y_coords`| 819191                       | `object`        |0|
    | 5  | `cd`      | 819191                       | `float64`       |0|
    | 6  | `cl`      | 819191                       | `float64`       |0|
    | 7  | `cm`      | 819191                       | `float64`       |0|

    #### Résumé des types de données
    - **Types** :
    - `float64` : 4 colonnes
    - `int64` : 1 colonne
    - `object` : 3 colonnes

    #### Utilisation mémoire
    - La mémoire totale utilisée par cet ensemble de données est d'environ **50.0+ MB**.
    """)


    st.markdown("""
    ```python
    # Chargement des données
    print("Chargement des données")
    data_path = "victorienmichel/deeplearwing"

    # Download data
    dataset_path = kagglehub.dataset_download(data_path)
    print("Fin du téléchargement")

    # Verify that the data has been downloaded
    files = os.listdir(dataset_path)
    print("Data path:",dataset_path)
    print("Files in dataset directory:",files)

    # Load data
    csv_file = os.path.join(dataset_path, files[0])
    df = pd.read_csv(csv_file)

    print("Chargement avec succès")

    # Affichage des 5 premières lignes
    print("--------------------------------- DeepLearWing dataset ---------------------------------")
    df.head(5)
    ```""")
    st.write(df.head(5))

    st.markdown("""### shape_""")


    st.markdown("""
    ```python
    print("Database dimensions:",df.shape)
    print("---------------------")
    print("Database information:")
    print("---------------------")
    df.info()
    ```""")
    st.markdown("""
    ```txt
    Database dimensions: (819191, 8)
    ---------------------

    Database information:
    ---------------------
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 819191 entries, 0 to 819190
    Data columns (total 8 columns):
    #   Column    Non-Null Count   Dtype  
    ---  ------    --------------   -----  
    0   name      819191 non-null  object 
    1   angle     819191 non-null  float64
    2   reynolds  819191 non-null  int64  
    3   x_coords  819191 non-null  object 
    4   y_coords  819191 non-null  object 
    5   cd        819191 non-null  float64
    6   cl        819191 non-null  float64
    7   cm        819191 non-null  float64
    dtypes: float64(4), int64(1), object(3)
    memory usage: 50.0+ MB
    ```""")

    st.markdown("""### describe_""")

    st.markdown("""
    ```python
    df.describe()
    ```""")

    st.write(df.describe())

    st.markdown("""### Gestion des valeurs manquantes""")
    st.markdown("""
    ```python
    empty_before = df.isna().sum()
    print(f"Empty values (Before cleaning) :-------------------------------------{empty_before}")

    remove_na_values(df)
    df = refill_na_values(df)

    empty_after = df.isna().sum()
    print(f"Empty values (After cleaning) :------------------------------------{empty_after}")
    ```""")
    st.markdown("""
    ```txt
    Empty values (Before cleaning) :
    -------------------------------------
    name        0
    angle       0
    reynolds    0
    x_coords    0
    y_coords    0
    cd          0
    cl          0
    cm          0
    dtype: int64

    Empty values (After cleaning) :
    ------------------------------------
    name        0
    angle       0
    reynolds    0
    x_coords    0
    y_coords    0
    cd          0
    cl          0
    cm          0
    dtype: int64
    ```""")


    st.markdown("""## Extraction de données""")

    st.markdown("### Chargement de données")
    st.markdown("""
    ```python
    engineer_aerodynamic_features(
        df,
        sample_size=200000,
        save_path='data/feature_engineered_dataset.csv'
    )

    featured_df = pd.read_csv("data/echantillon_stratifie.csv")
    print("Database dimension:",featured_df.shape)
    featured_df.head(5)""")
    # featured_df = pd.read_csv("data/echantillon_stratifie.csv")
    st.markdown("```Database dimension: (200000, 14)```")
    st.write(stratified_df.head(5))

    col1, col2 = st.columns([2, 3])

    with col1:
        st.markdown("### Statistiques des coefficients aéorodynamiques")
        st.markdown("""
        ```python
        featured_df[["cd", "cl", "cm"]].describe()
        ```""")
        st.write(stratified_df[["cd", "cl", "cm"]].describe())

        st.markdown("""
        #### 🔍 Analyse de la Matrice de Corrélation

        ##### 1. Corrélations Fortes (> 0.7)
        - **Angle et cl** : Très forte corrélation (0.919)
        - **Angle × Reynolds** : Forte influence sur la performance (0.782)
        - **Reynolds** : Forte corrélation entre ses différentes représentations (>0.93)

        ##### 2. Corrélations Modérées (0.3-0.7)
        - **Lift-to-Drag Ratio** : 
          - Positif avec angle (0.544) et cl (0.733)
          - Négatif avec cd (-0.327)
        - **cd** : Corrélation modérée avec Angle^2 (0.516) et |Angle| (0.514)

        ##### 3. Point Clé
        - Le coefficient de moment (cm) reste largement indépendant (corrélations < 0.2)
        - Les transformations du Reynolds (Log, Normalized) sont fortement corrélées entre elles
        """)

        st.markdown("""
        #### 🎯 Impact sur le Ratio Lift-to-Drag

        La matrice révèle des informations cruciales pour l'optimisation du ratio L/D :

        ##### Influences Positives
        - **Coefficient de portance (cl)** : Forte corrélation (0.733)
        - **Angle d'attaque** : Corrélation modérée (0.544)
        - **Angle × Reynolds** : Corrélation modérée (0.533)

        ##### Influences Négatives
        - **Coefficient de traînée (cd)** : Corrélation négative (-0.327)
        - **Coefficient de moment (cm)** : Légèrement négatif (-0.271)

        
        """)

    with col2:
        st.markdown("### Matrice de corrélation")

        # Afficher les colonnes disponibles
        st.write("Colonnes disponibles dans le dataset :", stratified_df.columns.tolist())
        
        # Sélectionner toutes les colonnes numériques pertinentes pour la corrélation
        numeric_cols = stratified_df.select_dtypes(include=['float64', 'int64']).columns.tolist()
        correlation_matrix = stratified_df[numeric_cols].corr()
        
        # Créer la heatmap
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(correlation_matrix, 
                   annot=True, 
                   cmap='RdGy_r', 
                   center=0,
                   fmt='.3f',
                   square=True,
                   ax=ax)
        
        plt.title('Correlation Matrix')
        plt.tight_layout()

        # Display the plot in Streamlit
        st.pyplot(fig)

        st.markdown("""
        ##### Conclusion
        Pour maximiser le ratio L/D, il faut :
        1. Privilégier les configurations optimisant cl
        2. Contrôler l'angle d'attaque dans la plage optimale
        3. Minimiser cd tout en maintenant un bon cl
        4. Le nombre de Reynolds a un effet modéré mais positif (0.237)
        """)


    st.markdown("### 📊 Distribution des Mesures par Profil")

    # Analyse de la distribution des mesures par profil
    
    profile_counts = stratified_df.groupby('name').agg({
        'Lift-to-Drag Ratio': ['count', 'mean']
    }).round(2)
    
    profile_counts.columns = ['Nombre de Mesures', 'L/D Moyen']
    profile_counts = profile_counts.sort_values('Nombre de Mesures', ascending=True)

    # Création du graphique
    fig = go.Figure()

    # Barres horizontales pour le nombre de mesures
    fig.add_trace(go.Bar(
        x=profile_counts['Nombre de Mesures'],
        y=profile_counts.index,
        orientation='h',
        name='Nombre de Mesures',
        marker_color='lightgreen',
        hovertemplate="<b>%{y}</b><br>" +
                     "Mesures: %{x}<br>" +
                     "<extra></extra>"
    ))

    # Mise en page
    fig.update_layout(
        title=dict(
            text='Répartition des Mesures par Profil',
            x=0.5,
            xanchor='center'
        ),
        xaxis_title="Nombre de Mesures",
        yaxis_title="Profil d'Aile",
        height=800,
        showlegend=False,
        plot_bgcolor='white',
        yaxis=dict(
            showgrid=True,
            gridcolor='rgba(128,128,128,0.2)',
            zeroline=False,
        ),
        xaxis=dict(
            showgrid=True,
            gridcolor='rgba(128,128,128,0.2)',
            zeroline=False,
        ),
        hoverlabel=dict(
            bgcolor="white",
            font_size=12,
            font_family="Arial"
        )
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    #### 📈 Analyse de la Distribution

    Ce graphique montre la répartition des mesures entre les différents profils :
    - Permet d'identifier les profils les plus/moins testés
    - Aide à évaluer la représentativité des données par profil
    - Met en évidence d'éventuels déséquilibres dans notre jeu de données
        
    #### 💡 Implications pour l'Analyse
    - Les profils avec plus de mesures offrent des résultats plus fiables
    - Les profils moins testés pourraient nécessiter des tests supplémentaires
    - La distribution peut influencer la robustesse de nos comparaisons
    """)

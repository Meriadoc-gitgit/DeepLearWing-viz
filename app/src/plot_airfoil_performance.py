import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import streamlit as st

@st.cache_data
def plot_airfoil_performance(df, sample_size=30):
    # Calculer CL max et CD min pour chaque profil
    performance_metrics = df.groupby('name').agg({
        'cl': 'max',  # Maximum CL
        'cd': 'min'   # Minimum CD
    }).reset_index()
    
    # Calculer L/D ratio moyen pour le tri
    efficiency = df.groupby('name')['Lift-to-Drag Ratio'].mean()
    performance_metrics['efficiency'] = performance_metrics['name'].map(efficiency)
    
    # Trier par efficacité et prendre un échantillon des meilleurs profils
    top_profiles = performance_metrics.nlargest(sample_size, 'efficiency')
    
    # Créer le plot
    plt.figure(figsize=(12, 8))
    
    # Créer un colormap unique pour chaque profil
    colors = sns.color_palette("husl", n_colors=len(top_profiles))
    
    # Scatter plot
    scatter = plt.scatter(top_profiles['cd'], top_profiles['cl'], 
                         c=range(len(top_profiles)), 
                         cmap='viridis', 
                         s=100)
    
    # Ajouter les labels pour les points les plus performants
    for idx, row in top_profiles.iterrows():
        plt.annotate(row['name'], 
                    (row['cd'], row['cl']),
                    xytext=(5, 5), 
                    textcoords='offset points',
                    fontsize=8,
                    alpha=0.7)
    
    # Personnalisation du plot
    plt.xlabel('Coefficient de traînée minimum (CD)', fontsize=12)
    plt.ylabel('Coefficient de portance maximum (CL)', fontsize=12)
    plt.title('Performance des Profils Aérodynamiques\nCL max vs CD min', fontsize=14)
    
    # Ajouter une colorbar pour montrer l'efficacité
    cbar = plt.colorbar(scatter)
    cbar.set_label('Rang d\'efficacité (du meilleur au moins bon)', fontsize=10)
    
    # Identifier les profils Pareto-optimaux
    pareto_optimal = []
    for i, row_i in top_profiles.iterrows():
        is_pareto = True
        for j, row_j in top_profiles.iterrows():
            if i != j:
                if (row_j['cd'] <= row_i['cd'] and row_j['cl'] >= row_i['cl']):
                    is_pareto = False
                    break
        if is_pareto:
            pareto_optimal.append(row_i['name'])
    
    # Mettre en évidence les profils Pareto-optimaux
    pareto_profiles = top_profiles[top_profiles['name'].isin(pareto_optimal)]
    plt.scatter(pareto_profiles['cd'], pareto_profiles['cl'], 
               color='red', s=200, facecolors='none', 
               label='Profils Pareto-optimaux')
    
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    
    # Afficher les profils Pareto-optimaux
    print("\nProfils Pareto-optimaux (meilleurs compromis CL/CD):")
    for name in pareto_optimal:
        profile_data = top_profiles[top_profiles['name'] == name].iloc[0]
        print(f"\nProfil: {name}")
        print(f"CL max: {profile_data['cl']:.3f}")
        print(f"CD min: {profile_data['cd']:.5f}")
        print(f"Efficacité moyenne (L/D): {profile_data['efficiency']:.2f}")
    
    
    return pareto_optimal
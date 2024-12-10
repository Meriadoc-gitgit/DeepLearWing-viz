# ---------------------
# HANDLE MISSING VALUES
# ---------------------
def remove_na_values(df):
    """
    Remove columns with more than 50% of missing values

    Input:
    -------------------
    df: pandas dataframe
    columns: list of columns to be removed

    Ouput:
    -------------------
    None
    """
    columns = df.columns
    for column in columns:
        if df.isna().sum()[column] > len(df)/2:
            df = df.drop(column, axis=1)
    return
        

def refill_na_values(df):
    """
    Refill missing values with the mean of the column
    
    Input:
    -------------------
    df: pandas dataframe
    columns: list of columns to be removed

    Ouput:
    -------------------
    df: pandas dataframe
    """
    for column in df.columns:
        df[column] = df[column].fillna(df[column].mode())
    return df


# --------------------------------
# FEATURE ENGINEERING
# --------------------------------
import pandas as pd
import numpy as np
from typing import Union, Optional
from tqdm import tqdm


def engineer_aerodynamic_features(
    df,
    sample_size: Optional[int] = None,
    random_state: int = 42,
    save_path: Optional[str] = None
) -> None:
    """
    Effectue le feature engineering sur un dataset aérodynamique.
    
    Args:
        data_path (str): Chemin vers le fichier CSV source
        sample_size (int, optional): Nombre d'échantillons à sélectionner. Si None, garde toutes les données
        random_state (int): Seed pour la reproduction des résultats
        save_path (str, optional): Chemin pour sauvegarder le résultat. Si None, pas de sauvegarde
        
    Returns:
        pd.DataFrame: DataFrame avec les nouvelles features
    """
    
    print("Début du feature engineering...")




    # Liste des opérations à effectuer
    # A list of feature engineering operations defined as tuples:
    # (Description of the operation, Lambda function to calculate the feature).
    operations = [
        ("Calcul du Lift-to-Drag Ratio", lambda df: df['cl'] / df['cd']),  # Compute the lift-to-drag ratio
        ("Calcul de Angle x Reynolds", lambda df: df['angle'] * df['reynolds']),  # Multiply angle by Reynolds number
        ("Calcul du Log Reynolds", lambda df: np.log(df['reynolds'])),  # Compute the logarithm of Reynolds number
        ("Calcul du Normalized Reynolds", lambda df: (df['reynolds'] - df['reynolds'].mean()) / df['reynolds'].std()),  # Normalize Reynolds number
        ("Calcul de Angle^2", lambda df: df['angle'] ** 2),  # Square the angle
        ("Calcul de |Angle|", lambda df: abs(df['angle']))  # Compute the absolute value of the angle
    ]

    # Noms des nouvelles colonnes
    # A list of names for the new columns created by the feature engineering operations.
    new_columns = [
        'Lift-to-Drag Ratio',  # Column for lift-to-drag ratio
        'Angle x Reynolds',  # Column for angle multiplied by Reynolds number
        'Log Reynolds',  # Column for logarithm of Reynolds number
        'Normalized Reynolds',  # Column for normalized Reynolds number
        'Angle^2',  # Column for squared angle
        '|Angle|'  # Column for absolute value of the angle
    ]

    # Exécution des opérations avec barre de progression
    # Apply each operation to the DataFrame and create a new column for the result.
    for (description, operation), col_name in tqdm(zip(operations, new_columns), 
                                                total=len(operations), 
                                                desc="Feature Engineering"):
        # Apply the operation (lambda function) to the DataFrame and store the result in the new column
        df[col_name] = operation(df)

    # Échantillonnage si demandé
    # If a sample size is specified, randomly sample rows from the DataFrame.
    if sample_size is not None:
        print(f"Sélection aléatoire de {sample_size} échantillons...")  # Inform about random sampling
        df = df.sample(n=sample_size, random_state=random_state)  # Randomly sample rows with a fixed seed for reproducibility
        df.reset_index(inplace=True,drop=True)  # Reset the index of the DataFrame after sampling

    
    # # Sauvegarde si un chemin est spécifié
    # if save_path:
    #     print(f"Sauvegarde des résultats dans {save_path}...")
    #     df.to_csv(save_path,index=False)
    
    print("Feature engineering terminé!")
    return df



# --------------------------------
# STRATIFIED SAMPLING
# --------------------------------
def create_stratified_sample(
    df, 
    save_path: Optional[str] = None,
    n=10000):

    print("Début du sample stratification..")

    # Stratification par profil et angle
    df['angle_bin'] = pd.qcut(df['angle'], q=5, duplicates='drop')
    df['reynolds_bin'] = pd.qcut(df['reynolds'], q=5, duplicates='drop')
    
    # Échantillonnage stratifié
    sample = df.groupby(['name', 'angle_bin', 'reynolds_bin']).apply(
        lambda x: x.sample(n=max(1, int(n * len(x)/len(df))), random_state=42)
    ).reset_index(drop=True)
    
    # Nettoyage
    sample = sample.drop(['angle_bin', 'reynolds_bin'], axis=1)

    # Sauvegarde si un chemin est spécifié
    # if save_path:
    #     print(f"Sauvegarde des résultats dans {save_path}...")
    #     sample.sample(n=min(n, len(sample)), random_state=42).to_csv(save_path,index=False)

    print("Sample stratification terminé!")
    
    return sample.sample(n=min(n, len(sample)), random_state=42)
import os
import kagglehub
import pandas as pd

from src.preprocess import *



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


featured_df = engineer_aerodynamic_features(
    df,
    sample_size=200000,
    save_path='data/feature_engineered_dataset.csv'
)

stratified_df = create_stratified_sample(
    featured_df, 
    n=200000, 
    save_path='data/echantillon_stratifie.csv'
) 
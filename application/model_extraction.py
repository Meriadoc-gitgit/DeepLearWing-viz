from src.improved_aerodynamic_model import *
import os
import pandas as pd 

df_model = pd.read_csv('data/echantillon_stratifie.csv')

print(os.getcwd())

if not os.path.exists('dashboard/content/xgboost_model.joblib'):
    model = ImprovedAerodynamicModel(df_model)
    model.add_geometric_features()
    model.create_advanced_features()
    model.segment_flow_regimes()
    model.prepare_features()
    model.train()
    model.save_model('dashboard/content/xgboost_model.joblib')
else:
    model = ImprovedAerodynamicModel.load_model('dashboard/content/xgboost_model.joblib')
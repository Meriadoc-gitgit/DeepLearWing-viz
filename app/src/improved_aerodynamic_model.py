import streamlit as st
import pandas as pd
import numpy as np

import base64
import os
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import xgboost as xgb
import math

# df = pd.read_csv('data/echantillon_stratifie.csv')
# ---------------------------------------------------------------------------
# Classe du modèle XGBoost
# ---------------------------------------------------------------------------
class ImprovedAerodynamicModel:
    def __init__(self, df):
        self.df = df.copy()
        self.target = 'Lift-to-Drag Ratio'
        self.feature_importances = {}
        
    def add_geometric_features(self):
        def process_coords(row):
            x_str = row['x_coords']
            y_str = row['y_coords']
            if isinstance(x_str, str) and isinstance(y_str, str):
                x = np.array([float(xc) for xc in x_str.split()])
                y = np.array([float(yc) for yc in y_str.split()])
            else:
                return pd.Series({
                    'thickness_max': np.nan,
                    'camber_max': np.nan,
                    'leading_edge_radius': np.nan,
                    'trailing_edge_angle': np.nan,
                    'profile_area': np.nan,
                    'max_thickness_position': np.nan
                })
            if len(x) < 3 or len(y) < 3:
                return pd.Series({
                    'thickness_max': np.nan,
                    'camber_max': np.nan,
                    'leading_edge_radius': np.nan,
                    'trailing_edge_angle': np.nan,
                    'profile_area': np.nan,
                    'max_thickness_position': np.nan
                })
            try:
                thickness_max = np.max(np.abs(y))
                camber_max = np.mean([max(y), min(y)])
                if (x[1]-x[0]) != 0:
                    leading_edge_radius = np.abs(y[1] - y[0]) / (x[1] - x[0])
                else:
                    leading_edge_radius = 0.0
                if (x[-1]-x[-2]) != 0:
                    trailing_edge_angle = np.arctan((y[-1] - y[-2])/(x[-1]-x[-2]))
                else:
                    trailing_edge_angle = 0.0
                profile_area = np.trapz(y, x)
                max_thickness_position = x[np.argmax(np.abs(y))]
                return pd.Series({
                    'thickness_max': thickness_max,
                    'camber_max': camber_max,
                    'leading_edge_radius': leading_edge_radius,
                    'trailing_edge_angle': trailing_edge_angle,
                    'profile_area': profile_area,
                    'max_thickness_position': max_thickness_position
                })
            except:
                return pd.Series({
                    'thickness_max': np.nan,
                    'camber_max': np.nan,
                    'leading_edge_radius': np.nan,
                    'trailing_edge_angle': np.nan,
                    'profile_area': np.nan,
                    'max_thickness_position': np.nan
                })
        
        geom_features = self.df.apply(process_coords, axis=1)
        self.df = pd.concat([self.df, geom_features], axis=1)
        return self
        
    def create_advanced_features(self):
        self.df['angle_norm'] = self.df['angle'] / 15
        self.df['reynolds_effect'] = self.df['reynolds'] / 5e5
        self.df['log_reynolds_norm'] = np.log(self.df['reynolds_effect'] + 1e-10)
        
        angles_rad = np.radians(self.df['angle'])
        self.df['sin_angle'] = np.sin(angles_rad)
        self.df['cos_angle'] = np.cos(angles_rad)
        self.df['tan_angle'] = np.tan(angles_rad)
        self.df['sin_2angle'] = np.sin(2 * angles_rad)
        
        self.df['thickness_effect'] = self.df['thickness_max'] * self.df['sin_angle']
        self.df['camber_effect'] = self.df['camber_max'] * self.df['angle_norm']
        self.df['leading_edge_effect'] = (self.df['leading_edge_radius'] * 
                                          self.df['sin_angle'] * 
                                          self.df['reynolds_effect'])
        
        angle_factor = self.df['angle_norm'] - 1
        self.df['stall_indicator'] = 1 / (1 + np.exp(angle_factor))
        self.df['stall_geometry'] = self.df['thickness_max'] * (1 - self.df['stall_indicator'])
        self.df['dynamic_stall'] = self.df['stall_indicator'] * self.df['reynolds_effect']
        
        self.df['boundary_layer'] = 5.0 / np.sqrt(self.df['reynolds_effect'] + 1e-10)
        self.df['pressure_gradient'] = (self.df['sin_angle'] * 
                                         self.df['thickness_max'] * 
                                         self.df['reynolds_effect'])
        
        return self
        
    def segment_flow_regimes(self):
        self.df['flow_regime'] = pd.cut(
            self.df['reynolds'],
            bins=[0, 1e5, 5e5, np.inf],
            labels=['laminar', 'transition', 'turbulent']
        )
        
        for regime in ['laminar', 'transition', 'turbulent']:
            mask = (self.df['flow_regime'] == regime)
            self.df[f'{regime}_effect'] = mask.astype(float)
            self.df[f'{regime}_angle'] = self.df['angle_norm'] * mask.astype(float)
            self.df[f'{regime}_reynolds'] = self.df['log_reynolds_norm'] * mask.astype(float)
        
        return self
        
    def prepare_features(self):
        self.features = [
            'angle_norm', 'log_reynolds_norm',
            'sin_angle', 'cos_angle', 'tan_angle', 'sin_2angle',
            'thickness_max', 'camber_max',
            'leading_edge_radius', 'trailing_edge_angle',
            'profile_area', 'max_thickness_position',
            'thickness_effect', 'camber_effect',
            'leading_edge_effect', 'stall_indicator',
            'stall_geometry', 'dynamic_stall',
            'boundary_layer', 'pressure_gradient',
            'laminar_effect', 'transition_effect', 'turbulent_effect',
            'laminar_angle', 'transition_angle', 'turbulent_angle',
            'laminar_reynolds', 'transition_reynolds', 'turbulent_reynolds'
        ]
        
        return self
        
    def train(self):
        X = self.df[self.features]
        y = self.df[self.target]
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=self.df['flow_regime']
        )
        
        xgb_params = {
            'objective': 'reg:squarederror',
            'n_estimators': 1000,
            'learning_rate': 0.05,
            'max_depth': 6,
            'min_child_weight': 1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'early_stopping_rounds': 50,
            'random_state': 42
        }
        
        self.regime_models = {}
        for regime in ['laminar', 'transition', 'turbulent']:
            mask_train = self.df.loc[self.y_train.index, 'flow_regime'] == regime
            mask_test = self.df.loc[self.y_test.index, 'flow_regime'] == regime
            
            X_train_regime = self.X_train[mask_train]
            y_train_regime = self.y_train[mask_train]
            X_train_final, X_val, y_train_final, y_val = train_test_split(
                X_train_regime, y_train_regime,
                test_size=0.2, random_state=42
            )
            
            model = xgb.XGBRegressor(**xgb_params)
            model.fit(
                X_train_final, y_train_final,
                eval_set=[(X_val, y_val)],
                verbose=False
            )
            
            y_pred_train = model.predict(X_train_regime)
            y_pred_test = model.predict(self.X_test[mask_test])
            
            self.regime_models[regime] = {
                'model': model,
                'train_r2': r2_score(y_train_regime, y_pred_train),
                'test_r2': r2_score(self.y_test[mask_test], y_pred_test),
                'train_mae': mean_absolute_error(y_train_regime, y_pred_train),
                'test_mae': mean_absolute_error(self.y_test[mask_test], y_pred_test),
                'train_rmse': np.sqrt(mean_squared_error(y_train_regime, y_pred_train)),
                'test_rmse': np.sqrt(mean_squared_error(self.y_test[mask_test], y_pred_test))
            }
            
            self.feature_importances[regime] = pd.DataFrame({
                'Feature': self.features,
                'Importance': model.feature_importances_
            }).sort_values('Importance', ascending=False)
        
        return self

    # def save_model(self, filepath):
    #     joblib.dump(self, filepath)
        
    # @classmethod
    # def load_model(cls, filepath):
    #     return joblib.load(filepath)

    def prepare_input_data(self, angle, reynolds):
        input_df = pd.DataFrame({
            'angle': [angle],
            'reynolds': [reynolds]
        })
        
        input_df['angle_norm'] = input_df['angle'] / 15
        input_df['reynolds_effect'] = input_df['reynolds'] / 5e5
        input_df['log_reynolds_norm'] = np.log(input_df['reynolds_effect'] + 1e-10)
        
        angles_rad = np.radians(input_df['angle'])
        input_df['sin_angle'] = np.sin(angles_rad)
        input_df['cos_angle'] = np.cos(angles_rad)
        input_df['tan_angle'] = np.tan(angles_rad)
        input_df['sin_2angle'] = np.sin(2 * angles_rad)
        
        input_df['thickness_max'] = 0.1  
        input_df['camber_max'] = 0.05  
        input_df['leading_edge_radius'] = 0.01
        input_df['trailing_edge_angle'] = 0.1 
        input_df['profile_area'] = 0.2  
        input_df['max_thickness_position'] = 0.5  
        
        input_df['thickness_effect'] = input_df['thickness_max'] * input_df['sin_angle']
        input_df['camber_effect'] = input_df['camber_max'] * input_df['angle_norm']
        input_df['leading_edge_effect'] = (input_df['leading_edge_radius'] * 
                                           input_df['sin_angle'] * 
                                           input_df['reynolds_effect'])
        
        angle_factor = input_df['angle_norm'] - 1
        input_df['stall_indicator'] = 1 / (1 + np.exp(angle_factor))
        input_df['stall_geometry'] = input_df['thickness_max'] * (1 - input_df['stall_indicator'])
        input_df['dynamic_stall'] = input_df['stall_indicator'] * input_df['reynolds_effect']
        
        input_df['boundary_layer'] = 5.0 / np.sqrt(input_df['reynolds_effect'] + 1e-10)
        input_df['pressure_gradient'] = (input_df['sin_angle'] * 
                                          input_df['thickness_max'] * 
                                          input_df['reynolds_effect'])
        
        input_df['laminar_effect'] = 0.0
        input_df['transition_effect'] = 1.0
        input_df['turbulent_effect'] = 0.0
        input_df['laminar_angle'] = 0.0
        input_df['transition_angle'] = input_df['angle_norm']
        input_df['turbulent_angle'] = 0.0
        input_df['laminar_reynolds'] = 0.0
        input_df['transition_reynolds'] = input_df['log_reynolds_norm']
        input_df['turbulent_reynolds'] = 0.0
        
        return input_df[self.features]

# def train_and_save_model():
#     df_model = df.copy()
#     model = ImprovedAerodynamicModel(df_model)
#     model.add_geometric_features()
#     model.create_advanced_features()
#     model.segment_flow_regimes()
#     model.prepare_features()
#     model.train()
#     model.save_model('model/xgboost_model.joblib')
#     return model

def display_model_performance(model):
    metrics_data = []
    for regime, metrics in model.regime_models.items():
        metrics_data.append({
            'Régime': regime.capitalize(),
            'Train R²': f"{metrics['train_r2']:.2f}",
            'Test R²': f"{metrics['test_r2']:.2f}",
            'Train MAE': f"{metrics['train_mae']:.2f}",
            'Test MAE': f"{metrics['test_mae']:.2f}",
            'Train RMSE': f"{metrics['train_rmse']:.2f}",
            'Test RMSE': f"{metrics['test_rmse']:.2f}"
        })
    metrics_df = pd.DataFrame(metrics_data)
    return metrics_df
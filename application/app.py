from dashboard.layout import *
import streamlit as st
import pandas as pd

def main():

    @st.cache_data
    def load_csv():
        
        featured_df = pd.read_csv("data/feature_engineered_dataset.csv")
        stratified_df = pd.read_csv("data/echantillon_stratifie.csv")
        
        return featured_df, stratified_df

    featured_df, stratified_df = load_csv()

    print("BEGINNING OF THE DASHBOARD")

    app_layout(featured_df, stratified_df)

if __name__ == '__main__':
    main()

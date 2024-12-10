from dashboard.layout import *
import kagglehub
import os

def main():

    # market_data = load_csv("data/output/Bitcoin_market_historical.csv")
    # tweet_data = load_csv("data/output/Bitcoin_tweets_historical.csv")


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

    # df = copy.deepcopy(stratified_df)

    print("BEGINNING OF THE DASHBOARD")

    app_layout(df, featured_df, stratified_df)

if __name__ == '__main__':
    main()

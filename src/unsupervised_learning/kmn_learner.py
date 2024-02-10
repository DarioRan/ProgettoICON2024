import ast

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from joblib import dump

class KMNClusterer:
    def __init__(self, df, n_clusters=5, random_state=42):
        """
        Initializes the KMNClusterer class with the provided dataset, number of clusters, and a random state for
        reproducibility. It sets up the basic configuration for clustering, including the default numerical and categorical
        features to be used in the analysis.

        Parameters:
        - df (pd.DataFrame): The dataset containing the dishes' information.
        - n_clusters (int, optional): The number of clusters to form. Defaults to 5.
        - random_state (int, optional): A seed value to ensure reproducibility. Defaults to 42.
        """
        self.dishes_df = df
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.model = None
        self.scaler = StandardScaler()
        self.categorical_features = ['restaurant_name', 'day_of_the_week', 'dish_name']
        self.numerical_features = ['latitude', 'longitude']
        self.initialize()

    def load_data(self):
        """
        Prepares the dataset for clustering by extracting latitude and longitude coordinates from the 'restaurant_location'
        column and creating a feature matrix 'X' with both categorical and numerical features.
        """
        self.dishes_df[['latitude', 'longitude']] = self.dishes_df['restaurant_location'].apply(
            lambda loc: pd.Series(ast.literal_eval(loc))
        )
        self.X = self.dishes_df[['restaurant_name', 'day_of_the_week', 'dish_name', 'latitude', 'longitude']]

    def preprocess(self):
        """
        Applies preprocessing to the feature matrix 'X', including one-hot encoding for categorical features and scaling
        for numerical features. The preprocessed data is stored in 'X_scaled', ready for clustering.
        """
        one_hot_encoder = OneHotEncoder(handle_unknown='ignore')
        self.preprocessor = ColumnTransformer(transformers=[
            ('cat', one_hot_encoder, ['restaurant_name', 'day_of_the_week', 'dish_name']),
            ('num', 'passthrough', ['latitude', 'longitude'])
        ])
        self.X_scaled = self.preprocessor.fit_transform(self.X)

    def initialize_model(self):
        """
       Initializes the KMeans clustering model with the specified number of clusters and random state, making it ready
       for training.
       """
        self.model = KMeans(n_clusters=self.n_clusters, random_state=self.random_state)

    def clusterize(self):
        """
        Applies the KMeans model to the preprocessed data to assign each dish to a cluster. The cluster labels are added
        to the original dataframe in a new 'cluster' column.

        Returns:
        - dishes_df (pd.DataFrame): The original dataframe augmented with a 'cluster' column indicating the cluster
          assignment for each dish.
        """
        cluster_labels = self.model.predict(self.X_scaled)
        self.dishes_df['cluster'] = cluster_labels
        return self.dishes_df

    def evaluate_model(self):
        """
        Evaluates the fitted KMeans model by printing its inertia. Inertia is the sum of squared distances of samples
        to their closest cluster center, indicating how well the clusters are formed.
        """
        inertia = self.model.inertia_
        print(f'Model Inertia: {inertia}')

    def train_model(self):
        """
        Fits the KMeans model to the scaled feature matrix 'X_scaled', effectively clustering the dataset based on the
        preprocessed features.
        """
        self.model.fit(self.X_scaled)

    def save_model(self):
        """
        Saves the trained KMeans model to a file for future use, allowing the model to be loaded and applied without
        retraining.
        """
        dump(self.model, 'output/models/kmn_clusterer.joblib')

    def initialize(self):
        self.load_data()
        self.preprocess()
        self.initialize_model()
        self.train_model()
        self.evaluate_model()

    def predict(self, new_data):
        new_data_scaled = self.preprocessor.transform(new_data)
        return self.model.predict(new_data_scaled)

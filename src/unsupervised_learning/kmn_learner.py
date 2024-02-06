import ast

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder


class KMNClusterer:
    def __init__(self, df, n_clusters=5, random_state=42):
        self.dishes_df = df
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.model = None
        self.scaler = StandardScaler()
        self.categorical_features = ['restaurant_name', 'day_of_the_week', 'dish_name']
        self.numerical_features = ['latitude', 'longitude']
        self.initialize()

    def load_data(self):
        self.dishes_df[['latitude', 'longitude']] = self.dishes_df['restaurant_location'].apply(
            lambda loc: pd.Series(ast.literal_eval(loc))
        )
        self.X = self.dishes_df[['restaurant_name', 'day_of_the_week', 'dish_name', 'latitude', 'longitude']]

    def preprocess(self):
        one_hot_encoder = OneHotEncoder(handle_unknown='ignore')
        self.preprocessor = ColumnTransformer(transformers=[
            ('cat', one_hot_encoder, ['restaurant_name', 'day_of_the_week', 'dish_name']),
            ('num', 'passthrough', ['latitude', 'longitude'])
        ])
        self.X_scaled = self.preprocessor.fit_transform(self.X)

    def clusterize(self):
        cluster_labels = self.model.predict(self.X_scaled)
        self.dishes_df['cluster'] = cluster_labels
        return self.dishes_df

    def evaluate_model(self):
        inertia = self.model.inertia_
        print(f'Model Inertia: {inertia}')

    def initialize_model(self):
        self.model = KMeans(n_clusters=self.n_clusters, random_state=self.random_state)

    def train_model(self):
        self.model.fit(self.X_scaled)

    def initialize(self):
        self.load_data()
        self.preprocess()
        self.initialize_model()
        self.train_model()
        self.evaluate_model()

    def predict(self, new_data):
        new_data_scaled = self.preprocessor.transform(new_data)
        return self.model.predict(new_data_scaled)

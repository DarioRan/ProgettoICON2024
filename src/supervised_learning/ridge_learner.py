import ast

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


class RidgeRegressor:
    def __init__(self, df, alpha=0.2, k=5):
        self.dishes_df = df
        self.alpha = alpha
        self.k = k
        self.model = None
        self.categorical_features = ['restaurant_name', 'day_of_the_week', 'dish_name']
        self.numerical_features = ['latitude', 'longitude']
        self.initialize()

    def load_data(self):
        self.dishes_df[['latitude', 'longitude']] = self.dishes_df['restaurant_location'].apply(
            lambda loc: pd.Series(ast.literal_eval(loc))
        )
        self.X = self.dishes_df[['restaurant_name', 'day_of_the_week', 'dish_name', 'latitude', 'longitude']]
        self.y = self.dishes_df['preparation_time']

    def preprocess(self):
        one_hot_encoder = OneHotEncoder(handle_unknown='ignore')
        self.preprocessor = ColumnTransformer(transformers=[
            ('cat', one_hot_encoder, self.categorical_features)],
            remainder='passthrough')

    def train_test_split(self, test_size=0.2, random_state=42):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=test_size,
                                                                                random_state=random_state)

    def initialize_model(self):
        self.model = Pipeline(steps=[
            ('preprocessor', self.preprocessor),
            ('regressor', Ridge(alpha=self.alpha))
        ])

    def cross_validate(self):
        ridge_scores = cross_val_score(self.model, self.X, self.y, cv=self.k, scoring='neg_mean_squared_error')
        self.ridge_rmse_scores = np.sqrt(-ridge_scores)
        print(f'Ridge RMSE: {self.ridge_rmse_scores.mean()} (± {self.ridge_rmse_scores.std()})')

    def initialize(self):
        self.load_data()
        self.preprocess()
        self.train_test_split()
        self.initialize_model()
        self.cross_validate()

    def predict(self, new_data):
        return self.model.predict(new_data)
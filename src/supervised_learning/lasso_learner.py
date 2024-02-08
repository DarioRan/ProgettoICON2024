import ast

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


class LassoRegressor:
    def __init__(self, df, cross_validation=False, random_state=42, alpha=0.2, k=5, categorical_features=None,
                 numerical_features=None):
        self.dishes_df = df
        if numerical_features is None:
            numerical_features = ['latitude', 'longitude']
        if categorical_features is None:
            categorical_features = ['restaurant_name', 'day_of_the_week', 'dish_name']
        self.random_state = random_state
        self.alpha = alpha
        self.k = k
        self.model = None
        self.categorical_features = categorical_features
        self.numerical_features = numerical_features
        self.cross_validation = cross_validation
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
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=test_size, random_state=random_state)

    def initialize_model(self):
        self.model = Pipeline(steps=[
            ('preprocessor', self.preprocessor),
            ('regressor', Lasso(alpha=self.alpha))
        ])

    def train_model(self):
        self.model.fit(self.X_train, self.y_train)

    def calculate_bic(self, mse, num_params):
        n = len(self.y_test)  # Number of data points
        rss = mse * n  # Residual sum of squares
        bic = n * np.log(rss / n) + num_params * np.log(n)
        return bic

    def evaluate_model(self):
        y_pred = self.model.predict(self.X_test)
        rmse = np.sqrt(mean_squared_error(self.y_test, y_pred))
        k = len(self.model.named_steps['regressor'].coef_)
        bic = self.calculate_bic(rmse, k)
        self.bic = bic
        self.rmse = rmse
        print(f'Lasso RMSE: {rmse}')
        print(f'Lasso BIC: {bic}')


    def cross_validate(self):
        lasso_scores = cross_val_score(self.model, self.X, self.y, cv=self.k, scoring='neg_mean_squared_error')
        lasso_rmse_scores = np.sqrt(-lasso_scores)
        k = len(self.model.named_steps['regressor'].coef_)
        bic = self.calculate_bic(lasso_rmse_scores, k)
        self.bic = bic
        self.rmse = lasso_rmse_scores.mean()
        print(f'Lasso Cross-validation RMSE: {lasso_rmse_scores.mean()} (± {lasso_rmse_scores.std()})')
        print(f'Lasso BIC: {bic}')

    def initialize(self):
        self.load_data()
        self.preprocess()
        self.train_test_split()
        self.initialize_model()
        self.train_model()

        if self.cross_validation:
            self.cross_validate()
        else:
            self.evaluate_model()

    def predict(self, new_data):
        return self.model.predict(new_data)

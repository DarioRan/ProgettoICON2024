import ast

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


class RidgeRegressor:
    def __init__(self, df, cross_validation = False, random_state=42, alpha=0.2, k=5, categorical_features=None, numerical_features=None):
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
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=test_size,
                                                                                random_state=random_state)

    def initialize_model(self):
        self.model = Pipeline(steps=[
            ('preprocessor', self.preprocessor),
            ('regressor', Ridge(alpha=self.alpha))
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
        print(f'Ridge RMSE: {rmse}')
        print(f'Ridge BIC: {bic}')

    def cross_validate(self):
        ridge_scores = cross_val_score(self.model, self.X, self.y, cv=self.k, scoring='neg_mean_squared_error')
        ridge_rmse_scores = np.sqrt(-ridge_scores)
        k = len(self.model.named_steps['regressor'].coef_)
        bic = self.calculate_bic(ridge_rmse_scores, k)
        self.bic = bic
        self.rmse = ridge_rmse_scores.mean()
        print(f'Ridge Cross-validation RMSE: {ridge_rmse_scores.mean()} (± {ridge_rmse_scores.std()})')
        print(f'Ridge BIC: {bic}')

    def tune_hyperparameters(self, param_grid):
        grid_search = GridSearchCV(self.model, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
        grid_search.fit(self.X_train, self.y_train)

        best_params = grid_search.best_params_
        print(best_params)
        best_estimator = grid_search.best_estimator_
        self.model = best_estimator
        return best_params

    def initialize(self):
        self.load_data()
        self.preprocess()
        self.train_test_split()
        self.initialize_model()
        param_grid = {
            'regressor__alpha': [0.0001, 0.001, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 2.0, 3.0,
                                 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 20, 50, 100, 500, 1000]
        }
        #self.tune_hyperparameters(param_grid

        self.train_model()

        if self.cross_validation:
            self.cross_validate()
        else:
            self.evaluate_model()

    def predict(self, new_data):
        return self.model.predict(new_data)

import ast

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


class LinearRegressor:
    def __init__(self, df, cross_validation=False, random_state=42, test_size=0.2, k=5, categorical_features=None, numerical_features=None):
        self.dishes_df = df
        if numerical_features is None:
            numerical_features = ['latitude', 'longitude']
        if categorical_features is None:
            categorical_features = ['restaurant_name', 'day_of_the_week', 'dish_name']
        self.random_state = random_state
        self.test_size = test_size
        self.model = None
        self.categorical_features = categorical_features
        self.numerical_features = numerical_features
        self.k = k
        self.cross_validation = cross_validation
        self.initialize()

    def load_data(self):
        self.dishes_df[['latitude', 'longitude']] = self.dishes_df['restaurant_location'].apply(
            lambda loc: pd.Series(ast.literal_eval(loc))
        )
        self.X = self.dishes_df[self.categorical_features + self.numerical_features]
        self.y = self.dishes_df['preparation_time']

    def preprocess(self):
        one_hot_encoder = OneHotEncoder(handle_unknown='ignore')
        self.preprocessor = ColumnTransformer(transformers=[
            ('cat', one_hot_encoder, self.categorical_features)],
            remainder='passthrough')

    def train_test_split(self):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=self.test_size, random_state=self.random_state)

    def initialize_model(self):
        self.model = Pipeline(steps=[
            ('preprocessor', self.preprocessor),
            ('regressor', LinearRegression())
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
        self.rmse = np.sqrt(mean_squared_error(self.y_test, y_pred))
        print(f'Linear RMSE: {self.rmse}')

    def cross_validate(self, scoring='neg_mean_squared_error'):
        scores = cross_val_score(self.model, self.X, self.y, cv=self.k, scoring=scoring)
        rmse_scores = np.sqrt(-scores)
        self.rmse = rmse_scores.mean()
        print(f'Linear Cross-validation RMSE: {rmse_scores.mean()} (± {rmse_scores.std()})')
        return rmse_scores

    def tune_hyperparameters(self, param_grid):
        grid_search = GridSearchCV(self.model, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
        grid_search.fit(self.X_train, self.y_train)

        best_params = grid_search.best_params_
        print(best_params)
        best_estimator = grid_search.best_estimator_
        self.model = best_estimator
        return best_params

    def tune_k_folds(self, k_values):
        k_fold_scores = {}
        for k in k_values:
            scores = cross_val_score(self.model, self.X, self.y, cv=k, scoring='neg_mean_squared_error')
            rmse_scores = np.sqrt(-scores)
            mean_rmse = rmse_scores.mean()
            k_fold_scores[k] = mean_rmse
            print(f'Linear {k}-fold CV RMSE: {mean_rmse} (± {rmse_scores.std()})')

        best_k = min(k_fold_scores, key=k_fold_scores.get)
        print(f'Linear Best k by lowest RMSE: {best_k}')
        self.k = best_k
        self.plot_cv_tuning(k_fold_scores)

    def plot_cv_tuning(self, k_fold_scores):
        # Plotting the RMSE for different k values
        plt.figure(figsize=(8, 6))
        plt.plot(list(k_fold_scores.keys()), list(k_fold_scores.values()), marker='o', linestyle='-', color='b')
        plt.xlabel('Number of folds (k)')
        plt.ylabel('Cross-validation RMSE')
        plt.title('Linear RMSE for Different k Values in Cross-validation')
        plt.xticks(list(k_fold_scores.keys()))
        plt.grid(True)
        plt.savefig('linear_k_tuning.png')

    def initialize(self):
        self.load_data()
        self.preprocess()
        self.train_test_split()
        self.initialize_model()

        param_grid = {
            'regressor__fit_intercept': [True, False],
            'regressor__copy_X': [True, False]
        }
        #diventa più scarso
        #self.tune_hyperparameters(param_grid)

        self.train_model()

        if self.cross_validation:
            k_values = [3, 5, 10, 20]
            self.tune_k_folds(k_values)
            self.cross_validate()
        else:
            self.evaluate_model()

        k = len(self.model.named_steps['regressor'].coef_)
        bic = self.calculate_bic(self.rmse, k)
        self.bic = bic


    def predict(self, new_data):
        return self.model.predict(new_data)

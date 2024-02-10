import ast

import numpy as np
import pandas as pd
from joblib import dump
from matplotlib import pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


class SGDLeaner:
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
            ('regressor', SGDRegressor(early_stopping=True, random_state=self.random_state))
        ])


    def tune_hyperparameters(self):
        learning_rate = ['constant', 'optimal', 'invscaling', 'adaptive']
        alpha = [0.001, 0.01, 0.1, 1, 10]
        max_iter = [5000]
        penalty = ['l2', 'l1', 'elasticnet']
        power_t = [0.25, 0.5, 0.75, 1]
        param_grid = {
            'regressor__learning_rate': learning_rate,
            'regressor__alpha': alpha,
            'regressor__max_iter': max_iter,
            'regressor__penalty': penalty,
            'regressor__power_t': power_t
        }
        grid_search = GridSearchCV(self.model, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
        grid_search.fit(self.X_train, self.y_train)

        best_params = grid_search.best_params_
        print(best_params)
        best_estimator = grid_search.best_estimator_
        self.model = best_estimator
        return best_params


    def train_model(self):
        self.model.fit(self.X_train, self.y_train)

    def cross_validate(self, scoring='neg_mean_squared_error'):
        scores = cross_val_score(self.model, self.X, self.y, cv=self.k, scoring=scoring)
        rmse_scores = np.sqrt(-scores)
        self.rmse = rmse_scores.mean()
        print(f'SGD Cross-validation RMSE: {rmse_scores.mean()} (± {rmse_scores.std()})')
        return rmse_scores

    def _tune_hyperparameters(self, param_grid):
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
            print(f'SGDLinear {k}-fold CV RMSE: {mean_rmse} (± {rmse_scores.std()})')

        best_k = min(k_fold_scores, key=k_fold_scores.get)
        print(f'SGDLinear Best k by lowest RMSE: {best_k}')
        self.k = best_k
        self.plot_cv_tuning(k_fold_scores)

    def plot_cv_tuning(self, k_fold_scores):
        # Plotting the RMSE for different k values
        plt.figure(figsize=(8, 6))
        plt.plot(list(k_fold_scores.keys()), list(k_fold_scores.values()), marker='o', linestyle='-', color='b')
        plt.xlabel('Number of folds (k)')
        plt.ylabel('Cross-validation RMSE')
        plt.title('SGDLinear RMSE for Different k Values in Cross-validation')
        plt.xticks(list(k_fold_scores.keys()))
        plt.grid(True)
        plt.savefig('output/SGDlinear_k_tuning.png')

    def calculate_bic(self, mse):
        n = len(self.y_test)  # Numero di osservazioni nel test set
        k = len(self.model.named_steps[
                    'regressor'].coef_)
        rss = mse * n  # Somma residua dei quadrati
        bic = n * np.log(rss / n) + k * np.log(n)
        return bic

    def evaluate_model(self):
        y_pred = self.model.predict(self.X_test)
        mse = mean_squared_error(self.y_test, y_pred)
        self.rmse = np.sqrt(mse)
        self.bic = self.calculate_bic(mse)
        print(f'SGDLinear RMSE: {self.rmse}')
        print(f'SGDLinear BIC: {self.bic}')

    def save_model(self):

        if self.cross_validation:
            print(f'Saving model to output/models/SGDlinear_regressor_cv.joblib')
            dump(self.model, 'output/models/SGDlinear_regressor_cv.joblib')

        else:
            print(f'Saving model to output/models/SGDlinear_regressor.joblib')
            dump(self.model, 'output/models/SGDlinear_regressor.joblib')


    def initialize(self):
        self.load_data()
        self.preprocess()
        self.train_test_split()
        self.initialize_model()

        self.train_model()

        self.tune_hyperparameters()


        if self.cross_validation:
            k_values = [3, 5, 10, 20]
            self.tune_k_folds(k_values)
            self.cross_validate()
        else:
            self.evaluate_model()


    def predict(self, new_data):
        return self.model.predict(new_data)

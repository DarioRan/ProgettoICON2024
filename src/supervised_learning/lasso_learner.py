import ast

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
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

    def cross_validate(self):
        lasso_scores = cross_val_score(self.model, self.X, self.y, cv=self.k, scoring='neg_mean_squared_error')
        lasso_rmse_scores = np.sqrt(-lasso_scores)
        self.rmse = lasso_rmse_scores.mean()
        print(f'Lasso Cross-validation RMSE: {lasso_rmse_scores.mean()} (± {lasso_rmse_scores.std()})')

    def tune_hyperparameters(self, param_grid):
        grid_search = GridSearchCV(self.model, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
        grid_search.fit(self.X_train, self.y_train)

        best_params = grid_search.best_params_
        print(f"Best parameters: {best_params}")

        alphas = grid_search.cv_results_['param_regressor__alpha']
        mean_scores = -grid_search.cv_results_['mean_test_score']

        self.plot_alpha_loss(alphas, mean_scores)

        best_estimator = grid_search.best_estimator_
        self.model = best_estimator
        return best_params

    def plot_alpha_loss(self, alphas, mean_scores):
        plt.figure(figsize=(10, 6))
        plt.plot(alphas, mean_scores, marker='o', linestyle='-', color='blue')
        plt.xlabel(r'$\alpha$')
        plt.ylabel('Mean Squared Error')
        plt.title('Lasso MSE vs. Alpha')
        plt.grid(True)
        plt.savefig('output/lasso_mse_vs_alpha.png')

    def tune_k_folds(self, k_values):
        k_fold_scores = {}
        for k in k_values:
            scores = cross_val_score(self.model, self.X, self.y, cv=k, scoring='neg_mean_squared_error')
            rmse_scores = np.sqrt(-scores)
            mean_rmse = rmse_scores.mean()
            k_fold_scores[k] = mean_rmse
            print(f'Lasso {k}-fold CV RMSE: {mean_rmse} (± {rmse_scores.std()})')

        best_k = min(k_fold_scores, key=k_fold_scores.get)
        print(f'Lasso Best k by lowest RMSE: {best_k}')
        self.k = best_k
        self.plot_cv_tuning(k_fold_scores)

    def plot_cv_tuning(self, k_fold_scores):
        # Plotting the RMSE for different k values
        plt.figure(figsize=(8, 6))
        plt.plot(list(k_fold_scores.keys()), list(k_fold_scores.values()), marker='o', linestyle='-', color='b')
        plt.xlabel('Number of folds (k)')
        plt.ylabel('Cross-validation RMSE')
        plt.title('Lasso RMSE for Different k Values in Cross-validation')
        plt.xticks(list(k_fold_scores.keys()))
        plt.grid(True)
        plt.savefig('output/lasso_k_tuning.png')

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
        print(f'Lasso RMSE: {self.rmse}')
        print(f'Lasso BIC: {self.bic}')


    def initialize(self):
        self.load_data()
        self.preprocess()
        self.train_test_split()
        self.initialize_model()
        param_grid = {
            'regressor__alpha': [0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.001, 0.0015, 0.002, 0.01, 0.05, 0.1, 0.2, 0.3,
                                 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0,
                                 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0]
        }
        self.tune_hyperparameters(param_grid)
        self.train_model()

        if self.cross_validation:
            k_values = [3, 5, 10, 20]
            self.tune_k_folds(k_values)
            self.cross_validate()
        else:
            self.evaluate_model()


    def predict(self, new_data):
        return self.model.predict(new_data)

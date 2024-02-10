import ast
from joblib import dump
import xgboost as xgb
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import OneHotEncoder


class BoostedRegressor:
    def __init__(self, df, cross_validation=False, param_tuning=False, random_state=42, test_size=0.2, k=5, categorical_features=None, numerical_features=None):
        self.dishes_df = df
        if numerical_features is None:
            numerical_features = ['latitude', 'longitude']
        if categorical_features is None:
            categorical_features = ['restaurant_name', 'day_of_the_week', 'dish_name']
        self.param_tuning = param_tuning
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

    def initialize_model(self, params):
        self.model = xgb.XGBRegressor(**params)

    def train_model(self):
        self.preprocessed_X_train = self.preprocessor.fit_transform(self.X_train)
        self.preprocessed_X_test = self.preprocessor.transform(self.X_test)
        self.model.fit(self.preprocessed_X_train, self.y_train)

    def tune_hyperparameters(self, param_grid):
        self.preprocessed_X_train = self.preprocessor.fit_transform(self.X_train)

        model = xgb.XGBRegressor(objective='reg:squarederror', random_state=self.random_state)
        grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error',
                                   n_jobs=-1)
        grid_search.fit(self.preprocessed_X_train, self.y_train)

        best_params = grid_search.best_params_
        print(f"Best hyperparameters: {best_params}")
        self.model = grid_search.best_estimator_

    def cross_validate(self, params):
        self.preprocessed_X = self.preprocessor.fit_transform(self.X)
        dtrain = xgb.DMatrix(self.preprocessed_X, label=self.y)
        cv_results = xgb.cv(dtrain=dtrain, params=params, nfold=self.k, num_boost_round=50, early_stopping_rounds=10,
                            metrics="rmse", as_pandas=True, seed=self.random_state)
        rmse = cv_results['test-rmse-mean'].tail(1).values[0]
        self.rmse = rmse
        print(f"Boosted CV RMSE with {self.k} folds: {cv_results['test-rmse-mean'].min()}")
        return rmse

    def tune_k_folds(self, k_values):
        rmse_scores = {}
        for k in k_values:
            self.k = k
            rmse = self.cross_validate(params=self.model.get_xgb_params())
            rmse_scores[k] = rmse

        best_k = min(rmse_scores, key=rmse_scores.get)
        print(f"Boosted Best k by lowest RMSE: {best_k}")
        self.k = best_k

        self.plot_k_tuning(rmse_scores)

    def plot_k_tuning(self, rmse_scores):
        plt.figure(figsize=(8, 6))
        plt.plot(list(rmse_scores.keys()), list(rmse_scores.values()), marker='o', linestyle='-', color='blue')
        plt.xlabel('Number of folds (k)')
        plt.ylabel('Cross-validation RMSE')
        plt.title('Boosted learner RMSE for Different k Values in k-fold CV')
        plt.xticks(list(rmse_scores.keys()))
        plt.grid(True)
        plt.savefig('output/boosted_k_tuning.png')

    def calculate_bic(self, mse):
        n = len(self.y_test)  # Numero di osservazioni nel test set
        booster = self.model.get_booster()
        num_params = sum(booster.get_fscore().values())  # Somma dei punteggi di importanza delle feature
        rss = mse * n  # Somma residua dei quadrati
        bic = n * np.log(rss / n) + num_params * np.log(n)
        return bic

    def evaluate_model(self):
        self.preprocessed_X_test = self.preprocessor.transform(self.X_test)
        y_pred = self.model.predict(self.preprocessed_X_test)
        mse = mean_squared_error(self.y_test, y_pred)
        self.rmse = np.sqrt(mse)
        self.bic = self.calculate_bic(mse)
        print(f"Boosted RMSE: {self.rmse}")
        print(f"Boosted BIC: {self.bic}")

    def save_model(self):
        if self.cross_validation:
            dump(self.model, 'output/models/boosted_regressor_cv.joblib')
            print(f'Model saved in output/models/boosted_regressor_cv.joblib')
        else:
            dump(self.model, 'output/models/boosted_regressor.joblib')
            print(f'Model saved in output/models/boosted_regressor.joblib')

    def initialize(self):
        self.load_data()
        self.preprocess()
        self.train_test_split()

        param_grid = {
            'max_depth': [3, 4, 5, 6, 7, 8],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'n_estimators': [100, 200, 300, 400, 500],
            'colsample_bytree': [0.3, 0.5, 0.7, 1],
            'subsample': [0.6, 0.8, 1.0]
        }
        if self.param_tuning:
            self.tune_hyperparameters(param_grid)
        else:
            params = {'colsample_bytree': 0.3,
                      'learning_rate': 0.2,
                      'max_depth': 3,
                      'n_estimators': 400,
                      'subsample': 0.8}
            self.initialize_model(params)


        if self.cross_validation:
            k_values = [3, 5, 10, 20]
            self.tune_k_folds(k_values)
            self.cross_validate(params=self.model.get_xgb_params())
            self.train_model()
        else:
            self.train_model()
            self.evaluate_model()



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
        """
        Initializes the BoostedRegressor class with the specified parameters, setting up the dataset and configuration
        for potential cross-validation, hyperparameter tuning, and the selection of features for the regression model.

        Parameters:
        - df (pd.DataFrame): The dataset containing information about dishes, expected to include both numerical and
          categorical features for analysis or modeling.
        - cross_validation (bool, optional): Indicates whether cross-validation should be employed in model training,
          enabling model validation across 'k' folds if set to True. Defaults to False.
        - param_tuning (bool, optional): Specifies whether hyperparameter tuning should be conducted to optimize the
          regression model. Defaults to False.
        - random_state (int, optional): A seed value for random operations to ensure reproducibility. Affects dataset
          splitting and any model initialization that involves randomness. Defaults to 42.
        - test_size (float, optional): The proportion of the dataset to allocate to the test set during the train-test
          split. Defaults to 0.2.
        - k (int, optional): The number of folds to use for cross-validation, relevant only if cross_validation is True.
          Defaults to 5.
        - categorical_features (list of str, optional): Column names in 'df' considered as categorical features. If not
          provided, defaults to ['restaurant_name', 'day_of_the_week', 'dish_name'].
        - numerical_features (list of str, optional): Column names in 'df' considered as numerical features. If not
          provided, defaults to ['latitude', 'longitude'].
        """
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
        """
       Loads and processes the dishes data from the instance's dataframe. This method specifically extracts
       geographical coordinates from the 'restaurant_location' column and adds them as separate 'latitude' and
       'longitude' columns to the dataframe. It then prepares the feature matrix 'X' with relevant columns for
       modeling and the target variable 'y' representing the preparation time of the dishes.
       """
        self.dishes_df[['latitude', 'longitude']] = self.dishes_df['restaurant_location'].apply(
            lambda loc: pd.Series(ast.literal_eval(loc))
        )
        self.X = self.dishes_df[self.categorical_features + self.numerical_features]
        self.y = self.dishes_df['preparation_time']

    def preprocess(self):
        """
        Prepares the preprocessing pipeline for the categorical features in the dataset. It initializes a
        OneHotEncoder to handle categorical variables by creating binary columns for each category and a
        ColumnTransformer to apply this encoder to the specified categorical features while leaving numerical
        features unchanged. This method sets up the 'preprocessor' attribute with the configured ColumnTransformer.
        """
        one_hot_encoder = OneHotEncoder(handle_unknown='ignore')
        self.preprocessor = ColumnTransformer(transformers=[
            ('cat', one_hot_encoder, self.categorical_features)],
            remainder='passthrough')

    def train_test_split(self):
        """
        Splits the dataset into training and testing sets. This method uses the feature matrix 'X' and the target
        variable 'y' to create training and testing subsets, with the\ size of the test set defined by the 'test_size'
        parameter and the splitting process controlled by a 'random_state' for reproducibility.

        Parameters:
            test_size (float, optional): The proportion of the dataset to include in the test split. Defaults to 0.2.
            random_state (int, optional): A seed value to ensure the reproducibility of the train-test split. Defaults to 42.
        """
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=self.test_size, random_state=self.random_state)

    def initialize_model(self, params):
        """
       Initializes the predictive model as an XGBoost regressor with the specified hyperparameters.
       The method sets the 'model' attribute of the instance to this pipeline, making it ready for training with
       the dataset.

       Parameters:
            params (dict): A dictionary of hyperparameters to configure the XGBoost regressor model.
       """
        self.model = xgb.XGBRegressor(**params)

    def train_model(self):
        """
       Trains the XGBoost model on the preprocessed training data. This method first applies the preprocessing pipeline
       to the training and testing feature sets, then fits the XGBoost model to the preprocessed training data.
       """
        self.preprocessed_X_train = self.preprocessor.fit_transform(self.X_train)
        self.preprocessed_X_test = self.preprocessor.transform(self.X_test)
        self.model.fit(self.preprocessed_X_train, self.y_train)

    def tune_hyperparameters(self, param_grid):
        """
        Performs hyperparameter tuning for the XGBoost model using grid search with cross-validation. This method applies
        the preprocessing pipeline to the training data, then searches through the specified parameter grid to find the
        best combination of parameters based on cross-validated mean squared error.

        Parameters:
        - param_grid (dict): A dictionary specifying the parameter grid to explore, where keys are parameter names (as
          expected by XGBoost) and values are lists of values to try.

        After finding the best parameters, the method updates the model to the best estimator found during the grid search.
        """
        self.preprocessed_X_train = self.preprocessor.fit_transform(self.X_train)

        model = xgb.XGBRegressor(objective='reg:squarederror', random_state=self.random_state)
        grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error',
                                   n_jobs=-1)
        grid_search.fit(self.preprocessed_X_train, self.y_train)

        best_params = grid_search.best_params_
        print(f"Best hyperparameters: {best_params}")
        self.model = grid_search.best_estimator_

    def cross_validate(self, params):
        """
        Performs cross-validation on the entire dataset using XGBoost's built-in CV function. This method is particularly
        useful for evaluating the effectiveness of the given XGBoost model parameters over 'k' folds of cross-validation.

        Parameters:
        - params (dict): A dictionary of parameters for the XGBoost model to be evaluated during cross-validation. This
          includes both the hyperparameters of the model and the cross-validation configuration (e.g., `nfold`, `metrics`).

        Returns:
        - rmse (float): The root mean square error (RMSE) metric averaged over all cross-validation folds, providing an
          estimate of the model's prediction error.
        """
        self.preprocessed_X = self.preprocessor.fit_transform(self.X)
        dtrain = xgb.DMatrix(self.preprocessed_X, label=self.y)
        cv_results = xgb.cv(dtrain=dtrain, params=params, nfold=self.k, num_boost_round=50, early_stopping_rounds=10,
                            metrics="rmse", as_pandas=True, seed=self.random_state)
        rmse = cv_results['test-rmse-mean'].tail(1).values[0]
        self.rmse = rmse
        print(f"Boosted CV RMSE with {self.k} folds: {cv_results['test-rmse-mean'].min()}")
        return rmse

    def tune_k_folds(self, k_values):
        """
        Tunes the number of folds 'k' used in cross-validation to identify the optimal 'k' that results in the lowest
        mean RMSE. It iterates over a range of 'k' values, performs cross-validation for each, and stores the mean RMSE
        for each 'k'. The method then selects the 'k' with the lowest mean RMSE, updates the instance's 'k' attribute,
        and plots the RMSE values for all tested 'k' values.

        Parameters:
            k_values (list of int): A list of 'k' values to test for finding the optimal number of folds in cross-validation.

        This method also calls `plot_cv_tuning` to visualize the performance of different 'k' values.
        """
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
        """
        Generates a plot to visualize the relationship between the number of folds used in k-fold cross-validation (k) and
        the resulting root mean square error (RMSE) for a boosted learner. This visualization aids in selecting the optimal
        number of folds that minimizes RMSE, thereby enhancing model validation.

        Parameters:
            rmse_scores (dict): A dictionary where keys are the number of folds (k) and values are RMSE scores obtained from cv

        The plot is saved to a file named 'boosted_k_tuning.png' in the 'output' directory. It displays the k values on the
        x-axis and the RMSE scores on the y-axis, with a point and line connecting each k-RMSE pair. This function is
        particularly useful for assessing how increasing the number of folds impacts the model's estimated prediction error
        and for documenting the cross-validation process's outcomes.
        """
        plt.figure(figsize=(8, 6))
        plt.plot(list(rmse_scores.keys()), list(rmse_scores.values()), marker='o', linestyle='-', color='blue')
        plt.xlabel('Number of folds (k)')
        plt.ylabel('Cross-validation RMSE')
        plt.title('Boosted learner RMSE for Different k Values in k-fold CV')
        plt.xticks(list(rmse_scores.keys()))
        plt.grid(True)
        plt.savefig('output/boosted_k_tuning.png')

    def calculate_bic(self, mse):
        """
        Calculates the Bayesian Information Criterion (BIC) for the model based on the mean squared error (MSE) of predictions.
        BIC is used to evaluate the model fit while penalizing the model complexity to prevent overfitting.

        Parameters:
            mse (float): The mean squared error of the model's predictions.

        Returns:
            bic (float): The calculated BIC value.

        The BIC is calculated using the formula: BIC = n * log(rss / n) + k * log(n), where:
        - n is the number of observations in the test set,
        - rss is the residual sum of squares, and
        - k is the number of model parameters (coefficients).
        """
        n = len(self.y_test)  # Numero di osservazioni nel test set
        booster = self.model.get_booster()
        num_params = sum(booster.get_fscore().values())  # Somma dei punteggi di importanza delle feature
        rss = mse * n  # Somma residua dei quadrati
        bic = n * np.log(rss / n) + num_params * np.log(n)
        return bic

    def evaluate_model(self):
        """
        Evaluates the model's performance on the test set. This method predicts the target variable using the test set
        features, calculates the mean squared error (MSE) and the root mean squared error (RMSE), and then calculates
        the Bayesian Information Criterion (BIC) to assess model fit and complexity.

        This method updates the instance attributes 'rmse' and 'bic' with the calculated values and prints these metrics.
        """
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



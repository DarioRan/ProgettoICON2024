import ast
from joblib import dump
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
        """
         Initializes the class instance with the given parameters, setting up the data and model configuration.

         Parameters:
         - df (pd.DataFrame): The dataset containing the dishes information. This dataframe is expected to include both
           numerical and categorical features that are relevant for the analysis or model building.
         - cross_validation (bool, optional): Determines whether cross-validation should be used when training the model.
           Defaults to False, indicating that cross-validation will not be used unless explicitly specified.
         - random_state (int, optional): A seed value to ensure reproducibility of results. This affects the randomness
           of any operations that involve randomization, such as splitting datasets or initializing model weights.
           Defaults to 42.
         - alpha (float, optional): A parameter that controls the regularization strength in models that support it,
           helping to prevent overfitting by penalizing large coefficients. Defaults to 0.2.
         - k (int, optional): The number of folds to use for cross-validation, if enabled. This parameter is only relevant
           when cross_validation is set to True. Defaults to 5.
         - categorical_features (list of str, optional): A list of column names from 'df' that are to be treated as
           categorical features. If not provided, defaults to ['restaurant_name', 'day_of_the_week', 'dish_name'].
         - numerical_features (list of str, optional): A list of column names from 'df' that are to be treated as
           numerical features. If not provided, defaults to ['latitude', 'longitude'].
         """
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
        """
       Loads and processes the dishes data from the instance's dataframe. This method specifically extracts
       geographical coordinates from the 'restaurant_location' column and adds them as separate 'latitude' and
       'longitude' columns to the dataframe. It then prepares the feature matrix 'X' with relevant columns for
       modeling and the target variable 'y' representing the preparation time of the dishes.
       """
        self.dishes_df[['latitude', 'longitude']] = self.dishes_df['restaurant_location'].apply(
            lambda loc: pd.Series(ast.literal_eval(loc))
        )
        self.X = self.dishes_df[['restaurant_name', 'day_of_the_week', 'dish_name', 'latitude', 'longitude']]
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

    def train_test_split(self, test_size=0.2, random_state=42):
        """
        Splits the dataset into training and testing sets. This method uses the feature matrix 'X' and the target
        variable 'y' to create training and testing subsets, with the size of the test set defined by the 'test_size'
        parameter and the splitting process controlled by a 'random_state' for reproducibility.

        Parameters:
            test_size (float, optional): The proportion of the dataset to include in the test split. Defaults to 0.2.
            random_state (int, optional): A seed value to ensure the reproducibility of the train-test split. Defaults to 42.
        """
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=test_size, random_state=random_state)

    def initialize_model(self):
        """
        Initializes the predictive model as a pipeline that includes preprocessing and a regression algorithm.
        The pipeline is composed of two main steps: preprocessing the data with the previously defined 'preprocessor'
        and applying a Lasso regression model. The Lasso model uses the 'alpha' parameter defined in the class
        initialization to control regularization strength, which helps in reducing overfitting by penalizing large
        coefficients in the model.

        The method sets the 'model' attribute of the instance to this pipeline, making it ready for training with
        the dataset.
        """
        self.model = Pipeline(steps=[
            ('preprocessor', self.preprocessor),
            ('regressor', Lasso(alpha=self.alpha))
        ])

    def train_model(self):
        """
        Trains the predictive model using the training data. This method fits the model pipeline to the training
        dataset, which involves applying the preprocessing steps to the feature matrix 'X_train' and then fitting
        the Lasso regressor to the processed features and target variable 'y_train'. The fitting process adjusts
        the model's parameters to minimize the prediction error, preparing the model for making predictions on new
        or unseen data.

        This method does not return any value but updates the 'model' attribute with the trained model.
        """
        self.model.fit(self.X_train, self.y_train)

    def cross_validate(self):
        """
        Performs cross-validation on the model using the specified scoring metric, calculates the root mean square error (RMSE)
        from the cross-validation scores, and prints the mean RMSE with its standard deviation.

        Parameters:
            scoring (str, optional): The scoring metric to use for evaluating model performance during cross-validation.

        Returns:
            rmse_scores (np.ndarray): An array of the RMSE scores for each fold of the cross-validation.

        This method updates the instance's 'rmse' attribute with the mean RMSE calculated from cross-validation scores.
        """
        lasso_scores = cross_val_score(self.model, self.X, self.y, cv=self.k, scoring='neg_mean_squared_error')
        lasso_rmse_scores = np.sqrt(-lasso_scores)
        self.rmse = lasso_rmse_scores.mean()
        print(f'Lasso Cross-validation RMSE: {lasso_rmse_scores.mean()} (± {lasso_rmse_scores.std()})')

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
        """
       Generates a plot of cross-validation RMSE values for different numbers of folds 'k'. The plot helps in visualizing
       how the choice of 'k' affects model validation performance.

       Parameters:
            k_fold_scores (dict): A dictionary with 'k' values as keys and their corresponding mean RMSE as values.

       The method creates and saves a plot that shows the relationship between the number of folds in cross-validation
       and the RMSE, aiding in the selection of an optimal 'k'. The plot is saved to 'output/linear_k_tuning.png'.
       """
        # Plotting the RMSE for different k values
        plt.figure(figsize=(8, 6))
        plt.plot(list(k_fold_scores.keys()), list(k_fold_scores.values()), marker='o', linestyle='-', color='b')
        plt.xlabel('Number of folds (k)')
        plt.ylabel('Cross-validation RMSE')
        plt.title('Lasso RMSE for Different k Values in Cross-validation')
        plt.xticks(list(k_fold_scores.keys()))
        plt.grid(True)
        plt.savefig('output/lasso_k_tuning.png')


    def tune_hyperparameters(self, param_grid):
        """
        Tunes the hyperparameters of the model using grid search with cross-validation. This method searches through
        the specified parameter grid to find the combination of parameters that minimizes the mean squared error (MSE)
        of predictions.

        Parameters:
            param_grid (dict): A dictionary where keys are parameter and values are lists of parameter settings to try as values.

        Returns:
            best_params (dict): The parameter setting that gave the best results on the hold out data.

        After finding the best parameters, this method updates the model with the best estimator found and plots
        the relationship between the alpha parameter and the MSE using the `plot_alpha_loss` method.
        """
        grid_search = GridSearchCV(self.model, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
        grid_search.fit(self.X_train, self.y_train)

        best_params = grid_search.best_params_
        print(f"Best parameters: {best_params}")

        alphas = grid_search.cv_results_['param_regressor__alpha']
        mean_scores = -grid_search.cv_results_['mean_test_score']

        self.plot_alpha_loss(alphas, mean_scores)

        best_estimator = grid_search.best_estimator_
        cv_data= pd.DataFrame(grid_search.cv_results_)
        plot_data = cv_data[['param_regressor__alpha', 'mean_test_score']]
        self.model = best_estimator
        return best_params

    def plot_alpha_loss(self, alphas, mean_scores):
        """
        Plots the mean squared error (MSE) against different values of the alpha parameter used in regularization.
        This visualization helps in understanding how the choice of alpha affects the model's performance.

        Parameters:
            alphas (np.ndarray): An array of alpha values that were evaluated during hyperparameter tuning.
            mean_scores (np.ndarray): An array of mean squared errors corresponding to each alpha value.

        The plot is saved to a file named 'lasso_mse_vs_alpha.png' in the 'output' directory. This function
        aims to provide insights into the trade-off between model complexity and model performance.
        """
        plt.figure(figsize=(10, 6))
        plt.plot(alphas, mean_scores, marker='o', linestyle='-', color='blue')
        plt.xlabel(r'$\alpha$')
        plt.ylabel('Mean Squared Error')
        plt.title('Lasso MSE vs. Alpha')
        plt.grid(True)
        plt.savefig('output/lasso_mse_vs_alpha.png')

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
        k = len(self.model.named_steps[
                    'regressor'].coef_)
        rss = mse * n  # Somma residua dei quadrati
        bic = n * np.log(rss / n) + k * np.log(n)
        return bic

    def evaluate_model(self):
        """
        Evaluates the model's performance on the test set. This method predicts the target variable using the test set
        features, calculates the mean squared error (MSE) and the root mean squared error (RMSE), and then calculates
        the Bayesian Information Criterion (BIC) to assess model fit and complexity.

        This method updates the instance attributes 'rmse' and 'bic' with the calculated values and prints these metrics.
        """
        y_pred = self.model.predict(self.X_test)
        mse = mean_squared_error(self.y_test, y_pred)

        self.rmse = np.sqrt(mse)
        self.bic = self.calculate_bic(mse)
        print(f'Lasso RMSE: {self.rmse}')
        print(f'Lasso BIC: {self.bic}')

    def save_model(self):
        """
        Saves the trained model to a file.
        """
        if self.cross_validation:
            dump(self.model, 'output/models/lasso_regressor_cv.joblib')
            print(f'Modeel saved in output/models/lasso_regressor_cv.joblib')
        else:
            dump(self.model, 'output/models/lasso_regressor.joblib')
            print(f'Modeel saved in output/models/lasso_regressor.joblib')

    def initialize(self):
        self.load_data()
        self.preprocess()
        self.train_test_split()
        self.initialize_model()
        param_grid = {
            'regressor__alpha': [0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.001, 0.0015, 0.002, 0.01]
        }
        self.train_model()

        if self.cross_validation:
            k_values = [3, 5, 10, 20]
            self.tune_k_folds(k_values)
            self.tune_hyperparameters(param_grid)
            self.cross_validate()
        else:
            self.evaluate_model()


    def predict(self, new_data):
        return self.model.predict(new_data)

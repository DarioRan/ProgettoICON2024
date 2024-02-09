import ast
import itertools

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score


class PyTorchRegressor(nn.Module):
    def __init__(self, input_shape, dropout):
        super(PyTorchRegressor, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_shape, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.network(x)


class NeuralRegressor:
    def __init__(self, df, cross_validation=False, random_state=42, test_size=0.2,  k=5, categorical_features=None, numerical_features=None):
        self.dishes_df = df
        self.random_state = random_state
        self.test_size = test_size
        if numerical_features is None:
            numerical_features = ['latitude', 'longitude']
        if categorical_features is None:
            categorical_features = ['restaurant_name', 'day_of_the_week', 'dish_name']
        self.categorical_features = categorical_features
        self.numerical_features = numerical_features
        self.model = None
        self.cross_validation = cross_validation
        self.initialize()

    def load_data(self):
        # Convert 'restaurant_location' from string to tuple of floats
        self.dishes_df[['latitude', 'longitude']] = self.dishes_df['restaurant_location'].apply(
            lambda loc: pd.Series(ast.literal_eval(loc))
        )
        self.X = self.dishes_df[self.categorical_features + self.numerical_features]
        self.y = self.dishes_df['preparation_time'].values.reshape(-1, 1)

    def preprocess(self):
        # Create a column transformer with OneHotEncoder and StandardScaler
        self.preprocessor = ColumnTransformer(transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), self.categorical_features),
            ('num', StandardScaler(), self.numerical_features)
        ])
        # Fit the preprocessor and transform the feature data
        self.X = self.preprocessor.fit_transform(self.X)

    def train_test_split(self):
        # Split the data into training and test sets
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=self.test_size, random_state=self.random_state)

    def initialize_model(self, dropout):
        # Determine the input shape for the PyTorch model
        input_shape = self.X_train.shape[1]
        # Instantiate the PyTorch model
        self.model = PyTorchRegressor(input_shape, dropout)

    def train_model(self, epochs=100, batch_size=10):
        # Convert the data to PyTorch tensors

        X_train_tensor = torch.tensor(self.X_train.todense().astype(np.float32))
        y_train_tensor = torch.tensor(self.y_train.astype(np.float32))

        # Create a DataLoader to handle batching of data
        train_data = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)

        # Define the loss function and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters())

        # Train the model
        self.model.train()  # Set the model to training mode
        for epoch in range(epochs):
            for batch_idx, (data, target) in enumerate(train_loader):
                optimizer.zero_grad()  # Clear the gradients
                output = self.model(data)  # Perform a forward pass
                loss = criterion(output, target)  # Compute the loss
                loss.backward()  # Perform a backward pass
                optimizer.step()  # Update the weights
            print(f'Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}')

    def tune_hyperparameters(self, param_grid):
        # Convert the data to PyTorch tensors
        X_train_tensor = torch.tensor(self.X_train.todense().astype(np.float32))
        y_train_tensor = torch.tensor(self.y_train.astype(np.float32))

        # Create a DataLoader to handle batching of data
        train_data = TensorDataset(X_train_tensor, y_train_tensor)

        loss_by_hyperparam = {k: {} for k in param_grid}
        loss_by_combination = {}

        best_loss = float('inf')
        best_params = {}
        best_epoch = -1

        for params in itertools.product(*param_grid.values()):
            hyperparams = dict(zip(param_grid.keys(), params))
            print(f"Testing hyperparameters: {hyperparams}")
            hyperparams_tuple = tuple(hyperparams.items())

            train_loader = DataLoader(dataset=train_data, batch_size=hyperparams['batch_size'], shuffle=True)

            # Initialize the model
            self.initialize_model(hyperparams['dropout'])

            # Define the loss function and optimizer with the current hyperparameters
            criterion = nn.MSELoss()
            optimizer = optim.Adam(self.model.parameters(), lr=hyperparams['lr'])

            epoch_losses = []  # Lista per tenere traccia delle perdite per ogni epoca

            for epoch in range(hyperparams['epochs']):
                self.model.train()
                epoch_loss = 0.0

                for batch_idx, (data, target) in enumerate(train_loader):
                    optimizer.zero_grad()
                    output = self.model(data)
                    loss = criterion(output, target)
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()

                epoch_loss /= len(train_loader)
                epoch_losses.append(epoch_loss)  # Aggiungi la perdita media dell'epoca
                print(f"Epoch {epoch + 1}/{hyperparams['epochs']}, Loss: {epoch_loss}")

                if epoch_loss < best_loss:
                    best_loss = epoch_loss
                    best_params = hyperparams
                    best_epoch = epoch

            avg_epoch_loss = np.mean(epoch_losses)  # Calcola la perdita media per questa combinazione di iperparametri
            print(f"Average loss for combination {hyperparams}: {avg_epoch_loss}")
            loss_by_combination[hyperparams_tuple] = avg_epoch_loss  # Aggiorna il dizionario con la perdita media

            for key, value in hyperparams.items():
                if value not in loss_by_hyperparam[key]:
                    loss_by_hyperparam[key][value] = []
                loss_by_hyperparam[key][value].append(avg_epoch_loss)

        self.plot_hyperparameter_tuning_results(param_grid, loss_by_hyperparam, loss_by_combination)
        self.best_params = best_params
        print(f'Overall best params: {best_params}, Best loss: {best_loss}')

    def calculate_num_params(self):
        model_parameters = filter(lambda p: p.requires_grad, self.model.parameters())
        params_count = sum([np.prod(p.size()) for p in model_parameters])
        return params_count

    def calculate_bic(self, mse, num_params):
        n = len(self.y_test)  # Number of data points
        rss = mse * n  # Residual sum of squares
        bic = n * np.log(rss / n) + num_params * np.log(n)
        return bic

    def cross_validate(self, cv=5, scoring='neg_mean_squared_error'):
        scores = cross_val_score(self.model, self.X, self.y, cv=cv, scoring=scoring)
        rmse_scores = np.sqrt(-scores)
        print(f'Neural NetCross-validation RMSE: {rmse_scores.mean()} (± {rmse_scores.std()})')
        self.rmse = rmse_scores.mean()
        return rmse_scores

    def evaluate_model(self):
        # Convert the test data to PyTorch tensors
        X_test_tensor = torch.tensor(self.X_test.todense().astype(np.float32))
        y_test_tensor = torch.tensor(self.y_test.astype(np.float32))

        # Evaluate the model
        self.model.eval()  # Set the model to evaluation mode
        with torch.no_grad():  # Disable gradient calculation
            predictions = self.model(X_test_tensor)
            mse = torch.mean((predictions - y_test_tensor) ** 2)
            rmse = torch.sqrt(mse)
            print(f'Neural Net RMSE: {rmse.item()}')
            self.rmse = rmse.item()
        num_params = self.calculate_num_params()
        bic = self.calculate_bic(mse.item(), num_params)
        self.bic = bic
        print(f'Neural Net BIC: {bic}')

    def predict(self, new_data):
        # Preprocess the new data
        new_data = self.preprocessor.transform(new_data)
        # Convert the new data to a PyTorch tensor
        new_data_tensor = torch.tensor(new_data.todense().astype(np.float32))

        # Make predictions using the model
        self.model.eval()  # Set the model to evaluation mode
        with torch.no_grad():  # Disable gradient calculation
            predictions = self.model(new_data_tensor)

        return predictions

    def plot_hyperparameter_tuning_results(self, param_grid, loss_by_hyperparam, loss_by_combination):
        for param, value_losses in loss_by_hyperparam.items():
            plt.figure()
            unique_values = sorted(value_losses.keys())
            avg_losses = [np.mean(value_losses[value]) for value in unique_values]
            plt.plot(unique_values, avg_losses, marker='o')
            plt.title(f'Loss by {param}')
            plt.xlabel(param)
            plt.ylabel('Average loss')
            plt.grid(True)
            plt.savefig(f'loss_by_{param}.png')

        plt.figure(figsize=(10, 6))
        sorted_combinations = sorted(loss_by_combination.items(), key=lambda x: x[1])[:10]
        combinations = [str(c[0]) for c in sorted_combinations]
        losses = [c[1] for c in sorted_combinations]
        plt.bar(range(len(combinations)), losses)
        plt.xticks(range(len(combinations)), combinations, rotation=90)
        plt.title('Top 10 Hyperparameter Combinations by Average Loss')
        plt.xlabel('Hyperparameter Combinations')
        plt.ylabel('Average Loss')
        plt.tight_layout()
        plt.grid(True)
        plt.savefig('top_10_hyperparameter_combinations.png')

    def train_model_with_best_params(self):
        X_train_tensor = torch.tensor(self.X_train.todense().astype(np.float32))
        y_train_tensor = torch.tensor(self.y_train.astype(np.float32))

        train_data = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(dataset=train_data, batch_size=self.best_params['batch_size'], shuffle=True)

        self.initialize_model(self.best_params['dropout'])

        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.best_params['lr'])

        self.model.train()
        for epoch in range(self.best_params['epochs']):
            for batch_idx, (data, target) in enumerate(train_loader):
                optimizer.zero_grad()
                output = self.model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
            print(f'Epoch {epoch + 1}/{self.best_params["epochs"]}, loss: {loss.item()}')

    def tune_k_folds(self, k_values):
        k_fold_scores = {}
        for k in k_values:
            scores = cross_val_score(self.model, self.X, self.y, cv=k, scoring='neg_mean_squared_error')
            rmse_scores = np.sqrt(-scores)
            mean_rmse = rmse_scores.mean()
            k_fold_scores[k] = mean_rmse
            print(f'Neural Net {k}-fold CV RMSE: {mean_rmse} (± {rmse_scores.std()})')

        best_k = min(k_fold_scores, key=k_fold_scores.get)
        print(f'Neural Net Best k by lowest RMSE: {best_k}')
        self.k = best_k


    def initialize(self):
        self.load_data()
        self.preprocess()
        self.train_test_split()
        param_grid = {
            'lr': [1e-2, 1e-1],
            'batch_size': [8, 16],
            'epochs': [100, 150],
            'dropout': [0, 0.1]
        }
        self.tune_hyperparameters(param_grid)
        self.train_model_with_best_params()

        if self.cross_validation:
            k_values = [3, 5, 10, 15, 20]
            self.tune_k_folds(k_values)
            #sbagliata di regola
            self.cross_validate()
        else:
            self.evaluate_model()


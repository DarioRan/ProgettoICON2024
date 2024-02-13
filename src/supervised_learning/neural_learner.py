import ast
import itertools
from joblib import dump
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
    def __init__(self, df, param_tuning=True, random_state=42, test_size=0.2, categorical_features=None, numerical_features=None):
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
        self.param_tuning = param_tuning
        self.initialize()

    def load_data(self):
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
        print(self.X.shape)
        print(self.X)

    def train_test_split(self):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=self.test_size, random_state=self.random_state)

    def initialize_model(self, dropout):
        input_shape = self.X_train.shape[1]
        self.model = PyTorchRegressor(input_shape, dropout)

    def train_model(self, epochs=100, batch_size=10):

        X_train_tensor = torch.tensor(self.X_train.todense().astype(np.float32))
        y_train_tensor = torch.tensor(self.y_train.astype(np.float32))

        train_data = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)

        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters())

        self.model.train()  # Set the model to training mode
        for epoch in range(epochs):
            for batch_idx, (data, target) in enumerate(train_loader):
                optimizer.zero_grad()  # Clear the gradients
                output = self.model(data)  # Perform a forward pass
                loss = criterion(output, target)  # Compute the loss
                loss.backward()  # Perform a backward pass
                optimizer.step()  # Update the weights
            print(f'Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}')

    def get_mean_error_validation(self, valid_loader, criterion):
        epoch_mean_loss_validation = 0.0
        self.model.eval()
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(valid_loader):
                output = self.model(data)
                loss = criterion(output, target)
                epoch_mean_loss_validation += loss.item()
        epoch_mean_loss_validation /= len(valid_loader)
        print(f"Validation loss: {epoch_mean_loss_validation}")
        return epoch_mean_loss_validation

    def tune_hyperparameters(self, param_grid):
        # Convert the data to PyTorch tensors
        X_train_tensor = torch.tensor(self.X_train.todense().astype(np.float32))
        y_train_tensor = torch.tensor(self.y_train.astype(np.float32))

        train_data = TensorDataset(X_train_tensor, y_train_tensor)

        loss_by_hyperparam = {k: {} for k in param_grid}
        loss_by_combination = {}

        valid_loss_by_hyperparam = {k: {} for k in param_grid}
        valid_loss_by_combination = {}

        best_loss = float('inf')
        best_params = {}
        best_epoch = -1

        epoch_losses_train = []  # Lista per tenere traccia delle perdite per ogni epoca
        epoch_losses_valid = []


        for params in itertools.product(*param_grid.values()):
            hyperparams = dict(zip(param_grid.keys(), params))
            print(f"Testing hyperparameters: {hyperparams}")
            hyperparams_tuple = tuple(hyperparams.items())

            #crea set validazione
            X_train, X_valid, y_train, y_valid = train_test_split(
                self.X_train, self.y_train, test_size=0.2, random_state=self.random_state)
            X_train_tensor = torch.tensor(X_train.todense().astype(np.float32))
            y_train_tensor = torch.tensor(y_train.astype(np.float32))
            train_data = TensorDataset(X_train_tensor, y_train_tensor)

            train_loader = DataLoader(dataset=train_data, batch_size=hyperparams['batch_size'], shuffle=True)

            X_valid_tensor = torch.tensor(X_valid.todense().astype(np.float32))
            y_valid_tensor = torch.tensor(y_valid.astype(np.float32))
            validation_data = TensorDataset(X_valid_tensor, y_valid_tensor)
            validation_loader = DataLoader(dataset=validation_data, batch_size=hyperparams['batch_size'], shuffle=True)


            # Initialize the model
            self.initialize_model(hyperparams['dropout'])

            # Define the loss function and optimizer with the current hyperparameters
            criterion = nn.MSELoss()
            optimizer = optim.Adam(self.model.parameters(), lr=hyperparams['lr'], weight_decay=hyperparams['weight_decay'])

            epoch_losses_train = []  # Lista per tenere traccia delle perdite per ogni epoca
            #epoch_losses_valid = []

            for epoch in range(hyperparams['epochs']):
                self.model.train()
                epoch_loss = 0.0
                epoch_loss_validation=0.0

                for batch_idx, (data, target) in enumerate(train_loader):
                    optimizer.zero_grad()
                    output = self.model(data)
                    loss = criterion(output, target)
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()

                mean_validation_loss = self.get_mean_error_validation(validation_loader, criterion)
                epoch_losses_valid.append(mean_validation_loss)
                epoch_loss /= len(train_loader)
                epoch_losses_train.append(epoch_loss)  # Aggiungi la perdita media dell'epoca
                print(f"Epoch {epoch + 1}/{hyperparams['epochs']}, Loss: {epoch_loss}")

                if epoch_loss < best_loss:
                    best_loss = epoch_loss
                    best_params = hyperparams
                    best_epoch = epoch

            avg_epoch_loss = np.mean(epoch_losses_train)  # Calcola la perdita media per questa combinazione di iperparametri
            avg_epoch_valid_loss = np.mean(epoch_losses_valid)
            print(f"Average train loss for combination {hyperparams}: {avg_epoch_loss}")
            print(f"Average validation loss for combination {hyperparams}: {avg_epoch_valid_loss}")
            loss_by_combination[hyperparams_tuple] = avg_epoch_loss  # Aggiorna il dizionario con la perdita media

            for key, value in hyperparams.items():
                if value not in valid_loss_by_hyperparam[key]:
                    valid_loss_by_hyperparam[key][value] = []
                valid_loss_by_hyperparam[key][value].append(avg_epoch_loss)

                if value not in loss_by_hyperparam[key]:
                    loss_by_hyperparam[key][value] = []
                loss_by_hyperparam[key][value].append(avg_epoch_loss)

        self.plot_hyperparameter_tuning_results(loss_by_hyperparam, loss_by_combination)
        self.best_params = best_params
        #plot loss train and validation for every epoch for the best combination

        print(f'Overall best params: {best_params}, Best loss: {best_loss}')

    def calculate_num_params(self):
        model_parameters = filter(lambda p: p.requires_grad, self.model.parameters())
        params_count = sum([np.prod(p.size()) for p in model_parameters])
        return params_count

    """
    def cross_validate_best_params(self):
    #cross validate with the best hyperparameters and output the average loss by epoch

        epoch_losses_valid=[]
        epoch_losses_train = []
        self.model=PyTorchRegressor(self.X_train.shape[1], self.best_params['dropout'])


       
        X_train_tensor = torch.tensor(X_train.todense().astype(np.float32))
        y_train_tensor = torch.tensor(y_train.astype(np.float32))
        train_data = TensorDataset(X_train_tensor, y_train_tensor)

        train_loader = DataLoader(dataset=train_data, batch_size=self.best_params['batch_size'], shuffle=True)

        

        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.best_params['lr'])

        self.model.train()
        for epoch in range(self.best_params['epochs']):
            epoch_loss = 0.0
            epoch_losses = []
            for batch_idx, (data, target) in enumerate(train_loader):
                optimizer.zero_grad()
                output = self.model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            epoch_loss /= len(train_loader)
            mean_validation_loss = self.get_mean_error_validation(validation_loader, criterion)
            epoch_losses_valid.append(mean_validation_loss)
            epoch_losses_train.append(epoch_loss)  # Aggiungi la perdita media dell'epoca
            print(f"Epoch {epoch + 1}/{self.best_params['epochs']}, Loss: {epoch_loss}")

        #save plot valid loss and train loss for every epoch
        print(epoch_losses_valid)
        print(epoch_losses_train)
        plt.plot(range(self.best_params['epochs']), epoch_losses_train, label='Train loss')
        plt.plot(range(self.best_params['epochs']), epoch_losses_valid, label='Validation loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Train and Validation Loss by Epoch')
        plt.legend()
        plt.grid(True)
        plt.savefig('output/train_validation_loss_best params.png')
        """




    def predict(self, new_data):
        new_data = self.preprocessor.transform(new_data)
        new_data_tensor = torch.tensor(new_data.todense().astype(np.float32))

        self.model.eval()
        with torch.no_grad():
            predictions = self.model(new_data_tensor)
        return predictions

    def plot_hyperparameter_tuning_results(self, loss_by_hyperparam, loss_by_combination):
        for param, value_losses in loss_by_hyperparam.items():
            plt.figure()
            unique_values = sorted(value_losses.keys())
            avg_losses = [np.mean(value_losses[value]) for value in unique_values]
            plt.plot(unique_values, avg_losses, marker='o')
            plt.title(f'Loss by {param}')
            plt.xlabel(param)
            plt.ylabel('Average loss')
            plt.grid(True)
            plt.savefig(f'output/loss_by_{param}.png')

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
        plt.savefig('output/top_10_hyperparameter_combinations.png')
        plt.close()

    def train_model_with_best_params(self):

        """X_train, X_valid, y_train, y_valid = train_test_split(
            self.X_train, self.y_train, test_size=0.2, random_state=self.random_state)"""

        X_train_tensor = torch.tensor(self.X_train.todense().astype(np.float32))
        y_train_tensor = torch.tensor(self.y_train.astype(np.float32))

        train_data = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(dataset=train_data, batch_size=self.best_params['batch_size'], shuffle=True)

        """X_valid_tensor = torch.tensor(X_valid.todense().astype(np.float32))
        y_valid_tensor = torch.tensor(y_valid.astype(np.float32))
        validation_data = TensorDataset(X_valid_tensor, y_valid_tensor)
        validation_loader = DataLoader(dataset=validation_data, batch_size=self.best_params['batch_size'], shuffle=True)
        """

        X_test_tensor = torch.tensor(self.X_test.todense().astype(np.float32))
        y_test_tensor = torch.tensor(self.y_test.astype(np.float32))
        test_data = TensorDataset(X_test_tensor, y_test_tensor)
        test_loader = DataLoader(dataset=test_data, batch_size=self.best_params['batch_size'], shuffle=True)

        self.initialize_model(self.best_params['dropout'])

        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.best_params['lr'])

        self.model.train()

        epoch_losses_test = []
        epoch_losses_train = []
        for epoch in range(self.best_params['epochs']):
            epoch_loss = 0.0
            for batch_idx, (data, target) in enumerate(train_loader):
                optimizer.zero_grad()
                output = self.model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            epoch_loss /= len(train_loader)
            mean_validation_loss = self.get_mean_error_validation(test_loader, criterion)

            epoch_losses_test.append(mean_validation_loss)
            epoch_losses_train.append(epoch_loss)
            print(f'Epoch {epoch + 1}/{self.best_params["epochs"]}, loss: {epoch_loss}')

        plt.plot(range(self.best_params['epochs']), epoch_losses_train, label='Train loss')
        plt.plot(range(self.best_params['epochs']), epoch_losses_test, label='Test loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Train and Test Loss by Epoch')
        plt.legend()
        plt.grid(True)
        plt.savefig('output/train_test_loss_best params.png')

    def calculate_bic(self, mse):
        n = len(self.y_test)
        k = self.calculate_num_params()
        rss = mse * n
        bic = n * np.log(rss / n) + k * np.log(n)
        return bic

    def evaluate_model(self):
        X_test_tensor = torch.tensor(self.X_test.todense().astype(np.float32))
        y_test_tensor = torch.tensor(self.y_test.astype(np.float32))

        self.model.eval()
        with torch.no_grad():
            predictions = self.model(X_test_tensor)
            mse = torch.mean((predictions - y_test_tensor) ** 2)
            rmse = torch.sqrt(mse)
            print(f'Neural Net RMSE: {rmse.item()}')
            self.rmse = rmse.item()
        bic = self.calculate_bic(mse.item())
        self.bic = bic
        print(f'Neural Net BIC: {bic}')

    def save_model(self):
        if self.param_tuning:
            dump(self.model, 'output/models/neural_regressor_cv.joblib')
            print(f'Modeel saved in output/models/neural_regressor_cv.joblib')
        else:
            dump(self.model, 'output/models/neural_regressor.joblib')
            dump(self.preprocessor, 'output/models/preprocessor.joblib')
            print(f'Modeel saved in output/models/neural_regressor.joblib')

    def initialize(self):
        self.load_data()
        self.preprocess()
        self.train_test_split()

        param_grid = {
            'lr': [1e-2, 1e-1],
            'batch_size': [8, 10],
            'epochs': [100, 150, 500],
            'dropout': [0, 0.01],
            'weight_decay': [0, 0.01]
        }

        if self.param_tuning:
            self.tune_hyperparameters(param_grid)
            #self.cross_validate_best_params()
        else:
            self.best_params = {
                'lr': 0.01,
                'batch_size': 10,
                'epochs': 500,
                'dropout': 0,
                'weight_decay': 0
            }

        self.train_model_with_best_params()


        self.evaluate_model()


import ast
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, GridSearchCV


class PyTorchRegressor(nn.Module):
    def __init__(self, input_shape):
        super(PyTorchRegressor, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_shape, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.network(x)


class NeuralRegressor:
    def __init__(self, df, random_state=42, test_size=0.2, categorical_features=None, numerical_features=None):
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

    def initialize_model(self):
        # Determine the input shape for the PyTorch model
        input_shape = self.X_train.shape[1]
        # Instantiate the PyTorch model
        self.model = PyTorchRegressor(input_shape)

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
        X_train_tensor = torch.tensor(self.X_train.todense().astype(np.float32))
        y_train_tensor = torch.tensor(self.y_train.astype(np.float32))

        model = PyTorchRegressor(input_shape=self.X_train.shape[1])

        grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)

        grid_search.fit(X_train_tensor, y_train_tensor)

        best_params = grid_search.best_params_
        print(f'Migliori parametri: {best_params}')

        self.model = grid_search.best_estimator_

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

    def initialize(self):
        self.load_data()
        self.preprocess()
        self.train_test_split()

        self.initialize_model()
        self.train_model()
        #questi 2 o questo sotto

        #self.tune_hyperparameters()
        self.evaluate_model()

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src.supervised_learning.utils.preprocessing import retrieve_dataframe


class LinearRegressor:
    def __init__(self, df, random_state=42, test_size=0.2):
        self.random_state = random_state
        self.test_size = test_size
        self.model = None
        self.categorical_features = ['restaurant_name', 'day_of_the_week', 'dish_name']
        self.initialize(df)

    def load_data(self, df):
        self.dishes_df = retrieve_dataframe(df)
        self.X = self.dishes_df[['restaurant_name', 'day_of_the_week', 'dish_name']]
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

    def evaluate_model(self):
        y_pred = self.model.predict(self.X_test)
        rmse = np.sqrt(mean_squared_error(self.y_test, y_pred))
        print(f'Basic RMSE: {rmse}')

    def initialize(self, df):
        self.load_data(df)
        self.preprocess()
        self.train_test_split()
        self.initialize_model()
        self.train_model()
        self.evaluate_model()

    def predict(self, new_data):
        return self.model.predict(new_data)

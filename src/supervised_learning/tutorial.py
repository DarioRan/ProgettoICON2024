import pandas as pd

from src.supervised_learning.ridge_learner import RidgeRegressor
from src.supervised_learning.linear_lerner import LinearRegressor
from src.supervised_learning.lasso_learner import LassoRegressor

df = pd.read_csv('../../dataset/food_order_final.csv')

linear_regressor = LinearRegressor(df)

ridge_regressor = RidgeRegressor(df, alpha=0.2, k=5)

lasso_regressor = LassoRegressor(df, alpha=0.2, k=5)

new_data = pd.DataFrame([
    ['Pylos', 'Weekday', 'Armenian Losh Kebab'],
    ['Pylos', 'Weekday', 'Greek Salad']
], columns=['restaurant_name', 'day_of_the_week', 'dish_name'])
predictions = linear_regressor.predict(new_data)

print(predictions)

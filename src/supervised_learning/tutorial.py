import pandas as pd

from src.supervised_learning.ridge_learner import RidgeRegressor
from src.supervised_learning.linear_lerner import LinearRegressor
from src.supervised_learning.lasso_learner import LassoRegressor
from src.unsupervised_learning.kmn_learner import KMNClusterer

df = pd.read_csv('../../dataset/dishes_df.csv')

knn_clusterer = KMNClusterer(df, n_clusters=10, random_state=42)
df_with_clusters = knn_clusterer.clusterize()

linear_regressor = LinearRegressor(df)

ridge_regressor = RidgeRegressor(df, alpha=0.1, k=5)

lasso_regressor = LassoRegressor(df, alpha=5, k=5)

new_data = pd.DataFrame([
    ['Pylos', 'Weekday', 'Armenian Losh Kebab', '40.7261637', '-73.9840813'],
    ['Pylos', 'Weekday', 'Greek Salad', '40.7261637', '-73.9840813']
], columns=['restaurant_name', 'day_of_the_week', 'dish_name', 'latitude', 'longitude'])

pred1 = linear_regressor.predict(new_data)

print(pred1)


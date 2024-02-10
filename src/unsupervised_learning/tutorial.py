import pandas as pd

from src.supervised_learning.linear_learner import LinearRegressor
from src.unsupervised_learning.kmn_learner import KMNClusterer

df = pd.read_csv('../../dataset/dishes_df.csv')

kmn_clusterer = KMNClusterer(df, n_clusters=10, random_state=42)
kmn_clusterer.save_model()
"""df_with_clusters = knn_clusterer.clusterize()
linear_regressor_with_ing_feature = LinearRegressor(df_with_clusters, False, False)
linear_regressor_with_ing_feature.save_model()"""

"""new_data = pd.DataFrame([
    ['Pylos', 'Weekday', 'Armenian Losh Kebab', '40.7261637', '-73.9840813'],
    ['Pylos', 'Weekday', 'Greek Salad', '40.7261637', '-73.9840813']
], columns=['restaurant_name', 'day_of_the_week', 'dish_name', 'latitude', 'longitude'])
clusters = knn_clusterer.predict(new_data)
new_data['cluster'] = clusters
pred1 = linear_regressor_with_ing_feature.predict(new_data)
print(pred1)"""
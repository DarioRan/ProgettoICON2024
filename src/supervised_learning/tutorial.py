import pandas as pd
from matplotlib import pyplot as plt

from src.supervised_learning.boosted_learner import BoostedRegressor
from src.supervised_learning.neural_learner import NeuralRegressor
from src.supervised_learning.ridge_learner import RidgeRegressor
from src.supervised_learning.linear_learner import LinearRegressor
from src.supervised_learning.lasso_learner import LassoRegressor
from src.unsupervised_learning.kmn_learner import KMNClusterer

df = pd.read_csv('../../dataset/dishes_df.csv')

boosted_regressor = BoostedRegressor(df, False, False)
boosted_regressor_with_cv = BoostedRegressor(df, True, False)

linear_regressor = LinearRegressor(df)
linear_regressor_with_cv = LinearRegressor(df, True)

ridge_regressor = RidgeRegressor(df, alpha=0.1)
ridge_regressor_with_cv = RidgeRegressor(df, True, alpha=0.1)

lasso_regressor = LassoRegressor(df,  alpha=0.1)
lasso_regressor_with_cv = LassoRegressor(df, True, alpha=0.1)

knn_clusterer = KMNClusterer(df, n_clusters=10, random_state=42)
df_with_clusters = knn_clusterer.clusterize()

#implementare nl cv
neural_regressor = NeuralRegressor(df, False, False)

linear_regressor_bic = linear_regressor.bic
ridge_regressor_bic = ridge_regressor.bic
lasso_regressor_bic = lasso_regressor.bic
neural_regressor_bic = neural_regressor.bic

plt.figure(figsize=(8, 5))
model_names = ['Linear Regressor', 'Ridge Regressor', 'Lasso Regressor']
bic_values = [linear_regressor_bic, ridge_regressor_bic, lasso_regressor_bic]
plt.bar(model_names, bic_values, color=['blue', 'orange'])
plt.ylabel('BIC')
plt.title('BIC of Different Regressors')
plt.savefig('bic_comparison.png')


plt.figure(figsize=(12, 8))
model_names = ['Linear Regressor', 'Linear Regressor with CV', 'Ridge Regressor', 'Ridge Regressor with CV', 'Lasso Regressor', 'Lasso Regressor with CV', 'Neural Regressor']
rmse_values = [linear_regressor.rmse, linear_regressor_with_cv.rmse, ridge_regressor.rmse, ridge_regressor_with_cv.rmse, lasso_regressor.rmse, lasso_regressor_with_cv.rmse, neural_regressor.rmse]
rmse_values.sort()
plt.bar(model_names, rmse_values, color=['blue', 'orange'])
plt.ylabel('RMSE')
plt.title('RMSE of Different Regressors')
plt.savefig('rmse_comparison.png')


"""
new_data = pd.DataFrame([
    ['Pylos', 'Weekday', 'Armenian Losh Kebab', '40.7261637', '-73.9840813'],
    ['Pylos', 'Weekday', 'Greek Salad', '40.7261637', '-73.9840813']
], columns=['restaurant_name', 'day_of_the_week', 'dish_name', 'latitude', 'longitude'])

pred1 = linear_regressor.predict(new_data)
pred2 = neural_regressor.predict(new_data)
pred3 = ridge_regressor.predict(new_data)
pred4 = lasso_regressor.predict(new_data)

print(pred1)
print(pred2)
print(pred3)
print(pred4)"""

#plotta i cv


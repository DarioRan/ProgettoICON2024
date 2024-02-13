import pandas as pd
from matplotlib import pyplot as plt

from src.supervised_learning.boosted_learner import BoostedRegressor
from src.supervised_learning.neural_learner import NeuralRegressor
from src.supervised_learning.ridge_learner import RidgeRegressor
from src.supervised_learning.linear_learner import LinearRegressor
from src.supervised_learning.lasso_learner import LassoRegressor

df = pd.read_csv('../../dataset/dishes_df.csv')


boosted_regressor = BoostedRegressor(df, False, False)
boosted_regressor.save_model()
boosted_regressor_with_cv = BoostedRegressor(df, True, False)
boosted_regressor_with_cv.save_model()

linear_regressor = LinearRegressor(df)
linear_regressor_with_cv = LinearRegressor(df, True)
linear_regressor.save_model()
linear_regressor_with_cv.save_model()


ridge_regressor = RidgeRegressor(df, alpha=0.1)
ridge_regressor.save_model()
ridge_regressor_with_cv = RidgeRegressor(df, True, alpha=0.1)
ridge_regressor_with_cv.save_model()

lasso_regressor = LassoRegressor(df,  alpha=0.1)
lasso_regressor.save_model()
lasso_regressor_with_cv = LassoRegressor(df, True, alpha=0.1)
lasso_regressor_with_cv.save_model()

neural_regressor = NeuralRegressor(df, False, False)
neural_regressor.save_model()



plt.figure(figsize=(12, 8))
model_names = ['Linear Regressor', 'Ridge Regressor', 'Lasso Regressor', 'Neural Regressor', 'Boosted Regressor']
bic_values = [linear_regressor.bic, ridge_regressor.bic, lasso_regressor.bic, neural_regressor.bic, boosted_regressor.bic]
bic_values.sort()
plt.bar(model_names, bic_values, color=['blue', 'orange'])
plt.ylabel('BIC')
plt.title('BIC of Different Regressors')
plt.savefig('output/bic_comparison.png')


plt.figure(figsize=(18, 14))
model_names = ['Linear Regressor', 'Ridge Regressor', 'Lasso Regressor', 'Neural Regressor', 'Boosted Regressor']
rmse_values = [linear_regressor.rmse, ridge_regressor.rmse,  lasso_regressor.rmse, neural_regressor.rmse, boosted_regressor.rmse]
rmse_values.sort()
plt.bar(model_names, rmse_values, color=['blue', 'orange'])
plt.ylabel('RMSE')
plt.title('RMSE of Different Regressors')
plt.savefig('output/rmse_comparison.png')


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


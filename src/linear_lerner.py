import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from src.app import df
import ast

data = pd.read_csv('../dataset/food_order_final.csv')
dishes_expanded_list = []

for index, row in df.iterrows():
    dishes = ast.literal_eval(row['dishes'])
    for dish in dishes:
        dishes_expanded_list.append({
            'restaurant_name': row['restaurant_name'],
            'day_of_the_week': row['day_of_the_week'],
            'dish_name': dish['dish_name'],  # Optional, if you want to use it as a feature
            'preparation_time': dish['preparation_time']
        })
dishes_df = pd.DataFrame(dishes_expanded_list)
dishes_df.to_csv("prova")

X = dishes_df[['restaurant_name', 'day_of_the_week', 'dish_name']]
y = dishes_df['preparation_time']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

categorical_features = ['restaurant_name', 'day_of_the_week', 'dish_name']
one_hot_encoder = OneHotEncoder(handle_unknown='ignore')
preprocessor = ColumnTransformer(transformers=[
    ('cat', one_hot_encoder, categorical_features)],
    remainder='passthrough')


alpha = 0.2  # Coefficiente di regolarizzazione
k = 20  # Numero di fold per la validazione incrociata


basic_model = Pipeline(steps=[('preprocessor', preprocessor),
                              ('regressor', LinearRegression())])
basic_model.fit(X_train, y_train)
basic_y_pred = basic_model.predict(X_test)
basic_rmse = np.sqrt(mean_squared_error(y_test, basic_y_pred))
print(f'basic RMSE: {basic_rmse}')

ridge_model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', Ridge(alpha=alpha))
])
ridge_scores = cross_val_score(ridge_model, X, y, cv=k, scoring='neg_mean_squared_error')
ridge_rmse_scores = np.sqrt(-ridge_scores)
print(f'Ridge RMSE: {ridge_rmse_scores.mean()} (± {ridge_rmse_scores.std()})')


lasso_model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', Lasso(alpha=alpha))
])
lasso_scores = cross_val_score(lasso_model, X, y, cv=k, scoring='neg_mean_squared_error')
lasso_rmse_scores = np.sqrt(-lasso_scores)
print(f'Lasso RMSE: {lasso_rmse_scores.mean()} (± {lasso_rmse_scores.std()})')



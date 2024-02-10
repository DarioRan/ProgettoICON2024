import pandas as pd

from src.supervised_learning.SGDLearner import SGDLeaner

df = pd.read_csv('../../dataset/dishes_df.csv')

sgd_regressor = SGDLeaner(df,False)
sgd_regressor.save_model()

sgd_regressor_with_cv = SGDLeaner(df, True)
sgd_regressor_with_cv.save_model()
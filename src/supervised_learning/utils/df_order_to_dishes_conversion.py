import ast
import pandas as pd

def retrieve_dataframe(df):

    dishes_expanded_list = []
    for index, row in df.iterrows():
        dishes = ast.literal_eval(row['dishes'])
        for dish in dishes:
            dishes_expanded_list.append({
                'restaurant_name': row['restaurant_name'],
                'restaurant_location': row['restaurant_location'],
                'day_of_the_week': row['day_of_the_week'],
                'dish_name': dish['dish_name'],
                'preparation_time': dish['preparation_time']
            })
    dishes_df = pd.DataFrame(dishes_expanded_list)
    return dishes_df
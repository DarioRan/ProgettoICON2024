from pyswip import Prolog
import pandas as pd
from pathlib import Path


def format_dishes(object_string):
    object_string = object_string.replace("['{}(,(:(dish_name, ", "").replace("), :(preparation_time, ",
                                                                              ", ").replace(
        ")))', '{}(,(:(dish_name, ", ", ").replace(")))']", "")

    object_list = object_string.split(", ")
    dict_list = []

    for i in range(0, len(object_list), 2):
        dict_list.append({'dish_name': object_list[i].strip(), 'preparation_time': int(object_list[i + 1])})

    return dict_list


class KB:
    def __init__(self):
        kb_file_path = str(Path(__file__).parent / 'knowledge_base.pl').replace("\\", "/")
        self.prolog = Prolog()
        self.prolog.consult(kb_file_path)

    def get_restaurants_by_cuisine(self, cuisine_type):
        list_query = list(self.prolog.query(
            f"findall(RestaurantName, restaurants_by_cuisine('{cuisine_type}', RestaurantName), Restaurants)"))
        restaurants = list_query[0]['Restaurants'] if list_query else []
        return restaurants

    def get_dishes_by_cuisine(self, cuisine_type):
        dishes = self.prolog.query(f"findall(Dish, dishes_by_cuisine('{cuisine_type}', Dish), Dishes)")
        dishes_list = [d['Dishes'] for d in dishes]
        return dishes_list[0]

    def get_all_cuisine_types(self):
        cuisine_types = list(self.prolog.query("setof(CuisineType, Dish^dish(CuisineType, Dish), CuisineTypes)"))
        cuisine_types_list = cuisine_types[0]['CuisineTypes'] if cuisine_types else []
        return cuisine_types_list

    def get_restaurant_location_by_cuisine(self, cuisine_type):
        list_query = self.prolog.query(
            f"restaurant_loc_by_cuisine('{cuisine_type}', RestaurantName, RestaurantLocation)")
        restaurants_data = []

        for result in list_query:
            restaurant_name = result['RestaurantName']
            restaurant_location = result['RestaurantLocation']
            restaurant_location = restaurant_location.replace(',(', '(')
            restaurants_data.append({'restaurant_name': restaurant_name, 'restaurant_location': restaurant_location})

        df = pd.DataFrame(restaurants_data)
        return df

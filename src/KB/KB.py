from pyswip import Prolog
import pandas as pd
from pathlib import Path


def format_dishes(object_string):
    """
    Format dishes string in a list of dictionaries

    :param object_string: the string to format

    :return: a list of dictionaries

    """
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
        kb_file_path_orders = str(Path(__file__).parent / 'knowledge_base_orders.pl').replace("\\", "/")
        self.prolog_orders = Prolog()
        self.prolog_orders.consult(kb_file_path_orders)
        kb_file_path_graph = str(Path(__file__).parent / 'knowledge_base_graph.pl').replace("\\", "/")
        self.prolog_graph = Prolog()
        self.prolog_graph.consult(kb_file_path_graph)

    def get_restaurants_by_cuisine(self, cuisine_type):
        """
        Get the restaurants by cuisine type

        :param cuisine_type: the cuisine type

        :return: a list of restaurants
        """
        list_query = list(self.prolog_orders.query(
            f"findall(RestaurantName, restaurants_by_cuisine('{cuisine_type}', RestaurantName), Restaurants)"))
        restaurants = list_query[0]['Restaurants'] if list_query else []
        return restaurants

    def get_dishes_by_cuisine(self, cuisine_type):
        """
        Get the dishes by cuisine type

        :param cuisine_type: the cuisine type

        :return: a list of dishes

        """
        dishes = self.prolog_orders.query(f"findall(Dish, dishes_by_cuisine('{cuisine_type}', Dish), Dishes)")
        dishes_list = [d['Dishes'] for d in dishes]
        return dishes_list[0]

    def get_all_cuisine_types(self):
        """
        Get all the cuisine types

        :return: a list of cuisine types

        """
        cuisine_types = list(self.prolog_orders.query("setof(CuisineType, Dish^dish(CuisineType, Dish), CuisineTypes)"))
        cuisine_types_list = cuisine_types[0]['CuisineTypes'] if cuisine_types else []
        return cuisine_types_list

    def get_restaurant_location_by_cuisine(self, cuisine_type):

        list_query = self.prolog_orders.query(
            f"restaurant_loc_by_cuisine('{cuisine_type}', RestaurantName, RestaurantLocation)")
        restaurants_data = []

        for result in list_query:
            restaurant_name = result['RestaurantName']
            restaurant_location = result['RestaurantLocation']
            restaurant_location = restaurant_location.replace(',(', '(')
            restaurants_data.append({'restaurant_name': restaurant_name, 'restaurant_location': restaurant_location})

        df = pd.DataFrame(restaurants_data)
        return df

    def get_dishes_info(self):
        """
        Get the dishes info

        :return: a DataFrame with the dishes info

        """
        dishes = self.prolog_orders.query("get_dishes_info(RestaurantName,RestaurantLocation,DayOfWeek,Dishes)")
        restaurants_data = []

        for result in dishes:
            restaurant_name = result['RestaurantName']
            restaurant_location = result['RestaurantLocation']
            restaurant_location = restaurant_location.replace(',(', '(')
            day_of_the_week = result['DayOfWeek']

            dish_name = result['Dishes']
            dishes_for_order = format_dishes(str(dish_name))
            for dish in dishes_for_order:
                restaurants_data.append({'restaurant_name': restaurant_name, 'restaurant_location': restaurant_location,
                                         'day_of_the_week': day_of_the_week, 'dish_name': dish['dish_name'],
                                         'preparation_time': dish['preparation_time']})

        df = pd.DataFrame(restaurants_data)
        return df

    def get_all_nodes(self):
        """
        Get all the nodes

        :return: a list of nodes

        """
        nodes = []
        result = list(self.prolog_graph.query("get_all_nodes(Nodes)"))
        if result:
            for node_dict in result:  # Stampiamo solo i primi 10 dizionari
                nodes.append(node_dict['Nodes'])
            return nodes
        else:
            print("Nessun nodo trovato.")
            return None

    def get_street_name(self, source, target):
        """
        Get the street name

        :param source: the source node

        :param target: the target node

        :return: the street name
        """
        query = "get_street_name('{}', '{}', StreetName)".format(source, target)
        result = list(self.prolog_graph.query(query))
        if result:
            return result[0]['StreetName']
        else:
            print("Nessun arco trovato tra i nodi {} e {}.".format(source, target))
            return None

    def get_edge_length(self, source, target):
        """
        Get the edge length

        :param source: the source node

        :param target: the target node

        :return: the edge length

        """
        query = "get_edge_length('{}', '{}', Length)".format(source, target)
        result = list(self.prolog_graph.query(query))
        if result:
            return result[0]['Length']
        else:
            print("Nessun arco trovato tra i nodi {} e {}.".format(source, target))
            return None

    def get_edge_flowSpeed(self, source, target):
        """
        Get the edge flow speed

        :param source: the source node

        :param target: the target node

        :return: the edge flow speed

        """
        query = "get_edge_flowSpeed('{}', '{}', FlowSpeed)".format(source, target)
        result = list(self.prolog_graph.query(query))
        if result:
            return result[0]['FlowSpeed']
        else:
            print("Nessun arco trovato tra i nodi {} e {}.".format(source, target))
            return None

    def get_neighbors(self, source):
        """
        Get the neighbors of a node

        :param source: the source node

        :return: a list of neighbors

        """
        query = f"get_neighbors('{source}',Neighbors)"
        result = list(self.prolog_graph.query(query))
        return result[0]['Neighbors']






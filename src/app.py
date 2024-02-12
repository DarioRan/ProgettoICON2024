import numpy as np
import torch
from flask import Flask, render_template, request, jsonify
import networkx as nx
import pandas as pd

from src.csp.csp_class import DriverAssignmentProblem
from src.find_path.utils import calculate_distance, find_path_BB, generate_map, calculate_delivery_time
from joblib import load
from KB.KB import KB
import datetime
from src.belief_network.belief_network import BeliefNetwork

app = Flask(__name__)

# Carica il grafo all'avvio dell'applicazione
G = nx.read_graphml('../dataset/newyork_final.graphml')

# Carica dataset incidenti
accidents_df = pd.read_csv('../dataset/road_closure.csv')

KB = KB()

cuisine_types = KB.get_all_cuisine_types()
cuisine_dishes_map = {}

# Popola il dizionario con le informazioni necessarie
for cuisine_type in cuisine_types:
    dishes_list = KB.get_dishes_by_cuisine(str(cuisine_type))
    cuisine_dishes_map[cuisine_type] = list(set(dishes_list))

# Coordinate dei punti di partenza e arrivo
start_coords = None
end_coords = None


@app.route('/')
def index():
    return render_template('index.html', cuisine_types=cuisine_types, cuisine_dishes_map=cuisine_dishes_map)


@app.route('/select_start', methods=['POST'])
def select_start():
    global start_coords
    start_coords = request.get_json()
    return jsonify({'status': 'success'})


@app.route('/select_end', methods=['POST'])
def select_end():
    global end_coords
    end_coords = request.get_json()
    return jsonify({'status': 'success'})


@app.route('/calculate_path', methods=['POST'])
def calculate_path():
    global start_coords, end_coords

    data = request.json
    start_coords = data.get('start_coords')
    end_coords = data.get('end_coords')

    if start_coords is None or end_coords is None:
        return jsonify({'error': 'Start and end coordinates must be selected'})

    lat_start, lon_start = start_coords['lat'], start_coords['lon']
    lat_end, lon_end = end_coords['lat'], end_coords['lon']

    # Trova il percorso più breve usando l'algoritmo Branch and Bound
    shortest_path, street_names = find_path_BB(G, lat_start, lon_start, lat_end, lon_end)

    # Calcola la distanza percorsa
    total_distance = calculate_distance(G, shortest_path)
    delivery_time = calculate_delivery_time(G, shortest_path)
    # Genera la mappa del percorso
    map_html = generate_map(G, shortest_path, (lat_start, lon_start), (lat_end, lon_end))

    return jsonify(
        {'path': shortest_path, 'street_names': street_names, 'total_distance': total_distance, 'map': map_html})


@app.route('/find_restaurant', methods=['POST'])
def trova_ristorante():
    print("\nInizio ricerca ristorante . . .\n")
    global total_expected_prep_time
    data = request.get_json()
    cuisine_type = data.get('cuisine_type')
    dishes = data.get('dishes')
    regression_mode = data.get('model')
    today = datetime.datetime.now()
    number_day_of_week = today.isoweekday()
    weekday = ''
    if number_day_of_week in range(1, 6):
        weekday = 'Weekday'
    else:
        weekday = 'Weekend'

    waiting_time = data.get('waiting_time')

    # fai con kb
    drivers_data = pd.read_csv('../dataset/drivers.csv')

    # dataframe con nome ristorante e location
    restaurant_locations = KB.get_restaurant_location_by_cuisine(str(cuisine_type)).drop_duplicates()

    linear_regressor = load('supervised_learning/output/models/linear_regressor.joblib')

    bn = BeliefNetwork(accidents_df)
    bn.train_model()

    # lista temporanea, verrà sostituita da csp
    temp_list = []
    for index, restaurant in restaurant_locations.iterrows():

        restaurant_location_str = restaurant['restaurant_location']
        restaurant_location_tuple = tuple(map(float, restaurant_location_str.strip('()').split(',')))
        lat = str(restaurant_location_tuple[0])
        lon = str(restaurant_location_tuple[1])

        new_data = pd.DataFrame([(restaurant['restaurant_name'], weekday, dish, lat, lon) for dish in dishes],
                                columns=['restaurant_name', 'day_of_the_week', 'dish_name', 'latitude', 'longitude'])

        expected_preparation_time_list = linear_regressor.predict(new_data)

        total_expected_prep_time = 0
        for value in expected_preparation_time_list:
            total_expected_prep_time += value

        temp_list.append((restaurant, total_expected_prep_time))

    # ordiniamo per tempo di preparazione e ci prendiamo il primo ristorante, sempre temporanea come cosa
    temp_list.sort(key=lambda x: x[1])

    # lista ristornati ordinati in base al tot delivery time
    temp_list2 = []
    for restaurant in temp_list[:10]:
        restaurant_location_str = restaurant[0]['restaurant_location']
        restaurant_location_tuple = tuple(map(float, restaurant_location_str.strip('()').split(',')))
        restaurant_loc_json = {'lat': float(restaurant_location_tuple[0]), 'lon': float(restaurant_location_tuple[1])}
        shortest_path, street_names = find_path_BB(G, restaurant_loc_json['lat'], restaurant_loc_json['lon'],
                                                   data.get('start_coords')['lat'], data.get('start_coords')['lon'])

        delivery_time = calculate_delivery_time(G, shortest_path)

        delivery_time_sec = delivery_time[0] * 60 + delivery_time[1]
        preparation_time_sec = restaurant[1] * 60

        tot_delivery_time_seconds = delivery_time_sec + preparation_time_sec

        time = datetime.datetime.now()
        time = time.replace(second=0, microsecond=0)
        time = time.replace(minute=30) if time.minute >= 30 else time.replace(minute=0)
        time = time.strftime('%H:%M')
        time = str(time)

        road_closure_prob = bn.predict_road_closure_probability(G, time, shortest_path)
        print(road_closure_prob)

        temp_list2.append((restaurant[0]['restaurant_name'], restaurant[0]['restaurant_location'],
                           tot_delivery_time_seconds, delivery_time_sec, preparation_time_sec, road_closure_prob))

    temp_list2.sort(key=lambda x: x[2])

    restaurant_name = temp_list2[0][0]
    restaurant_location_str = temp_list2[0][1]
    restaurant_location_tuple = tuple(map(float, restaurant_location_str.strip('()').split(',')))

    print("\nInizio ricerca driver\n")
    assignment_problem = DriverAssignmentProblem(drivers_data, restaurant_location_tuple)
    assignment_problem.create_problem()
    assignment_problem.solve_problem()
    assigned_driver_details = assignment_problem.get_assigned_driver_details()

    driver_profile = {}
    if assigned_driver_details['status'] == 'Optimal':
        driver_profile["id"] = assigned_driver_details["driver_id"]
        driver_profile["distance"] = round(float(assigned_driver_details["distance"]) * 111)
    else:
        driver_profile = None

    return jsonify_restaurant(restaurant_name, restaurant_location_str, temp_list2[0][3], temp_list2[0][4],
                              waiting_time, driver_profile, temp_list2[0][5])


def jsonify_restaurant(restaurant_name, restaurant_location, delivery_time, preparation_time, waiting_time,
                       driver_profile, probability):
    nome_ristorante = restaurant_name
    lat_lon_string = restaurant_location.strip('()').split(', ')
    lat, lon = map(float, lat_lon_string)
    posizione_ristorante = {'lat': lat, 'lon': lon}
    tempo_preparazione = str(round(preparation_time / 60))
    tempo_consegna = str(round(delivery_time / 60))
    if round(preparation_time / 60) + round(delivery_time / 60) <= int(waiting_time):
        if driver_profile:
            return jsonify({'nome_ristorante': nome_ristorante,
                            'posizione_ristorante': posizione_ristorante,
                            'tempo_preparazione': tempo_preparazione,
                            'tempo_consegna': tempo_consegna,
                            'driver_id': driver_profile['id'],
                            'driver_distance': driver_profile['distance'],
                            'road_closure_probability': probability
                            })
        else:
            message = f'Non ci sono driver in grado di effettuare la consegna in {waiting_time} minuti'
            return jsonify({'message': message, 'nome_ristorante': nome_ristorante,
                            'tempo_preparazione': tempo_preparazione, 'tempo_consegna': tempo_consegna})
    else:
        message = f'Non ci sono ristoranti in grado di effettuare la consegna in {waiting_time} minuti'
        return jsonify({'message': message, 'nome_ristorante': nome_ristorante,
                        'tempo_preparazione': tempo_preparazione, 'tempo_consegna': tempo_consegna})


if __name__ == '__main__':
    app.run(debug=True)

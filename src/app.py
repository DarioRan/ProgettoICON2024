import numpy as np
import torch
from flask import Flask, render_template, request, jsonify
import networkx as nx
import pandas as pd
from src.find_path.utils import calculate_distance, find_path_BB, generate_map, calculate_delivery_time
from joblib import load
from KB.KB import KB
import datetime
from src.belief_network.utils.utils import predict_road_closure_probability


app = Flask(__name__)

# Carica il grafo all'avvio dell'applicazione
G = nx.read_graphml('../dataset/newyork_final.graphml')

# Carica dataset incidenti
accidents_df = pd.read_csv('../dataset/accidents_ny.csv')

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

    # dataframe con nome ristorante e location
    restaurant_locations = KB.get_restaurant_location_by_cuisine(str(cuisine_type)).drop_duplicates()

    preprocessor = load('supervised_learning/output/models/preprocessor.joblib')

    kmn_clusterer = load('unsupervised_learning/output/models/kmn_clusterer.joblib')

    #linear_regressor = load('supervised_learning/output/models/linear_regressor.joblib')
    linear_regressor_with_cv = load('supervised_learning/output/models/linear_regressor_cv.joblib')
    linear_regressor_with_ing_feature = load('supervised_learning/output/models/linear_regressor_engineered.joblib')

    linear_regressor_with_sgd = load('supervised_learning/output/models/SGDlinear_regressor.joblib')
    linear_regressor_with_sgd_cv = load('supervised_learning/output/models/SGDlinear_regressor_cv.joblib')

    ridge_regressor = load('supervised_learning/output/models/ridge_regressor.joblib')
    ridge_regressor_with_cv = load('supervised_learning/output/models/ridge_regressor_cv.joblib')

    lasso_regressor = load('supervised_learning/output/models/lasso_regressor.joblib')
    lasso_regressor_with_cv = load('supervised_learning/output/models/lasso_regressor_cv.joblib')

    neural_regressor = load('supervised_learning/output/models/neural_regressor.joblib')

    boosted_regressor = load('supervised_learning/output/models/boosted_regressor.joblib')
    boosted_regressor_with_cv = load('supervised_learning/output/models/boosted_regressor_cv.joblib')

    # lista temporanea, verrà sostituita da csp
    temp_list = []
    for index, restaurant in restaurant_locations.iterrows():

        restaurant_location_str = restaurant['restaurant_location']
        restaurant_location_tuple = tuple(map(float, restaurant_location_str.strip('()').split(',')))
        lat = str(restaurant_location_tuple[0])
        lon = str(restaurant_location_tuple[1])


        new_data = pd.DataFrame([(restaurant['restaurant_name'], weekday, dish, lat, lon) for dish in dishes],
                                columns=['restaurant_name', 'day_of_the_week', 'dish_name', 'latitude', 'longitude'])

        new_data_processed = preprocessor.transform(new_data)
        new_data_tensor = torch.tensor(new_data_processed.todense().astype(np.float32))

        expected_preparation_time_list = []
        #if regression_mode == 'linearRegressor':
            #expected_preparation_time_list = linear_regressor.predict(new_data)
        if regression_mode == 'linearRegressorCV':
            expected_preparation_time_list = linear_regressor_with_cv.predict(new_data)
        elif regression_mode == 'LinearRegressorSGD':
            expected_preparation_time_list = linear_regressor_with_sgd.predict(new_data)
        elif regression_mode == 'LinearRegressorSGDCV':
            expected_preparation_time_list = linear_regressor_with_sgd_cv.predict(new_data)
        elif regression_mode == 'ridge':
            expected_preparation_time_list = ridge_regressor.predict(new_data)
        elif regression_mode == 'ridgeCV':
            expected_preparation_time_list = ridge_regressor_with_cv.predict(new_data)
        elif regression_mode == 'lasso':
            expected_preparation_time_list = lasso_regressor.predict(new_data)
        elif regression_mode == 'lassoCV':
            expected_preparation_time_list = lasso_regressor_with_cv.predict(new_data)
        elif regression_mode == 'neuralNetwork':
            expected_preparation_time_list = neural_regressor(new_data_tensor).detach().numpy().flatten()
        elif regression_mode == 'boostedRegressor':
            expected_preparation_time_list = boosted_regressor.predict(new_data_processed)
        elif regression_mode == 'boostedRegressorCV':
            expected_preparation_time_list = boosted_regressor_with_cv.predict(new_data_processed)
        elif regression_mode == 'LinearRegressorEF':
            cluster_labels = kmn_clusterer.predict((preprocessor.transform(new_data)))
            new_data['cluster'] = cluster_labels
            expected_preparation_time_list = linear_regressor_with_ing_feature.predict(new_data)

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

        temp_list2.append((restaurant[0]['restaurant_name'], restaurant[0]['restaurant_location'],
                           tot_delivery_time_seconds, delivery_time_sec, preparation_time_sec))

        #calcola probabilità trovare un blocco stradale
        #prob_delayed=predict_road_closure_probability(accidents_df,G, '18:00', shortest_path)

    temp_list2.sort(key=lambda x: x[2])

    restaurant_name = temp_list2[0][0]
    restaurant_location_str = temp_list2[0][1]

    return jsonify_restaurant(restaurant_name, restaurant_location_str, temp_list2[0][3], temp_list2[0][4], waiting_time)


def jsonify_restaurant(restaurant_name, restaurant_location, delivery_time, preparation_time, waiting_time):

    nome_ristorante = restaurant_name
    lat_lon_string = restaurant_location.strip('()').split(', ')
    lat, lon = map(float, lat_lon_string)
    posizione_ristorante = {'lat': lat, 'lon': lon}
    tempo_preparazione = str(round(preparation_time/60))
    tempo_consegna = str(round(delivery_time/60))
    if round(preparation_time/60) + round(delivery_time/60) <= int(waiting_time):
        return jsonify({'nome_ristorante': nome_ristorante,
                        'posizione_ristorante': posizione_ristorante,
                        'tempo_preparazione': tempo_preparazione,
                        'tempo_consegna': tempo_consegna})
    else:
        message = f'Non ci sono ristoranti in grado di effettuare la consegna in {waiting_time} minuti'
        return jsonify({'message': message, 'nome_ristorante': nome_ristorante,
                        'tempo_preparazione': tempo_preparazione, 'tempo_consegna': tempo_consegna})


if __name__ == '__main__':
    app.run(debug=True)

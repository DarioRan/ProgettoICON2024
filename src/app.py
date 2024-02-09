from flask import Flask, render_template, request, jsonify
import networkx as nx
import pandas as pd
from src.find_path.utils import calculate_distance, find_path_BB, generate_map, calculate_delivery_time
from joblib import load

app = Flask(__name__)

# Carica il grafo all'avvio dell'applicazione
G = nx.read_graphml('../dataset/newyork_final.graphml')

df = pd.read_csv('../dataset/food_order_final.csv')

cuisine_types = df['cuisine_type'].unique()

cuisine_dishes_map = {}

# Popola il dizionario con le informazioni necessarie
for cuisine_type in cuisine_types:
    dishes_list = []
    for index, row in df[df['cuisine_type'] == cuisine_type].iterrows():
        dishes_list.extend([dish['dish_name'] for dish in eval(row['dishes'])])
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
    restourant_list = df[df['cuisine_type'] == cuisine_type]['restaurant_name']

    # dataframe con nome ristorante e location
    restaurant_locations = df[df['cuisine_type'] == cuisine_type][['restaurant_name', 'restaurant_location']]

    linear_regressor = load('supervised_learning/output/models/linear_regressor.joblib')
    linear_regressor_with_cv = load('supervised_learning/output/models/linear_regressor_cv.joblib')

    ridge_regressor = load('supervised_learning/output/models/ridge_regressor.joblib')
    ridge_regressor_with_cv = load('supervised_learning/output/models/ridge_regressor_cv.joblib')

    lasso_regressor = load('supervised_learning/output/models/lasso_regressor.joblib')
    lasso_regressor_with_cv = load('supervised_learning/output/models/lasso_regressor_cv.joblib')

    neural_regressor = load('supervised_learning/output/models/neural_regressor.joblib')

    boosted_regressor = load('supervised_learning/output/models/boosted_regressor.joblib')
    boosted_regressor_with_cv = load('supervised_learning/output/models/boosted_regressor_cv.joblib')

    # lista temporanea, verrà sostituita da csp
    temp_list = []
    for restaurant in restourant_list:
        dishes = data.get('dishes')
        lat = data.get('start_coords')['lat']
        lon = data.get('start_coords')['lon']
        regression_mode = data.get('model')
        # inserire box per weekday o retrive da solo
        new_data = pd.DataFrame([(restaurant, 'Weekday', dish, lat, lon) for dish in dishes],
                                columns=['restaurant_name', 'day_of_the_week', 'dish_name', 'latitude', 'longitude'])

        expected_preparation_time_list = []
        if regression_mode == 'linearRegressor':
            print("sono qui")
            expected_preparation_time_list = linear_regressor.predict(new_data)
            print(expected_preparation_time_list)
        elif regression_mode == 'linearRegressorCV':
            expected_preparation_time_list = linear_regressor_with_cv.predict(new_data)
        elif regression_mode == 'ridge':
            expected_preparation_time_list = ridge_regressor.predict(new_data)
        elif regression_mode == 'ridgeCV':
            expected_preparation_time_list = ridge_regressor_with_cv.predict(new_data)
        elif regression_mode == 'lasso':
            expected_preparation_time_list = lasso_regressor.predict(new_data)
        elif regression_mode == 'lassoCV':
            expected_preparation_time_list = lasso_regressor_with_cv.predict(new_data)
        elif regression_mode == 'neuralNetwork':
            expected_preparation_time_list = neural_regressor.predict(new_data)
        elif regression_mode == 'boostedRegressor':
            expected_preparation_time_list = boosted_regressor.predict(new_data)
        elif regression_mode == 'boostedRegressorCV':
            expected_preparation_time_list = boosted_regressor_with_cv.predict(new_data)

        total_expected_prep_time = 0
        for value in expected_preparation_time_list:
            total_expected_prep_time += value

        temp_list.append((restaurant, total_expected_prep_time))

    # ordiniamo per tempo di preparazione e ci prendiamo il primo ristorante, sempre temporanea come cosa
    temp_list.sort(key=lambda x: x[1])

    restaurant_name = temp_list[0][0]
    preparation_time = temp_list[0][1]

    # csp su ristoranti e visualizzare
    for restaurant in temp_list[:10]:
        restaurant_lat_long = \
        restaurant_locations[restaurant_locations['restaurant_name'] == restaurant[0]]['restaurant_location'].iloc[
            0].strip('()').split(', ')
        restaurant_loc_json = {'lat': float(restaurant_lat_long[0]), 'lon': float(restaurant_lat_long[1])}
        shortest_path, street_names = find_path_BB(G, data.get('start_coords')['lat'], data.get('start_coords')['lon'],
                                                   restaurant_loc_json['lat'], restaurant_loc_json['lon'])

        delivery_time = calculate_delivery_time(G, shortest_path)

        tot_delivery_time = delivery_time[0] * 60 + delivery_time[1] + total_expected_prep_time
        # lista ristornati ordinati in base al tot delivery time
        temp_list2 = []
        temp_list2.append((restaurant[0], tot_delivery_time))

    temp_list2.sort(key=lambda x: x[1])

    best_restaurant_tot_time = df[df['restaurant_name'] == temp_list2[0][0]].iloc[0]

    return jsonify_restaurant(best_restaurant_tot_time, preparation_time)


def jsonify_restaurant(restaurant, preparation_time):
    if not restaurant.empty:
        nome_ristorante = restaurant['restaurant_name']
        lat_lon_string = restaurant['restaurant_location'].strip('()').split(', ')
        lat, lon = map(float, lat_lon_string)
        posizione_ristorante = {'lat': lat, 'lon': lon}
        rating_ristorante = str(restaurant['rating'])
        tempo_preparazione = str(preparation_time)
        tempo_consegna = str(restaurant['delivery_time'])
        return jsonify({'nome_ristorante': nome_ristorante,
                        'posizione_ristorante': posizione_ristorante,
                        'rating_ristorante': rating_ristorante,
                        'tempo_preparazione': tempo_preparazione,
                        'tempo_consegna': tempo_consegna})
    else:
        return jsonify({'nome_ristorante': None, 'posizione_ristorante': None})


if __name__ == '__main__':
    app.run(debug=True)

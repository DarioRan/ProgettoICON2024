from flask import Flask, render_template, request, jsonify
import networkx as nx
import pandas as pd
from find_path import calculate_distance, find_shortest_path, generate_map
from supervised_learning.linear_lerner import LinearRegressor

app = Flask(__name__)

# Carica il grafo all'avvio dell'applicazione
G = nx.read_graphml('newyork.graphml')

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

    # Trova il percorso più breve usando l'algoritmo A*
    shortest_path, street_names = find_shortest_path(G, lat_start, lon_start, lat_end, lon_end)

    # Calcola la distanza percorsa
    total_distance = calculate_distance(G, shortest_path)

    # Genera la mappa del percorso
    map_html = generate_map(G, shortest_path, (lat_start, lon_start), (lat_end, lon_end))

    return jsonify(
        {'path': shortest_path, 'street_names': street_names, 'total_distance': total_distance, 'map': map_html})



@app.route('/find_restaurant', methods=['POST'])
def trova_ristorante():
    data = request.get_json()
    cuisine_type = data.get('cuisine_type')
    restourant_list = df[df['cuisine_type'] == cuisine_type]['restaurant_name']
    linear_regressor = LinearRegressor(df)

   #lista temporanea, verrà sostituita da csp
    temp_list = []
    for restaurant in restourant_list:
        dishes = data.get('dishes')
        new_data = pd.DataFrame([(restaurant, 'Weekday', dish) for dish in dishes],
                                columns=['restaurant_name', 'day_of_the_week', 'dish_name'])
        expected_preparation_time_list = linear_regressor.predict(new_data)
        total_expected_time = expected_preparation_time_list.sum()
        temp_list.append((restaurant, total_expected_time))

    #ordiniamo per tempo di preparazione e ci prendiamo il primo ristorante, sempre temporanea come cosa
    temp_list.sort(key=lambda x: x[1])

    restaurant_name = temp_list[0][0]
    preparation_time = temp_list[0][1]

    restaurant = df[df['restaurant_name'] == restaurant_name].iloc[0]

    #csp su ristoranti e visualizzare

    return jsonify_restaurant(restaurant, preparation_time)


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

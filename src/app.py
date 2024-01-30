from flask import Flask, render_template, request, jsonify
import networkx as nx
from find_path import calculate_distance, find_shortest_path, generate_map
from folium import IFrame

app = Flask(__name__)

# Carica il grafo all'avvio dell'applicazione
G = nx.read_graphml('newyork.graphml')

# Coordinate dei punti di partenza e arrivo
start_coords = None
end_coords = None

@app.route('/')
def index():
    return render_template('index.html')

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

    return jsonify({'path': shortest_path, 'street_names': street_names, 'total_distance': total_distance, 'map': map_html})



if __name__ == '__main__':
    app.run(debug=True)

import networkx as nx
import numpy as np

import geopy.distance
import folium

from astar import astar_path

def find_closest_node(graph, lat, lon):
    # Calcola la distanza euclidea tra le coordinate di input e le coordinate di ogni nodo
    distances = {}
    for node, data in graph.nodes(data=True):
        if 'y' in data and 'x' in data:
            node_lat = float(data['y'])
            node_lon = float(data['x'])
            distances[node] = np.sqrt((node_lat - lat)**2 + (node_lon - lon)**2)
    # Trova il nodo con la distanza minima
    closest_node = min(distances, key=distances.get)
    return closest_node

def find_street_names(graph, path):
    street_names = []
    for i in range(len(path) - 1):
        edge_data = graph.get_edge_data(path[i], path[i+1])
        if edge_data and 'name' in edge_data[0]:
            street_name = edge_data[0]['name']
            if not street_names or (street_names and street_names[-1] != street_name):
                street_names.append(street_name)
    return street_names

def heuristic(node_1, node_2):
    # Calcola la distanza euclidea tra i due nodi
    coords_1 = G.nodes[node_1]['y'], G.nodes[node_1]['x']
    coords_2 = G.nodes[node_2]['y'], G.nodes[node_2]['x']
    return geopy.distance.distance(coords_1, coords_2).miles


# Leggi il file GraphML
G = nx.read_graphml('newyork.graphml')

# Definisci le coordinate di partenza e di arrivo
lat_start, lon_start = 40.765063, -73.976782 # sostituisci con le tue coordinate di partenza
lat_end, lon_end = 40.675822, -73.944467 # sostituisci con le tue coordinate di arrivo

# Trova i nodi più vicini alle coordinate di partenza e di arrivo
start_node = find_closest_node(G, lat_start, lon_start)
end_node = find_closest_node(G, lat_end, lon_end)

for u, v, data in G.edges(data=True):
    data['length'] = float(data['length'])

# Trova il percorso più breve usando l'algoritmo A*
shortest_path = astar_path(G, source=start_node, target=end_node, heuristic=heuristic, weight='length')

# Trova i nomi delle strade lungo il percorso più breve
street_names = find_street_names(G, shortest_path)

# Stampa i nomi delle strade
print('Le strade da percorrere sono:', street_names)

map_center = [float(G.nodes[shortest_path[0]]['y']), float(G.nodes[shortest_path[0]]['x'])]
mymap = folium.Map(location=map_center, zoom_start=14)

# Aggiungi il percorso alla mappa
coordinates = [(float(G.nodes[node]['y']), float(G.nodes[node]['x'])) for node in shortest_path]
folium.PolyLine(coordinates, color='blue', weight=5, opacity=1).add_to(mymap)

# Aggiungi i marcatori per i punti di partenza e arrivo
folium.Marker([lat_start, lon_start], popup='Start', icon=folium.Icon(color='green')).add_to(mymap)
folium.Marker([lat_end, lon_end], popup='End', icon=folium.Icon(color='red')).add_to(mymap)

# Visualizza la mappa
mymap.save('map.html')
import networkx as nx
import numpy as np
import geopy.distance
import folium
from astar import astar_path


def find_closest_node(graph, lat, lon):
    """Trova il nodo più vicino alle coordinate date nel grafo.

    Parametri
    ----------
    graph : Grafo.

    lat : float
        Latitudine delle coordinate.

    lon : float
        Longitudine delle coordinate.

    Restituisce
    -------
    node
        Il nodo più vicino alle coordinate date nel grafo.
    """
    distances = {}
    for node, data in graph.nodes(data=True):
        if 'y' in data and 'x' in data:
            node_lat = float(data['y'])
            node_lon = float(data['x'])
            distances[node] = np.sqrt((node_lat - lat) ** 2 + (node_lon - lon) ** 2)
    closest_node = min(distances, key=distances.get)
    return closest_node


def find_street_names(graph, path):
    """Trova i nomi delle strade lungo un percorso nel grafo.

    Parametri
    ----------
    graph : Grafo.

    path : Lista di nodi
        Percorso nel grafo.

    Restituisce
    -------
    list
        Lista dei nomi delle strade lungo il percorso.
    """
    street_names = []
    for i in range(len(path) - 1):
        edge_data = graph.get_edge_data(path[i], path[i + 1])
        if edge_data and 'name' in edge_data[0]:
            street_name = edge_data[0]['name']
            if not street_names or (street_names and street_names[-1] != street_name):
                street_names.append(street_name)
    return street_names


def heuristic(node_1, node_2, graph):
    """Calcola la distanza euclidea tra due nodi nel grafo.

    Parametri
    ----------
    node_1 : Nodo
        Primo nodo.

    node_2 : Nodo
        Secondo nodo.

    graph : Grafo contenente i nodi.

    Restituisce
    -------
    float
        Distanza euclidea tra i due nodi in miglia.
    """
    coords_1 = graph.nodes[node_1]['y'], graph.nodes[node_1]['x']
    coords_2 = graph.nodes[node_2]['y'], graph.nodes[node_2]['x']
    return geopy.distance.distance(coords_1, coords_2).miles


def find_shortest_path(graph, lat_start, lon_start, lat_end, lon_end):
    """Trova il percorso più breve tra due punti nel grafo utilizzando l'algoritmo A*.

    Parametri
    ----------
    graph : Grafo.

    lat_start : float
        Latitudine del punto di partenza.

    lon_start : float
        Longitudine del punto di partenza.

    lat_end : float
        Latitudine del punto di arrivo.

    lon_end : float
        Longitudine del punto di arrivo.

    Restituisce
    -------
    tuple
        Una tupla contenente il percorso più breve e i nomi delle strade lungo il percorso.
    """
    start_node = find_closest_node(graph, lat_start, lon_start)
    end_node = find_closest_node(graph, lat_end, lon_end)

    for u, v, data in graph.edges(data=True):
        data['length'] = float(data['length'])

    shortest_path = astar_path(graph, source=start_node, target=end_node, heuristic=lambda x, y: heuristic(x, y, graph),
                               weight='length')
    street_names = find_street_names(graph, shortest_path)

    return shortest_path, street_names


def calculate_distance(graph, path):
    """Calcola la distanza totale di un percorso nel grafo.

    Parametri
    ----------
    graph : Grafo.

    path : Lista di nodi
        Percorso nel grafo.

    Restituisce
    -------
    float
        Distanza totale del percorso in metri.
    """
    total_distance = 0
    for i in range(len(path) - 1):
        edge_data = graph.get_edge_data(path[i], path[i + 1])
        if edge_data and 'length' in edge_data[0]:
            total_distance += edge_data[0]['length']
    return total_distance


def generate_map(graph, shortest_path):
    """Genera una mappa con il percorso evidenziato.

    Parametri
    ----------
    graph : Grafo.

    shortest_path : Lista di nodi
        Percorso nel grafo da evidenziare sulla mappa.
    """
    map_center = [float(graph.nodes[shortest_path[0]]['y']), float(graph.nodes[shortest_path[0]]['x'])]
    mymap = folium.Map(location=map_center, zoom_start=14)

    # Aggiungi il percorso alla mappa
    coordinates = [(float(graph.nodes[node]['y']), float(graph.nodes[node]['x'])) for node in shortest_path]
    folium.PolyLine(coordinates, color='blue', weight=5, opacity=1).add_to(mymap)

    # Aggiungi i marcatori per i punti di partenza e arrivo
    folium.Marker([lat_start, lon_start], popup='Start', icon=folium.Icon(color='green')).add_to(mymap)
    folium.Marker([lat_end, lon_end], popup='End', icon=folium.Icon(color='red')).add_to(mymap)

    # Visualizza la mappa
    mymap.save('map.html')


if __name__ == "__main__":
    # Leggi il file GraphML
    G = nx.read_graphml('newyork.graphml')

    # Definisci le coordinate di partenza e di arrivo
    lat_start, lon_start = 40.661002, -73.947765  # sostituisci con le tue coordinate di partenza
    lat_end, lon_end = 40.66322150342033, -73.96088752574344  # sostituisci con le tue coordinate di arrivo

    # Trova il percorso più breve usando l'algoritmo A*
    shortest_path, street_names = find_shortest_path(G, lat_start, lon_start, lat_end, lon_end)

    # Calcola la distanza percorsa
    total_distance = calculate_distance(G, shortest_path)

    # Stampa i nomi delle strade e la distanza
    print('Le strade da percorrere sono:', street_names)
    print(f'La lunghezza totale del percorso è: {total_distance / 1000:.2f} km')

    # Genera la mappa del percorso
    generate_map(G, shortest_path)

import numpy as np
import geopy.distance
import folium
from src.find_path.algorithms.Astar import astar_path
from src.find_path.algorithms.BranchAndBound import branch_and_bound
from src.find_path.algorithms.Dijkstra import dijkstra


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


def _heuristic(node_1, node_2, graph):
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


def find_path_Dijkstra(graph, lat_start, lon_start, lat_end, lon_end):
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

    shortest_path = dijkstra(graph, start=start_node, end=end_node)

    street_names = find_street_names(graph, shortest_path)

    return shortest_path, street_names


def find_path_BB(graph, lat_start, lon_start, lat_end, lon_end):
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

    shortest_path = branch_and_bound(graph, start=start_node, end=end_node)

    street_names = find_street_names(graph, shortest_path)

    return shortest_path, street_names


def find_path_A_star(graph, lat_start, lon_start, lat_end, lon_end):
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

    shortest_path = astar_path(graph, source=start_node, target=end_node, heuristic=lambda x, y: _heuristic(x, y, graph),
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


def calculate_delivery_time(G, shortest_path):
    total_time = 0
    for i in range(len(shortest_path) - 1):
        edge_data = G.get_edge_data(shortest_path[i], shortest_path[i + 1])
        if edge_data and 'length' in edge_data[0] and 'flowSpeed' in edge_data[0]:
            total_time += edge_data[0]['length'] / (edge_data[0]['flowSpeed'] * 0.44704)

    total_seconds = total_time  # Converti il tempo totale in secondi
    total_minutes = int(total_seconds // 60)  # Calcola i minuti
    total_seconds = int(total_seconds % 60)  # Ottieni solo la parte intera dei secondi

    return total_minutes, total_seconds


def generate_map(graph, shortest_path, start_coords=None, end_coords=None):
    """Genera una mappa con il percorso evidenziato.

    Parametri
    ----------
    graph : Grafo.

    shortest_path : Lista di nodi
        Percorso nel grafo da evidenziare sulla mappa.

    start_coords : Tuple, opzionale
        Coordinate di partenza (latitudine, longitudine).

    end_coords : Tuple, opzionale
        Coordinate di arrivo (latitudine, longitudine).

    Restituisce
    -------
    str
        HTML della mappa generata.
    """
    map_center = [float(graph.nodes[shortest_path[0]]['y']), float(graph.nodes[shortest_path[0]]['x'])]
    mymap = folium.Map(location=map_center, zoom_start=14)

    # Aggiungi il percorso alla mappa
    coordinates = [(float(graph.nodes[node]['y']), float(graph.nodes[node]['x'])) for node in shortest_path]
    folium.PolyLine(coordinates, color='blue', weight=5, opacity=1).add_to(mymap)

    # Aggiungi i marcatori per i punti di partenza e arrivo se forniti
    if start_coords:
        folium.Marker([start_coords[0], start_coords[1]], popup='Start', icon=folium.Icon(color='green')).add_to(mymap)
    if end_coords:
        folium.Marker([end_coords[0], end_coords[1]], popup='End', icon=folium.Icon(color='red')).add_to(mymap)

    # Restituisci l'HTML della mappa
    return mymap._repr_html_()

def save_map_html(html_content, filename):
    """Salva l'HTML della mappa in un file HTML.

    Parametri
    ----------
    html_content : str
        HTML della mappa.

    filename : str
        Nome del file HTML da salvare.
    """
    with open(filename, 'w') as file:
        file.write(html_content)
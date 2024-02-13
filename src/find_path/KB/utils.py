import numpy as np
import geopy.distance
import folium
from find_path.KB.Astar_revisited import astar_revisited



def find_coords_by_id(nodes, node_id):
    for node in nodes:
        if node[0] == node_id:
            return (node[1], node[2])
    return None

def find_closest_node(KB, lat, lon):
    """Trova il nodo più vicino alle coordinate date nel grafo.

    Parametri
    ----------
    KB : Knoledge Base.

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
    for node in KB.get_all_nodes()[0]:
        node_lat = float(node[1])
        node_lon = float(node[2])
        distances[node[0]] = np.sqrt((node_lat - lat) ** 2 + (node_lon - lon) ** 2)
    closest_node = min(distances, key=distances.get)
    return closest_node


def find_street_names(KB, path):
    """Trova i nomi delle strade lungo un percorso nel grafo.

    Parametri
    ----------
    KB : Knoledge Base.

    path : Lista di nodi
        Percorso nel grafo.

    Restituisce
    -------
    list
        Lista dei nomi delle strade lungo il percorso.
    """
    street_names = []
    for i in range(len(path) - 1):
        street_name = KB.get_street_name(path[i], path[i + 1])
        if not street_names or (street_names and street_names[-1] != street_name):
            street_names.append(street_name)

    return street_names

def find_path_Astar_revisited(KB, lat_start, lon_start, lat_end, lon_end):
    """Trova il percorso più breve tra due punti nel grafo utilizzando l'algoritmo A*.

    Parametri
    ----------
    KB : Knoledge Base.

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
    start_node = find_closest_node(KB, lat_start, lon_start)
    end_node = find_closest_node(KB, lat_end, lon_end)

    shortest_path = astar_revisited(KB, start=start_node, end=end_node)

    street_names = find_street_names(KB, shortest_path)

    return shortest_path, street_names


def calculate_distance(KB, path):
    """Calcola la distanza totale di un percorso nel grafo.

    Parametri
    ----------
    KB : Knoledge Base.

    path : Lista di nodi
        Percorso nel grafo.

    Restituisce
    -------
    float
        Distanza totale del percorso in metri.
    """
    total_distance = 0
    for i in range(len(path) - 1):
        total_distance += float(KB.get_edge_length(path[i], path[i + 1]))

    return total_distance


def calculate_delivery_time(KB, path):
    """Calcola la distanza totale di un percorso nel grafo.

        Parametri
        ----------
        KB : Knoledge Base.

        path : Lista di nodi
            Percorso nel grafo.

        Restituisce
        -------
        tuple : minuti e secondi.
        """
    total_time = 0
    for i in range(len(path) - 1):
        edge_flowSpeed = float(KB.get_edge_flowSpeed(path[i], path[i + 1]))
        edge_length = float(KB.get_edge_length(path[i], path[i + 1]))
        total_time += edge_length / (edge_flowSpeed * 0.44704)

    total_seconds = total_time  # Converti il tempo totale in secondi
    total_minutes = int(total_seconds // 60)  # Calcola i minuti
    total_seconds = int(total_seconds % 60)  # Ottieni solo la parte intera dei secondi

    return total_minutes, total_seconds


def generate_map(KB, shortest_path, start_coords=None, end_coords=None):
    """Genera una mappa con il percorso evidenziato.

    Parametri
    ----------
    KB : Knoledge Base.

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
    nodes = KB.get_all_nodes()[0]
    map_center = [find_coords_by_id(nodes, shortest_path[0])[0], find_coords_by_id(nodes, shortest_path[0])[1]]
    mymap = folium.Map(location=map_center, zoom_start=14)

    # Aggiungi il percorso alla mappa
    coordinates = [find_coords_by_id(nodes, node) for node in shortest_path]
    coordinates = [(float(lat), float(lon)) for lat, lon in coordinates]
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





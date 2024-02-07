import networkx as nx
import time
from src.find_path import utils

if __name__ == "__main__":
    # Leggi il file GraphML
    G = nx.read_graphml('../../dataset/newyork_final.graphml')

    # Definisci le coordinate di partenza e di arrivo
    lat_start, lon_start = 40.766725362591636, -73.92855802315485  # sostituisci con le tue coordinate di partenza
    lat_end, lon_end = 40.76282963149498, -73.93403177789565  # sostituisci con le tue coordinate di arrivo

    # Trova il percorso più breve usando l'algoritmo A*
    start_time = time.time()
    path, street_names = utils.find_path_A_star(G, lat_start, lon_start, lat_end, lon_end)
    end_time = time.time()

    # Calcola la distanza percorsa
    total_distance = utils.calculate_distance(G, path)
    total_minutes, total_seconds = utils.calculate_delivery_time(G, path)

    # Stampa i nomi delle strade e la distanza
    print("A*")
    print('Le strade da percorrere sono:', street_names)
    print(f'La lunghezza totale del percorso è: {total_distance / 1000:.2f} km')
    print(f'Il tempo totale del percorso è: {total_minutes} minuti e {total_seconds} secondi')
    print(f'Tempo di esecuzione: {end_time - start_time:.5f} secondi')
    html_map_content = utils.generate_map(G, path, path[0], path[-1])
    utils.save_map_html(html_map_content, 'results/A.html')

    # Trova il percorso più breve usando l'algoritmo BB
    start_time = time.time()
    path, street_names = utils.find_path_BB(G, lat_start, lon_start, lat_end, lon_end)
    end_time = time.time()

    # Calcola la distanza percorsa
    total_distance = utils.calculate_distance(G, path)
    total_minutes, total_seconds = utils.calculate_delivery_time(G, path)

    print("\nBranch and bound")
    print('Le strade da percorrere sono:', street_names)
    print(f'La lunghezza totale del percorso è: {total_distance / 1000:.2f} km')
    print(f'Il tempo totale del percorso è: {total_minutes} minuti e {total_seconds} secondi')
    print(f'Tempo di esecuzione: {end_time - start_time:.5f} secondi')
    html_map_content = utils.generate_map(G, path, path[0], path[-1])
    utils.save_map_html(html_map_content, 'results/BB.html')

    # Trova il percorso più breve usando l'algoritmo di Dijkstra
    start_time = time.time()
    path, street_names = utils.find_path_Dijkstra(G, lat_start, lon_start, lat_end, lon_end)
    end_time = time.time()

    # Calcola la distanza percorsa
    total_distance = utils.calculate_distance(G, path)
    total_minutes, total_seconds = utils.calculate_delivery_time(G, path)

    print("\nDijkstra")
    print('Le strade da percorrere sono:', street_names)
    print(f'La lunghezza totale del percorso è: {total_distance / 1000:.2f} km')
    print(f'Il tempo totale del percorso è: {total_minutes} minuti e {total_seconds} secondi')
    print(f'Tempo di esecuzione: {end_time - start_time:.5f} secondi')
    html_map_content = utils.generate_map(G, path, path[0], path[-1])
    utils.save_map_html(html_map_content, 'results/Dijkstra.html')

import networkx as nx
import time
import random
from geopy.distance import geodesic
from src.find_path import utils as utils1
from src.find_path.KB import utils as utils2
from KB.KB import KB

if __name__ == "__main__":
    # Leggi il file GraphML
    G = nx.read_graphml('../../../dataset/newyork_final.graphml')
    KB = KB()

    # Definisci le coordinate di partenza
    lat_start, lon_start = 40.80904511912113, -73.94459026721321  # sostituisci con le tue coordinate di partenza



    # Genera le coordinate di arrivo in base alla distanza
    random_direction = random.uniform(0, 359)
    destination = geodesic(kilometers=5).destination(point=(lat_start, lon_start), bearing=random_direction)
    lat_end, lon_end = destination.latitude, destination.longitude

    # Inizializza le variabili per il tempo di esecuzione
    start_time, end_time = 0, 0

    print(f"\nA* rivisitato senza accesso a KB\n")
    # Trova il percorso più breve usando l'algoritmo A*
    start_time = time.time()
    path_a_star, street_names_a_star = utils1.find_path_Astar_revisited(G, lat_start, lon_start, lat_end, lon_end)
    end_time = time.time()
    execution_time_a_star = end_time - start_time
    total_distance_a_star = utils1.calculate_distance(G, path_a_star)
    total_minutes_a_star, total_seconds_a_star = utils1.calculate_delivery_time(G, path_a_star)

    # Trova il percorso più breve usando l'algoritmo A* rivisitato
    print(f"\nA* rivisitato con accesso a KB\n")
    start_time = time.time()
    path_astar_riv, street_names_astar_riv = utils2.find_path_Astar_revisited(KB, lat_start, lon_start, lat_end, lon_end)
    end_time = time.time()
    execution_time_bb = end_time - start_time
    total_distance_bb = utils2.calculate_distance(KB, path_astar_riv)
    total_minutes_bb, total_seconds_bb = utils2.calculate_delivery_time(KB, path_astar_riv)



    print("A* rivisitato senza accesso a KB")
    print(f'La lunghezza totale del percorso è: {total_distance_a_star / 1000:.2f} km')
    print(
        f'Il tempo totale del percorso è: {total_minutes_a_star} minuti e {total_seconds_a_star} secondi')
    print(f'Tempo di esecuzione: {execution_time_a_star:.5f} secondi\n')

    print("A* rivisitato con accesso a KB")
    print(f'La lunghezza totale del percorso è: {total_distance_bb / 1000:.2f} km')
    print(f'Il tempo totale del percorso è: {total_minutes_bb} minuti e {total_seconds_bb} secondi')
    print(f'Tempo di esecuzione: {execution_time_bb:.5f} secondi\n')



import networkx as nx
import time
import random
from geopy.distance import geodesic
from src.find_path import utils

if __name__ == "__main__":
    # Leggi il file GraphML
    G = nx.read_graphml('../../../dataset/newyork_final.graphml')

    # Definisci le coordinate di partenza
    lat_start, lon_start = 40.80904511912113, -73.94459026721321  # sostituisci con le tue coordinate di partenza

    # Definisci le distanze da testare
    distanze = [0.2, 0.5, 1, 2, 5, 10, 20, 50]  # in chilometri

    for distanza in distanze:
        # Genera le coordinate di arrivo in base alla distanza
        random_direction = random.uniform(0, 359)
        destination = geodesic(kilometers=distanza).destination(point=(lat_start, lon_start), bearing=random_direction)
        lat_end, lon_end = destination.latitude, destination.longitude

        # Inizializza le variabili per il tempo di esecuzione
        start_time, end_time = 0, 0

        print(f"\nA* per {distanza}\n")
        # Trova il percorso più breve usando l'algoritmo A*
        start_time = time.time()
        path_a_star, street_names_a_star = utils.find_path_A_star(G, lat_start, lon_start, lat_end, lon_end)
        end_time = time.time()
        execution_time_a_star = end_time - start_time
        total_distance_a_star = utils.calculate_distance(G, path_a_star)
        total_minutes_a_star, total_seconds_a_star = utils.calculate_delivery_time(G, path_a_star)

        # Trova il percorso più breve usando l'algoritmo A* rivisitato
        print(f"\nA* rivisitato per {distanza}\n")
        start_time = time.time()
        path_astar_riv, street_names_astar_riv = utils.find_path_Astar_rivisited(G, lat_start, lon_start, lat_end, lon_end)
        end_time = time.time()
        execution_time_bb = end_time - start_time
        total_distance_bb = utils.calculate_distance(G, path_astar_riv)
        total_minutes_bb, total_seconds_bb = utils.calculate_delivery_time(G, path_astar_riv)

        if distanza <= 2: # dopo i 2k Dijkstra fallisce spesso (complessità computazionale troppo alta)
            print(f"\nDijkstra per {distanza}\n")
            # Trova il percorso più breve usando l'algoritmo di Dijkstra
            start_time = time.time()
            path_dijkstra, street_names_dijkstra = utils.find_path_Dijkstra(G, lat_start, lon_start, lat_end, lon_end)
            end_time = time.time()
            execution_time_dijkstra = end_time - start_time
            total_distance_dijkstra = utils.calculate_distance(G, path_dijkstra)
            total_minutes_dijkstra, total_seconds_dijkstra = utils.calculate_delivery_time(G, path_dijkstra)


        # Salva i risultati su file
        with open(f'results/{int(distanza * 1000)}_m.txt', 'a') as file:
            file.write(f"Distanza: {distanza} km\n\n")

            file.write("A*\n")
            file.write(f'La lunghezza totale del percorso è: {total_distance_a_star / 1000:.2f} km\n')
            file.write(
                f'Il tempo totale del percorso è: {total_minutes_a_star} minuti e {total_seconds_a_star} secondi\n')
            file.write(f'Tempo di esecuzione: {execution_time_a_star:.5f} secondi\n\n')

            file.write("A* rivisitato\n")
            file.write(f'La lunghezza totale del percorso è: {total_distance_bb / 1000:.2f} km\n')
            file.write(f'Il tempo totale del percorso è: {total_minutes_bb} minuti e {total_seconds_bb} secondi\n')
            file.write(f'Tempo di esecuzione: {execution_time_bb:.5f} secondi\n\n')

            if distanza <= 2:
                file.write("Dijkstra\n")
                file.write(f'La lunghezza totale del percorso è: {total_distance_dijkstra / 1000:.2f} km\n')
                file.write(
                    f'Il tempo totale del percorso è: {total_minutes_dijkstra} minuti e {total_seconds_dijkstra} secondi\n')
                file.write(f'Tempo di esecuzione: {execution_time_dijkstra:.5f} secondi\n\n')

    print("Tutti i risultati sono stati salvati correttamente.")

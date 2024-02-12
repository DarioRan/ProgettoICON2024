import geopy.distance
import heapq

def euclidean_distance(graph, node_1, node_2):
    # Calcola la distanza euclidea tra due nodi nel piano
    coords_1 = graph.nodes[node_1]['y'], graph.nodes[node_1]['x']
    coords_2 = graph.nodes[node_2]['y'], graph.nodes[node_2]['x']
    return geopy.distance.distance(coords_1, coords_2).miles

def astar_revisited(graph, start, end):
    # Inizializzazione della coda con la tupla (distanza, nodo)
    queue = [(0, start)]
    # Dizionario per tenere traccia dei nodi visitati
    visited = set()
    # Dizionario per tenere traccia del predecessore di ciascun nodo nel percorso più breve
    predecessors = {node: (float('inf'), None) for node in graph.nodes}

    while queue:
        # Estraiamo il nodo con la minima distanza
        current_distance, current_node = heapq.heappop(queue)

        # Se raggiungiamo il nodo di destinazione, costruiamo il percorso e terminiamo l'algoritmo
        if current_node == end:
            shortest_path = []
            node = end
            while node is not None:
                shortest_path.insert(0, node)
                node = predecessors[node][1]
            return shortest_path

        # Aggiungiamo il nodo corrente all'insieme dei visitati
        visited.add(current_node)

        # Iteriamo sui vicini del nodo corrente
        for neighbor, edge_data in graph[current_node].items():
            if neighbor not in visited and 'flowSpeed' in edge_data[0] and 'length' in edge_data[0]:
                # Calcoliamo il tempo di percorrenza dell'arco
                # Consideriamo la lunghezza dell'arco e la velocità del flusso
                weight = edge_data[0]['length'] / (edge_data[0]['flowSpeed'] * 0.44704)
                # Calcoliamo la distanza euclidea tra il vicino e il nodo di destinazione
                heuristic = euclidean_distance(graph, neighbor, end)

                # Calcoliamo la stima della distanza totale
                total_estimate = current_distance + weight + heuristic

                # Aggiorniamo il predecessore del vicino se abbiamo trovato un percorso più breve
                if total_estimate < predecessors[neighbor][0]:
                    predecessors[neighbor] = (total_estimate, current_node)
                    # Aggiungiamo il vicino alla coda con la stima della distanza come priorità
                    heapq.heappush(queue, (total_estimate, neighbor))

    # Se non viene trovato un percorso, restituiamo una lista vuota
    return []

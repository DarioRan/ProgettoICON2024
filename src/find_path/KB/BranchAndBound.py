import geopy.distance


def find_coords_by_id(nodes, node_id):
    for node in nodes:
        if node[0] == node_id:
            return (node[1], node[2])
    return None


def euclidean_distance(nodes, node_1, node_2):
    # Calcola la distanza euclidea tra due nodi nel piano
    coords_1 = find_coords_by_id(nodes, node_1)
    coords_2 = find_coords_by_id(nodes, node_2)
    return geopy.distance.distance(coords_1, coords_2).miles


def branch_and_bound(KB, start, end):
    # Inizializzazione della coda con la tupla (distanza, nodo)
    queue = [(0, start)]
    # Dizionario per tenere traccia dei nodi visitati
    visited = set()

    nodes = KB.get_all_nodes()[0]

    # Dizionario per tenere traccia del predecessore di ciascun nodo nel percorso più breve
    predecessors = {node[0]: (float('inf'), None) for node in nodes}

    while queue:
        # Estraiamo il nodo con la minima distanza
        current_distance, current_node = queue.pop(0)

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
        for neighbor in KB.get_neighbors(current_node):
            edge_flowSpeed = float(KB.get_edge_flowSpeed(current_node, neighbor))
            edge_length = float(KB.get_edge_length(current_node, neighbor))
            if neighbor not in visited:
                # Calcoliamo il tempo di percorrenza dell'arco
                # Consideriamo la lunghezza dell'arco e la velocità del flusso
                weight = edge_length / (edge_flowSpeed * 0.44704)
                # Calcoliamo la distanza euclidea tra il vicino e il nodo di destinazione
                heuristic = euclidean_distance(nodes, neighbor, end)

                # Calcoliamo la stima della distanza totale
                total_estimate = current_distance + weight + heuristic

                # Aggiorniamo il predecessore del vicino se abbiamo trovato un percorso più breve
                if total_estimate < predecessors[neighbor][0]:
                    predecessors[neighbor] = (total_estimate, current_node)
                    # Aggiungiamo il vicino alla coda con la stima della distanza come priorità
                    queue.append((total_estimate, neighbor))

        # Ordiniamo la coda in base alla stima della distanza totale
        queue.sort()

    # Se non viene trovato un percorso, restituiamo una lista vuota
    return []

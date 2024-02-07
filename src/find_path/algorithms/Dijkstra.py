import heapq


def dijkstra(graph, start, end):
    # Inizializzazione della coda con la tupla (tempo, nodo)
    global weight
    queue = [(0, start)]
    # Dizionario per tenere traccia dei tempi di percorrenza più brevi
    times = {node: float('inf') for node in graph.nodes}
    times[start] = 0
    # Dizionario per tenere traccia del predecessore di ciascun nodo nel percorso più breve
    predecessors = {node: None for node in graph.nodes}

    while queue:
        # Estraiamo il nodo con il tempo minimo
        current_time, current_node = heapq.heappop(queue)

        # Se il tempo corrente è maggiore del tempo noto, ignoriamo questo nodo
        if current_time > times[current_node]:
            continue

        # Se raggiungiamo il nodo di fine, interrompiamo l'algoritmo
        if current_node == end:
            break

        # Iteriamo sui vicini del nodo corrente
        for neighbor in graph.neighbors(current_node):
            edge_data = graph.get_edge_data(current_node, neighbor)
            if edge_data and 'flowSpeed' in edge_data[0] and 'length' in edge_data[0]:
                weight = edge_data[0]['length'] / (
                            edge_data[0]['flowSpeed'] * 0.44704)  # flowSpeed è mph -> tramuto in m al secondo

            time = current_time + weight
            # Se troviamo un percorso più breve per raggiungere il vicino, aggiorniamo il tempo
            if time < times[neighbor]:
                times[neighbor] = time
                predecessors[neighbor] = current_node
                heapq.heappush(queue, (time, neighbor))

    # Costruiamo il percorso più breve come lista
    shortest_path = []
    node = end
    while node is not None:
        shortest_path.insert(0, node)
        node = predecessors[node]

    return shortest_path

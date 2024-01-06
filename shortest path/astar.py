from heapq import heappop, heappush
from itertools import count

__all__ = ["astar_path"]

def _weight_function(G, weight):
    """Restituisce una funzione che restituisce il peso di un arco.

    Parametri
    ----------
    G : grafo.

    weight : stringa o funzione che rappresenta il peso di un arco.

    Restituisce
    -------
    function
        Questa funzione restituisce una funzione richiamabile che accetta esattamente tre input:
        un nodo, un nodo adiacente al primo e l'attributo dell'arco
        dizionario per l'arco che unisce quei nodi. Tale funzione restituisce
        un numero che rappresenta il peso di un arco.

    """
    if callable(weight):
        return weight

    if G.is_multigraph():
        return lambda u, v, d: min(attr.get(weight, 1) for attr in d.values())
    return lambda u, v, data: data.get(weight, 1)


def astar_path(G, source, target, heuristic=None, weight="weight"):
    """Restituisce una lista di nodi in un percorso più breve tra sorgente e destinazione
    utilizzando l'algoritmo A* ("A-star").

    Parametri
    ----------
    G : grafo

    source : nodo
       Nodo di partenza per il percorso

    target : nodo
       Nodo finale per il percorso

    heuristic : funzione
       Una funzione per valutare la stima della distanza
       da un nodo alla destinazione.

    weight : stringa o funzione che rappresenta il peso di un arco.

    """
    if source not in G or target not in G:
        msg = f"Sia la sorgente {source} che la destinazione {target} non sono presenti nel grafo"
        raise Exception(msg)

    if heuristic is None:
        # L'euristica predefinita è h=0 - la stessa di Dijkstra
        def heuristic(u, v):
            return 0

    push = heappush
    pop = heappop
    weight = _weight_function(G, weight)

    G_succ = G._adj  # Per velocizzare (e funziona per grafi diretti e non diretti)

    c = count()
    queue = [(0, next(c), source, 0, None)]
    enqueued = {}
    explored = {}

    while queue:
        _, __, curnode, dist, parent = pop(queue)

        if curnode == target:
            path = [curnode]
            node = parent
            while node is not None:
                path.append(node)
                node = explored[node]
            path.reverse()
            return path

        if curnode in explored:
            if explored[curnode] is None:
                continue

            # Salta i percorsi errati che sono stati inseriti in coda prima di trovare uno migliore
            qcost, h = enqueued[curnode]
            if qcost < dist:
                continue

        explored[curnode] = parent

        for neighbor, w in G_succ[curnode].items():
            cost = weight(curnode, neighbor, w)
            if cost is None:
                continue
            ncost = dist + cost
            if neighbor in enqueued:
                qcost, h = enqueued[neighbor]
                # se qcost <= ncost, un percorso meno costoso dal
                # vicino alla sorgente è stato già determinato.
                # Pertanto, non cercheremo di inserire questo vicino
                # nella coda
                if qcost <= ncost:
                    continue
            else:
                h = heuristic(neighbor, target)
            enqueued[neighbor] = ncost, h
            push(queue, (ncost + h, next(c), neighbor, ncost, curnode))

    raise Exception(f"Nodo {target} non raggiungibile da {source}")
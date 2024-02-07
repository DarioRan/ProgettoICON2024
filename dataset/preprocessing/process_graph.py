import xml.etree.ElementTree as ET
import random
import re

# Funzione per calcolare il flowSpeed
def calculate_flow_speed(maxspeed, highway):
    if isinstance(maxspeed, list):
        # Estrai il valore più alto dalla lista di maxspeed
        maxspeed_values = [int(value.strip("[]u'")) for value in maxspeed]
        maxspeed_value = max(maxspeed_values)
    elif maxspeed:
        # Estrai il valore numerico da maxspeed
        match = re.search(r'\d+', maxspeed)
        if match:
            maxspeed_value = int(match.group())
        else:
            # Se non trova un numero valido, ritorna un valore di default
            return 20
    else:
        maxspeed_value = None

    if maxspeed_value:
        # Genera un valore casuale tra maxspeed - 10 e maxspeed come flowSpeed
        return random.randint(maxspeed_value - 10, maxspeed_value)
    else:
        if "[" in highway:
            # Se l'highway è una lista di tipi, prendi il primo tipo
            highway_types = [type.strip("[]u'") for type in highway.split(", ")]
            highway = highway_types[0]
        # Mappa i tipi di strade con le velocità medie approssimative in km/h
        speed_dict = {
            "motorway": 75,
            "trunk": 60,
            "primary": 50,
            "secondary": 35,
            "tertiary": 30,
            "unclassified": 25,
            "residential": 20,
            "service": 12,
            "living_street": 5
        }
        # Ritorna il valore di velocità corrispondente al tipo di strada nel dizionario
        return random.randint(speed_dict.get(highway, 20) - 10, speed_dict.get(highway, 20))


# Percorso del file GraphML
graphml_file = '../old/newyork.graphml'

# Parsa il file GraphML
tree = ET.parse(graphml_file)
root = tree.getroot()

# Namespace
ns = {'graphml': 'http://graphml.graphdrawing.org/xmlns'}

# Trova tutti gli archi nel grafo
edges = root.findall('.//graphml:edge', ns)

# Itera su tutti gli archi
for edge in edges:
    # Trova il campo maxspeed e highway
    maxspeed_elem = edge.find('.//graphml:data[@key="d16"]', ns)
    highway_elem = edge.find('.//graphml:data[@key="d11"]', ns)

    # Se il campo maxspeed è presente, calcola il flowSpeed
    if maxspeed_elem is not None:
        maxspeed = maxspeed_elem.text
        highway = highway_elem.text
        flow_speed = calculate_flow_speed(maxspeed, highway)
        # Aggiungi il campo flowSpeed al nodo
        flow_speed_elem = ET.Element('data', attrib={'key': 'd23'})
        flow_speed_elem.text = str(flow_speed)
        edge.append(flow_speed_elem)
    else:
        # Se maxspeed non è presente, usa direttamente highway per calcolare flowSpeed
        highway = highway_elem.text
        flow_speed = calculate_flow_speed(None, highway)
        # Aggiungi il campo flowSpeed al nodo
        flow_speed_elem = ET.Element('data', attrib={'key': 'd23'})
        flow_speed_elem.text = str(flow_speed)
        edge.append(flow_speed_elem)

# Salva il file GraphML con le modifiche
tree.write('newyork_final.graphml', encoding='utf-8', xml_declaration=True)

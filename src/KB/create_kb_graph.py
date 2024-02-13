import xml.etree.ElementTree as ET


def parse_graphml(filename):
    """
    Parse a GraphML file and return the nodes and edges

    :param filename: the name of the GraphML file to parse

    :return: a tuple containing the nodes and edges
    """
    tree = ET.parse(filename)
    root = tree.getroot()

    nodes = []
    edges = []

    # Estrai nodi e archi dal file GraphML
    for node in root.findall('.//{http://graphml.graphdrawing.org/xmlns}node'):
        node_id = node.get('id')
        data = {elem.get('key'): elem.text for elem in node.findall('.//{http://graphml.graphdrawing.org/xmlns}data')}
        nodes.append((node_id, data))

    for edge in root.findall('.//{http://graphml.graphdrawing.org/xmlns}edge'):
        source = edge.get('source')
        target = edge.get('target')
        data = {elem.get('key'): elem.text for elem in edge.findall('.//{http://graphml.graphdrawing.org/xmlns}data')}
        edges.append((source, target, data))

    return nodes, edges


def add_prolog_queries(output_filename):
    """
    Add Prolog queries to the output file
    :param output_filename:

    :return:

    """
    with open(output_filename, 'a') as f:
        f.write("\n% Queries\n")
        f.write("get_all_nodes(Nodes) :- findall([Osmid, Lat, Lon], node(Osmid, Lat, Lon), Nodes).\n")
        f.write("get_street_name(Source, Target, StreetName) :- edge(Source, Target, _, _, StreetName, _).\n")
        f.write("get_edge_length(Source, Target, Length) :- edge(Source, Target, Length, _, _, _).\n")
        f.write("get_neighbors(Source, Neighbors) :- findall(Neighbor, edge(Source, Neighbor, _, _, _, _), Neighbors).\n")
        f.write("get_edge_flowSpeed(Source, Target, FlowSpeed) :- edge(Source, Target, _, _, _, FlowSpeed).\n")


def generate_prolog_facts(nodes, edges):
    """
    Generate Prolog facts from the nodes and edges

    :param nodes: the nodes of the graph

    :param edges: the edges of the graph

    """
    prolog_facts = []

    for node_id, data in nodes:
        prolog_fact = "node('{}', '{}', '{}').".format(str(node_id), str(data.get('d3', '')), str(data.get('d4', '')))
        prolog_facts.append(prolog_fact)

    for source, target, data in edges:
        # Rimuovi eventuali virgolette singole presenti nei nomi delle strade
        street_name = data.get('d12', '').replace("u'", "")
        street_name = street_name.replace("'", "")
        # Aggiungi le virgolette singole per gli attributi delle strade
        edge_data = [data.get(attr, '') for attr in ['d8', 'd10', 'd12', 'd23']]
        # Aggiungi street_name al posto giusto nella lista edge_data
        edge_data[2] = "'{}'".format(street_name)
        prolog_fact = "edge('{}', '{}', '{}', '{}', {}, '{}').".format(source, target, *edge_data)
        prolog_facts.append(prolog_fact)

    return prolog_facts


def write_prolog_file(prolog_facts, output_filename):
    """
    Write the Prolog facts to a file

    :param prolog_facts: the Prolog facts to write

    :param output_filename: the name of the file to write to

    """
    with open(output_filename, 'w') as f:
        for fact in prolog_facts:
            f.write(fact + '\n')


if __name__ == "__main__":
    graphml_file = "../dataset/newyork_final.graphml"
    nodes, edges = parse_graphml(graphml_file)
    prolog_facts = generate_prolog_facts(nodes, edges)
    write_prolog_file(prolog_facts, "knowledge_base_graph.pl")
    add_prolog_queries("knowledge_base_graph.pl")

import TFLiteParser
import os

# Function to calculate duplication statistics op operators i.e.
# number of unique tensors and operators accross models 
def get_duplication(tflite_parser):
    node_vis = dict()
    edge_vis = dict()

    total_nodes = 0
    duplicated_nodes = 0

    total_edges = 0
    duplicated_edges = 0

    for file in os.listdir("models/"):
        graph = tflite_parser.parse_graph("models/" + file)

        # Calculating number of of total nodes and duplicated nodes in each graph
        total_nodes += len(graph.nodes)
        total_edges += len(graph.get_traversed_edges())

        for node in graph.nodes:
            key = node.serialize()

            if key not in node_vis:
                node_vis[key] = 1
            else:
                duplicated_nodes += 1

        for edge in graph.get_traversed_edges():
            key = edge.serialize()

            if key not in edge_vis:
                edge_vis[key] = 1
            else:
                duplicated_edges += 1

    print("Total nodes:", total_nodes, "Duplicated nodes:", duplicated_nodes)
    print("Total edges:", total_edges, "Duplicated edges:", duplicated_edges)

if __name__ == "__main__":

    tflite_parser = TFLiteParser.TFLiteParser()

    get_duplication(tflite_parser)

    graph = tflite_parser.parse_graph("models/smartreply.tflite")

    print("Number of inputs:", graph.num_inputs)
    print("Number of outputs:", graph.num_outputs)
    print("Max fan-in:", graph.max_fan_in)
    print("Max fan-out:", graph.max_fan_out)
    graph.print_graph()
    graph.print_nodes()
    graph.print_edges()
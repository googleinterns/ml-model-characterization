import TFLiteParser
import os
from common import Storage
import argparse

import timeit

# Function to calculate duplication statistics op operators i.e.
# number of unique tensors and operators accross files in models directory
def get_duplication():
    tflite_parser = TFLiteParser.TFLiteParser()
    node_vis = dict()
    edge_vis = dict()

    total_nodes = 0
    duplicated_nodes = 0

    total_edges = 0
    duplicated_edges = 0

    for file in os.listdir("models/"):
        graph = tflite_parser.parse_graph("models/" + file, None, None)

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

    print("Duplication stats")
    print("Total nodes:", total_nodes, "Duplicated nodes:", duplicated_nodes)
    print("Total traversed edges:", total_edges, "Duplicated traversed edges:", duplicated_edges)
    print()

# Function to load data in models/filename to spanner DB
def load_data(filename, model_name, category):
    tflite_parser = TFLiteParser.TFLiteParser()
    graph = tflite_parser.parse_graph("models/" + filename, model_name, category)
    storage = Storage.Storage('ml-models-characterization-db', 'models_db')
    storage.load_data(graph)

# Run inference on models/filename and print graph, operators and tensors
def run_inference(filename, model_name = None, category = None):
    tflite_parser = TFLiteParser.TFLiteParser()
    graph = tflite_parser.parse_graph("./models/" + filename, model_name, category)

    print("Name of model:", graph.model_name)
    print("Number of inputs:", graph.num_inputs)
    print("Number of outputs:", graph.num_outputs)
    print("Max fan-in:", graph.max_fan_in)
    print("Max fan-out:", graph.max_fan_out)
    graph.print_graph()
    graph.print_nodes()
    graph.print_edges()

if __name__ == "__main__":
    # filename and model_name are arguments to the command line
    # filename must be present in the models directory
    # modelname must be unique for every model added to the database
    # category is the category of problem the model solves ex. Object Detection
    parser = argparse.ArgumentParser()
    parser.add_argument('filename')
    parser.add_argument('model_name')
    parser.add_argument('category') 
    args = parser.parse_args()

    filename = args.filename
    model_name = args.model_name
    category = args.category

    # load_data(filename, model_name, category)

    # get_duplication()

    run_inference(filename, model_name, category)
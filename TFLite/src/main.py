from common import Storage
import TFLiteParser
import os
import argparse

import timeit

# Function to load data in models/filename to spanner DB
def load_data(filename, model_name, category):
    tflite_parser = TFLiteParser.TFLiteParser()
    graph = tflite_parser.parse_graph("./TFLite/models/" + filename, model_name, category)
    storage = Storage.Storage('ml-models-characterization-db', 'models_db')
    storage.load_data(graph)

# Run inference on models/filename and print graph, operators and tensors
def run_inference(filename, model_name = None, category = None):
    tflite_parser = TFLiteParser.TFLiteParser()
    graph = tflite_parser.parse_graph("./TFLite/models/" + filename, model_name, category)

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
    parser.add_argument('--filename')
    parser.add_argument('--model_name')
    parser.add_argument('--category') 
    args = parser.parse_args()

    filename = args.filename
    model_name = args.model_name
    category = args.category

    # load_data(filename, model_name, category)

    # get_duplication()

    run_inference(filename, model_name, category)
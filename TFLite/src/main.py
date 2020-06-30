""" main module to run inference or load data into database

Contains functions to load model data into database or run inference 
and print it.

Attributes:
    MODELS_DIR (str) : Directory in which module will look for --filename

CLA to module and common args to all functions in module:
    filename (str) : name of the file to be parsed. Must be present in 
        models directory.
    model_name (str) : unique model name of the model being parsed.
    category (str) : problem category of the model.
    sub_category (str) : problem sub category of the model.
    is_canonical (str) : Boolean to separate unique architectures 
            from duplicates, The first model to be inserted into database
            with a specific architecture will have this to be "True", the other
            models with same architecture will have this to be "False", 
            defaults to "False". 
"""

import argparse
import os

from common import Storage
import TFLiteParser

MODELS_DIR = "./TFLite/models/"

def load_data(filename, model_name, category, sub_category, is_canonical):
    """Function to parse TFLite file and load data into spanner database

    Parses a TFLite file into a Graph object and stores the Graph into a 
    spanner database.
    """

    tflite_parser = TFLiteParser.TFLiteParser()
    graph = tflite_parser.parse_graph(MODELS_DIR + filename, 
                model_name, category, sub_category)

    # Constants for spanner DB details
    INSTANCE_ID = 'ml-models-characterization-db'
    DATABASE_ID = 'models_db'

    storage = Storage.Storage(INSTANCE_ID, DATABASE_ID)
    storage.load_data(graph, is_canonical)

def run_inference(filename, model_name, category, sub_category):
    """Function to parse TFLite file and print graph information

    Parses a TFLite file and prints the metadata, graph structure, nodes and edges
    """
    tflite_parser = TFLiteParser.TFLiteParser()
    graph = tflite_parser.parse_graph(MODELS_DIR + filename, 
                model_name, category, sub_category)

    print("Name of model:", graph.model_name)
    print("Number of inputs:", graph.num_inputs)
    print("Number of outputs:", graph.num_outputs)
    print("Max fan-in:", graph.max_fan_in)
    print("Max fan-out:", graph.max_fan_out)
    graph.print_graph()
    graph.print_nodes()
    graph.print_edges()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--filename')
    parser.add_argument('--model_name')
    parser.add_argument('--category', default = "None") 
    parser.add_argument('--sub_category', default = "None")
    parser.add_argument('--is_canonical', default = "False")
    args = parser.parse_args()

    filename = args.filename
    model_name = args.model_name
    category = args.category
    sub_category = args.sub_category
    is_canonical = args.is_canonical

    # load_data(filename, model_name, category, sub_category, is_canonical)

    run_inference(filename, model_name, category, sub_category)
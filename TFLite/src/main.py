"""Main module to run inference or load data into database.

Contains functions to load model data into database or run inference 
and print it.

Attributes:
    MODELS_DIR (str) : Directory in which module will look for --filename

CLA to module and common args to all functions in module:
    filename (str) : Name of the file to be parsed. Must be present in 
        models directory.
    model_name (str) : Unique model name of the model being parsed.
    category (str) : Problem category of the model.
    sub_category (str) : Problem sub category of the model.
    model_type (str) :String to denote type of model architecture, if a 
                model is the first of its architecture, then value is set to
                "canonical", if it is a module then set to "module", else 
                "additional".
"""

import argparse
import os

from common import Storage
from TFLite.src import TFLiteParser

MODELS_DIR = "./TFLite/models/"

def load_data(filename, model_name, category, sub_category, model_type):
    """Function to parse TFLite file and load data into spanner database.

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
    storage.load_data(graph, model_type)

def run_inference(filename, model_name, category, sub_category):
    """Function to parse TFLite file and print graph information.

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
    parser.add_argument('--filename', required = True)
    parser.add_argument('--model_name', required = True)
    parser.add_argument('--category', default = "None") 
    parser.add_argument('--sub_category', default = "None")
    parser.add_argument('--model_type', default = "canonical")
    args = parser.parse_args()

    filename = args.filename
    model_name = args.model_name
    category = args.category
    sub_category = args.sub_category
    model_type = args.model_type

    # load_data(filename, model_name, category, sub_category, model_type)

    run_inference(filename, model_name, category, sub_category)
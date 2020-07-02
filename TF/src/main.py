""" main module to run inference or load data into database

Contains functions to load model data into database or run inference 
and print it.

Attributes:
    MODELS_DIR (str) : Directory in which module will look for --filename

CLA to module and Args to all functions in module:
    filename (str): name of the file to be parsed. Must be present in 
        models directory.
    model_name (str): unique model name of the model being parsed.
    category (str): problem category of the model.
    is_saved_model (str, optional): "True" if file is in SavedModel format, 
        defaults to "True".
    input_operation_names (list of str, optional) : Names of the operations 
        that are inputs to the model, defaults to [].

"""

import argparse

from common import Storage
import TFParser

MODELS_DIR = "./TF/models/"

def load_data(filename, model_name, category, is_saved_model, input_operation_names):
    """Function to parse TF file and load data into spanner database

    Parses a TF file (SavedModel or FrozenGraph format) into a Graph object 
    and stores the Graph into a spanner database.
    """

    tf_parser = TFParser.TFParser()
    graph = tf_parser.parse_graph(MODELS_DIR + filename, model_name, 
                category, is_saved_model, input_operation_names)

    if graph == None:
        print("Parsing failed, model not loaded")

    # Constants for spanner DB details
    INSTANCE_ID = 'ml-models-characterization-db'
    DATABASE_ID = 'models_db'

    storage = Storage.Storage(INSTANCE_ID, DATABASE_ID)
    storage.load_data(graph)

def run_inference(filename, model_name, category, is_saved_model, input_operation_names):
    """Function to parse TF file and print graph information

    Parses a TF file (SavedModel or FrozenGraph format) and prints the 
    metadata, graph structure, nodes and edges
    """
    tflite_parser = TFParser.TFParser()
    graph = tflite_parser.parse_graph(MODELS_DIR + filename, model_name, 
                category, is_saved_model, input_operation_names)

    if graph == None:
        print("Parsing failed, cannot run inference")
        return

    print("Name of model:", graph.model_name)
    print("Number of inputs:", graph.num_inputs)
    print("Number of outputs:", graph.num_outputs)
    print("Max fan-in:", graph.max_fan_in)
    print("Max fan-out:", graph.max_fan_out)
    graph.print_graph()
    graph.print_nodes()
    # graph.print_edges()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--filename')
    parser.add_argument('--model_name')
    parser.add_argument('--category', default = "None") 
    parser.add_argument('--is_saved_model', default = "True")
    parser.add_argument('--input_operation_names', nargs="+", default = [])
    args = parser.parse_args()

    filename = args.filename
    model_name = args.model_name
    category = args.category
    is_saved_model = args.is_saved_model
    input_operation_names = args.input_operation_names

    # load_data(filename, model_name, category, is_saved_model, input_operation_names)

    run_inference(filename, model_name, category, is_saved_model, input_operation_names)
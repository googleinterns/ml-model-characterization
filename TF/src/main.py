import TFParser
import argparse
from common import Storage

def load_data(filename, model_name, category, is_saved_model):
    tf_parser = TFParser.TFParser()
    graph = tf_parser.parse_graph("models/" + filename, model_name, category, is_saved_model)
    storage = Storage.Storage('ml-models-characterization-db', 'models_db')
    storage.load_data(graph)

# Run inference on models/filename and print graph, operators and tensors
def run_inference(filename, model_name, category, is_saved_model):
    tflite_parser = TFParser.TFParser()
    graph = tflite_parser.parse_graph("./models/" + filename, model_name, category, is_saved_model)

    # print("Name of model:", graph.model_name)
    # print("Number of inputs:", graph.num_inputs)
    # print("Number of outputs:", graph.num_outputs)
    # print("Max fan-in:", graph.max_fan_in)
    # print("Max fan-out:", graph.max_fan_out)
    graph.print_graph()
    graph.print_nodes()
    # graph.print_edges()

if __name__ == "__main__":
    # filename and model_name are arguments to the command line
    # filename must be present in the models directory
    # modelname must be unique for every model added to the database
    # category is the category of problem the model solves ex. Object Detection
    # is_saved model is either True or False, denoting whether the .pb file 
    # is a frozen graph or saved model
    parser = argparse.ArgumentParser()
    parser.add_argument('filename')
    parser.add_argument('model_name')
    parser.add_argument('category') 
    parser.add_argument('is_saved_model')
    args = parser.parse_args()

    filename = args.filename
    model_name = args.model_name
    category = args.category
    is_saved_model = args.is_saved_model

    load_data(filename, model_name, category, is_saved_model)

    # get_duplication()

    # run_inference(filename, model_name, category, is_saved_model)



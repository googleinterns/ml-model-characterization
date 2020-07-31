""" Module to find similar models/modules in database to a input model/module

Module to display topk similar models/modules in database to an input parsed 
model/module. The input model/module is not stored in the database, it is 
fitted to graph2vec with the models/modules present in the database and 
cosine similarity is computed using embeddings.

CLA to module and Args to functions in module:
    file_path (str) : Full path to the model/module to be parsed.
    file_format (str) : String to denote the file format of the input model/module.
        One of "TFFrozenGraph", "TFSavedModel" or "TFLite".
    model_name (str) : Name of the input model/module.
    include_edge_attrs (str) : Case insensitive string to denote whether to 
        include edge attributes of input edge from _EDGE_ATTRS in feature,
        if "true" then they are included.
    include_node_attrs (str) : Case insensitive string to denote whether to 
        include node attributes from _NODE_ATTRS in feature, if "true" then 
        they are included.
    wl_iterations (int) : Depth of subgraph rooted at every node to be 
        considered for feature building in graph2vec.
"""

from sklearn import metrics
import argparse
import numpy

from common import Storage
from common import Vectorize
from TF.src import TFParser
from TFLite.src import TFLiteParser

def topk_similar_in_db(file_path, file_format, model_name, include_node_attrs,
                        include_edge_attrs, topk):
    """ Module to print topk similar models to a file

    Takes an input model/module and a list of model_graphs present in the 
    database, parses the file and adds its graph to the models graphs and
    obtains embeddings for the collective list. Then prints the topk models
    similar to the input model/module from the database models.
    """

    # Instance and database ID of spanner database which holds the models
    INSTANCE_ID = 'ml-models-characterization-db'
    DATABASE_ID = 'models_db'

    # Parsing models from database into Graph objects
    storage = Storage.Storage(INSTANCE_ID, DATABASE_ID)
    model_graphs = storage.parse_models()

    # Parsing file_path depending on file_format
    if file_format == "TFFrozenGraph":
        parser = TFParser.TFParser()
        graph = parser.parse_graph(file_path, model_name, None, None, "False", [])

    elif file_format == "TFSavedModel":
        parser = TFParser.TFParser()
        graph = parser.parse_graph(file_path, model_name, None, None, "True", [])

    elif file_format == "TFLite":
        parser = TFLiteParser.TFLiteParser()
        graph = parser.parse_graph(file_path, model_name, None, None)

    else:
        print("File format provided is not supported,"
                " must be one of TFLite, TFSavedFormat and TFFrozenGraph.")

    if graph == None:
        print("Failed to parse file.")
        return

    # Adding current graph to database models to fit to graph2vec
    model_graphs.append(graph)

    # Getting graph2vec embeddings
    vectorize = Vectorize.Vectorize()
    model_graphs, embeddings = vectorize.get_graph2vec_embeddings(
        model_graphs, include_edge_attrs, include_node_attrs, wl_iterations)

    # Cosine similarity
    similarity = metrics.pairwise.cosine_similarity(
        [embeddings[len(embeddings) - 1]], embeddings)
    
    model_graph = model_graphs[len(model_graphs) - 1]

    print(model_graph.model_name)
    indices = numpy.argsort(-similarity[0])

    num_models = len(model_graphs)

    # printing minimum of topk and len(model_graphs) models
    for rank in range(1,min(topk + 1, num_models)):
        print("\t", end = '')
        print(similarity[0][indices[rank]],
                model_graphs[indices[rank]].model_name)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_path', required = True)
    parser.add_argument('--file_format', required = True, 
                        choices = ["TFLite", "TFSavedModel", "TFFrozenGraph"])
    parser.add_argument('--model_name', default = "input_file")
    parser.add_argument('--include_edge_attrs', default = "False")
    parser.add_argument('--include_node_attrs', default = "True")
    parser.add_argument('--wl_iterations', type = int, default = 3)
    args = parser.parse_args()

    file_path = args.file_path
    file_format = args.file_format
    model_name = args.model_name
    include_edge_attrs = args.include_edge_attrs
    include_node_attrs = args.include_node_attrs
    wl_iterations = args.wl_iterations

    # Number of top similar models to print
    TOPK = 20

    topk_similar_in_db(file_path, file_format, model_name, 
                        include_node_attrs, include_edge_attrs, TOPK)
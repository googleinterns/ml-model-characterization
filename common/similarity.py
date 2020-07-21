"""Module to to display cosine similarity.

Module to display cosine similarity in models in database.
Also contains function to load graph embeddings into Models table.

CLA to module:
    include_edge_attrs (str) : Case insensitive string to denote whether to 
        include edge attributes of input edge from _EDGE_ATTRS in feature,
        if "true" then they are included.
    include_node_attrs (str) : Case insensitive string to denote whether to 
        include node attributes from _NODE_ATTRS in feature, if "true" then 
        they are included.
    wl_iterations (int) : Depth of subgraph rooted at every node to be 
        considered for feature building in graph2vec.
"""

from sklearn import cluster
from sklearn import metrics
import argparse
import numpy

import Storage
import Vectorize

def topk_similarity(model_graphs, embeddings, topk):
    """Function to print topk similar models to each model.

    Args:
        model_graphs (list of Graph objects) : List of Graph objects 
            representing the models and modules in the database.
        embeddings (list of vectors) : Embeddings with index correspondence
            to model_graph.
        topk (int) : Number of top similar models to print. 
    """

    similarity = metrics.pairwise.cosine_similarity(embeddings)
    for index, model_graph in enumerate(model_graphs):            
        print(model_graph.model_name)
        indices = numpy.argsort(-similarity[index])

        num_models = len(model_graphs)

        # printing minimum of topk and len(model_graphs) models
        for rank in range(1,min(topk + 1, num_models)):
            print("\t", end = '')
            print(similarity[index][indices[rank]],
                    model_graphs[indices[rank]].model_name)

def db_module_similarity(model_graphs, embeddings, topk):
    """Module to print topk similar models to each module in database.

    Args:
        model_graphs (list of Graph objects) : List of Graph objects 
            representing the models and modules in the database.
        embeddings (list of vectors) : Embeddings with index correspondence
            to model_graph.
        topk (int) : Number of top similar models to print. 
    """
    similarity = metrics.pairwise.cosine_similarity(embeddings)
    for index, model_graph in enumerate(model_graphs):  
        if(model_graph.model_name.endswith("module")):          
            print(model_graph.model_name)
            indices = numpy.argsort(-similarity[index])

            num_models = len(model_graphs)

            # printing minimum of topk and len(model_graphs) models
            for rank in range(1,min(topk + 1, num_models)):
                print("\t", end = '')
                print(similarity[index][indices[rank]],
                        model_graphs[indices[rank]].model_name)

def store_embeddings(model_graphs, embeddings, instance_id, database_id):
    """Function to store embeddings into Models table.

    Args:
        models_graphs (list of Graph objects) : List of Graph objects 
            representing the models and modules in the database.
        embeddings (list of vectors) : Embeddings with index correspondence
            to model_graph. 
        instance_id (str) : Id of the spanner instance.
        database_id (str) : Id of the database within the spanner instance.
    """

    storage = Storage.Storage(instance_id, database_id)
    storage.load_embeddings(model_graphs, embeddings)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--include_edge_attrs', default = "False")
    parser.add_argument('--include_node_attrs', default = "True")
    parser.add_argument('--wl_iterations', default = 3)
    args = parser.parse_args()

    include_edge_attrs = args.include_edge_attrs
    include_node_attrs = args.include_node_attrs
    wl_iterations = args.wl_iterations

    # Instance and database ID of spanner database which holds the models
    INSTANCE_ID = 'ml-models-characterization-db'
    DATABASE_ID = 'models_db'

    # Parsing models from database into Graph objects
    storage = Storage.Storage(INSTANCE_ID, DATABASE_ID)
    model_graphs = storage.parse_models()

    # Getting graph embeddings
    vectorize = Vectorize.Vectorize()
    model_graphs, embeddings = vectorize.get_graph2vec_embeddings(
        model_graphs, include_edge_attrs, include_node_attrs, wl_iterations)

    # Number of top similar models to print
    TOPK = 20

    topk_similarity(model_graphs, embeddings, TOPK)
    db_module_similarity(model_graphs, embeddings, TOPK)
    # store_embeddings(model_graphs, embeddings, INSTANCE_ID, DATABASE_ID)
""" module to obtain Graph embeddings and cosine similarity database models

CLA to module:
    include_edge_attrs (str) : case insensitive string to denote whether to 
        include edge attributes of input edge from _EDGE_ATTRS in feature,
        if "true" then they are included.
    include_node_attrs (str) : case insensitive string to denote whether to 
        include node attributes from _NODE_ATTRS in feature, if "true" then 
        they are included.
"""

from sklearn import cluster
from sklearn import metrics
import argparse
import numpy

import Vectorize

def print_topk_similarity(model_graphs, embeddings, topk):
    """ Function to print topk similar models to each model

    Args:
        models_graphs (list of Graph objects) : List of Graph objects 
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

def print_module_similarity(model_graphs, embeddings, topk):
    """ Module to print topk similar models to each module in database

    Args:
        models_graphs (list of Graph objects) : List of Graph objects 
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--include_edge_attrs', default = "False")
    parser.add_argument('--include_node_attrs', default = "True")
    args = parser.parse_args()

    include_edge_attrs = args.include_edge_attrs
    include_node_attrs = args.include_node_attrs

    # Instance and database ID of spanner database which holds the models
    INSTANCE_ID = 'ml-models-characterization-db'
    DATABASE_ID = 'models_db'

    # Getting graph embeddings
    vectorize = Vectorize.Vectorize(INSTANCE_ID, DATABASE_ID)
    model_graphs, embeddings = vectorize.get_graph2vec_embeddings(
        include_edge_attrs, include_node_attrs)

    # Number of top similar models to print
    TOPK = 20

    print_topk_similarity(model_graphs, embeddings, TOPK)
    print_module_similarity(model_graphs, embeddings, TOPK)
"""Module to obtain Graph embeddings of a list of model graphs"""

from google.cloud import spanner
from karateclub import Graph2Vec
import networkx
import time

class Vectorize:
    """Embedding class to create graph embeddings.

    Creates embeddings from graph structures of models.
    """

    _EDGE_ATTRS = [
        "tensor_shape", "tensor_type"
        ] # Input edge attributes to be used as features,
        # must be attributes of common/Edge class

    _NODE_ATTRS = [
        "operator_type"
        ] # Node attributes to be used as features,
        # must be attributes of common/Node class

    def __init__(self):
        pass

    def _graph_to_networkx_graph(self, graph, include_edge_attrs, 
                                    include_node_attrs):
        """Internal function to convert Graph object to networkx graph.

        Also adds a dummy node which is connected to all inputs
        if the graph is not connected.

        Arguments:
            graph (Graph object) : Graph object to be converted to networkx
            include_edge_attrs (str) : Case insensitive string to denote 
                whether to include edge attributes of input edge from 
                _EDGE_ATTRS in feature, if "true" then they are included.
            include_node_attrs (str) : Case insensitive string to denote 
                whether to include node attributes from _NODE_ATTRS in feature,
                if "true" then they are included.

        Returns:
            networkx graph corresponding to the converted Graph object        
        """

        # Initializing networkx graph
        networkx_graph = networkx.Graph(
            model_name = graph.model_name, category = graph.category,
            sub_category = graph.sub_category, source = graph.source
            )

        input_edges = dict()

        # Adding edges to the graph
        for src_node_index in graph.adj_list.keys():
            for [edge_index, dest_node_index] in graph.adj_list[src_node_index]:
                networkx_graph.add_edge(
                    src_node_index, dest_node_index,
                    tensor_shape = graph.edges[edge_index].tensor_shape,
                    tensor_type = graph.edges[edge_index].tensor_type)
                
                if dest_node_index not in input_edges:
                    input_edges.update({dest_node_index : []})
            
                input_edges[dest_node_index].append(graph.edges[edge_index])
        
        # Adding all nodes to the graph, and building features
        for index in range(len(graph.nodes)):
            features = list()

            if include_node_attrs.lower() == "true":
                for node_attr in self._NODE_ATTRS:
                    features.append(str(getattr(graph.nodes[index], node_attr)))

            if include_edge_attrs.lower() == "true" and index in input_edges:
                for input_edge in input_edges[index]:
                    for edge_attr in self._EDGE_ATTRS:
                        features.append(str(getattr(input_edge, edge_attr)))

            concat_feature = " ".join(features)

            networkx_graph.add_node(
                index, feature = concat_feature)

        # If graph is not connected, introduce a dummy node and connect all 
        # inputs to it to make it connected.
        if not networkx.algorithms.is_connected(networkx_graph):
            max_node_num = max(networkx_graph.nodes())

            networkx_graph.add_node(max_node_num + 1, feature = "Dummy")

            for index in graph.start_node_indices:
                networkx_graph.add_edge(max_node_num + 1, index)

        return networkx_graph

    def get_graph2vec_embeddings(self, model_graphs, include_edge_attrs, 
                                    include_node_attrs, wl_iterations):
        """Getting embeddings using Graph2Vec.

        Converts the model graphs into networkx graphs and uses graph2vec
        for getting embeddings for them.

        Arguments:
            model_graphs (list of Graph objects) : List of model graphs to get
                embeddings for.
            include_edge_attrs (str) : Case insensitive string to denote 
                whether to include edge attributes of input edge from 
                _EDGE_ATTRS in feature, if "true" then they are included.
            include_node_attrs (str) : Case insensitive string to denote
                whether to include node attributes from _NODE_ATTRS in feature,
                if "true" then they are included.
            wl_iterations (int) : Depth of subgraph rooted at every node to be 
                considered for feature building in graph2vec.

        Returns:
            list of Graph objects corresponding to models in the database and
            a list of embeddings for the same with index correspondence.
        """

        """Parameters to Graph2Vec.

        WL_ITERATIONS (int) : Depth of subgraph rooted at every node to be 
            considered for feature building in graph2vec.
            Higher values extract more detailed graph structures.
        ATTRIBUTED (bool) : Boolean to denote whether to use 'feature' 
            attribute of node for feature building or use 
            default (degree of node).
        EPOCHS (int) : Number of epochs to train for.
        LEARNING_RATE (int) : Learning rate for training.
        MIN_COUNT (int) : Minimum count of feature occurence for it to be 
            considered in vocabulary.
        DIMENSIONS (int) : Dimension of the embeddings that are obtained.
        WORKERS (int) : Number of cores.
        """      

        # Values have been fine tuned by manual inspection of 
        # graphs in the database.
        WL_ITERATIONS = wl_iterations
        ATTRIBUTED = True
        EPOCHS = 500
        LEARNING_RATE = 0.15
        MIN_COUNT = 5
        DIMENSIONS = 128
        WORKERS = 8

        # Building list of networkx graphs to fit graph2vec
        start_time = time.time()
        networkx_graphs = list()
        for model_graph in model_graphs:
            networkx_graphs.append(
                self._graph_to_networkx_graph(
                    model_graph, include_edge_attrs, include_node_attrs
                    )
                )

        # Fitting models to graph2vec
        start_time = time.time()
        graph2vec = Graph2Vec(
            wl_iterations = WL_ITERATIONS, attributed = ATTRIBUTED, 
            epochs = EPOCHS, learning_rate = LEARNING_RATE, 
            min_count = MIN_COUNT, dimensions = DIMENSIONS, workers = WORKERS
            )
        graph2vec.fit(networkx_graphs)
        print("Time to fit graphs to graph2vec model: %s" % (time.time() - start_time))
        
        embeddings = graph2vec.get_embedding()
        return model_graphs, embeddings
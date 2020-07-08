from google.cloud import spanner
from karateclub import Graph2Vec
from sklearn import metrics
from sklearn import cluster
import networkx
import numpy

import Graph
import Node
import Edge

class Vectorize:
    """ Embedding class to create graph embeddings

    Creates embeddings from graph structures of models, the models are read
    from a database which are taken as class arguments.
    
    Attributes:
        spanner_client (cloud spanner Client object) : Instance to access 
            spanner API for python.
        instance (cloud spanner Instance object): Instance to access the 
            required spanner instance.
        database (cloud spanner Database object): Instance to access the
            database from where the models will be retrieved

    """

    def __init__(self, instance_id, database_id):
        self.spanner_client = spanner.Client()
        self.instance = self.spanner_client.instance(instance_id)
        self.database = self.instance.database(database_id)

    def _parse_models(self):
        """ Internal function to query and read data from database

        Internal function to query database and read models into Graph objects

        Returns:
            list of Graph objects corresponding to the graph objects the models
            in the spanner database have been parsed into.
        """

        model_graphs = list()
        
        # Query to get all models from Models table
        with self.database.snapshot() as snapshot:
            results1 = snapshot.execute_sql(
                "SELECT model_name, category, sub_category, source, num_inputs"
                " FROM Models"
                )

        for row in results1:

            if row[4] == 0:
                continue

            # Extracting model attributes
            model_name = row[0]
            category = row[1]
            sub_category = row[2]
            source = row[3]

            nodes = list()
            edges = list()
            start_node_indices = list()

            adj_list = dict()

            # Querying Operators of model_name
            with self.database.snapshot() as snapshot:
                results2 = snapshot.execute_sql(
                    "SELECT * from Models JOIN Operators"
                    " ON Models.model_name = Operators.model_name"
                    " WHERE Models.model_name = '" + model_name + "'"
                    " ORDER BY operator_id"
                )
            
            # Dictionary to hold which field is in which index of query results
            field_to_index = dict()

            # Boolean to check if field_to_dict needs to be populated
            populate_dicts = True

            # Extracting Node attributes
            for row in results2:
                if populate_dicts:
                    for index in range(len(results2.metadata.row_type.fields)):
                        field_name = results2.metadata.row_type.fields[index].name
                        field_to_index[field_name] = index
                    
                    populate_dicts = False

                new_node = Node.Node(None, None)

                for attr in vars(new_node).keys():
                    if attr in field_to_index:
                        setattr(new_node, attr, row[field_to_index[attr]])

                nodes.append(new_node)

                # populating start_node_indices using is_input field
                if row[field_to_index['is_input']]:
                    start_node_indices.append(len(nodes) - 1)
            
            # Querying Tensors of model_name
            with self.database.snapshot() as snapshot:
                results2 = snapshot.execute_sql(
                    "SELECT * from Models JOIN Tensors"
                    " ON Models.model_name = Tensors.model_name"
                    " WHERE Models.model_name = '" + model_name + "'"
                    " ORDER BY tensor_id"
                )

            # Dictionary to hold which field is in which index of query results
            field_to_index.clear()

            # Boolean to check if field_to_dict needs to be populated
            populate_dicts = True

            # Extracting Edge attributes
            for row in results2:
                if populate_dicts:
                    for index in range(len(results2.metadata.row_type.fields)):
                        field_name = results2.metadata.row_type.fields[index].name
                        field_to_index[field_name] = index
                    
                    populate_dicts = False

                new_edge = Edge.Edge(None, None)

                for attr in vars(new_edge).keys():
                    if attr in field_to_index:
                        setattr(new_edge, attr, row[field_to_index[attr]])

                edges.append(new_edge)

                to_operator_ids = row[field_to_index['to_operator_ids']]
                from_operator_ids = row[field_to_index['from_operator_ids']]

                edge_index = len(edges) - 1

                for src_node_index in from_operator_ids:
                    src_node_index -= 1
                    for dest_node_index in to_operator_ids:
                        dest_node_index -= 1

                        if src_node_index not in adj_list:
                            adj_list.update({src_node_index : []})
                        
                        adj_list[src_node_index].append([edge_index, 
                                                            dest_node_index])

            new_graph = Graph.Graph(nodes, start_node_indices, edges, adj_list, 
                                    model_name, category, sub_category)
            new_graph.source = source

            model_graphs.append(new_graph)

        return model_graphs

    def _graph_to_networkx_graph(self, graph):
        """ Converting Graph object to networkx graph

        Also adds a dummy node which is connected to all inputs
        if the graph is not connected.

        Arguments:
            graph (Graph object) : Graph object to be converted to networkx

        Returns:
            networkx graph corresponding to the converted Graph object        
        """

        networkx_graph = networkx.Graph(
            model_name = graph.model_name, category = graph.category,
            sub_category = graph.sub_category, source = graph.source
            )

        # Adding all nodes to the graph
        for index in range(len(graph.nodes)):
            networkx_graph.add_node(
                index, feature = [graph.nodes[index].operator_type])

        # Adding edges to the graph
        for src_node_index in graph.adj_list.keys():
            for [edge_index, dest_node_index] in graph.adj_list[src_node_index]:
                # print(src_node_index, dest_node_index)
                networkx_graph.add_edge(
                    src_node_index, dest_node_index,
                    tensor_shape = graph.edges[edge_index].tensor_shape,
                    tensor_type = graph.edges[edge_index].tensor_type)

        # If graph is not connected, introduce a dummy node and connect all 
        # inputs to it to make it connected.
        if not networkx.algorithms.is_connected(networkx_graph):
            max_node_num = max(networkx_graph.nodes)

            networkx_graph.add_node(max_node_num + 1, feature = "Dummy")

            for index in graph.start_node_indices:
                networkx_graph.add_edge(max_node_num + 1, index)

        return networkx_graph

    def get_graph2vec_embeddings(self):
        """ Getting embeddings using graph2vec

        Parses models from DB into Graph objects, converts them into networkx
        graphs and uses graph2vec for getting embeddings for them.

        Returns:
            list of Graph objects corresponding to models in the database and
            a list of embeddings for the same.  
        """
        
        model_graphs = self._parse_models()
        networkx_graphs = list()
        for model_graph in model_graphs:
            networkx_graphs.append(self._graph_to_networkx_graph(model_graph))
        
        # Fitting models to graph2vec
        model = Graph2Vec(attributed = True, epochs = 100, 
                            learning_rate = 0.05, min_count = 1)
        model.fit(networkx_graphs)

        # Computing pairwise cosine similarity
        embeddings = model.get_embedding()
        similarity = metrics.pairwise.cosine_similarity(embeddings)


        # Prints atmost top 10 similar models and similarity value for each model
        for index1, model_graph1 in enumerate(model_graphs):            
            print(model_graph1.model_name)
            indices = numpy.argsort(-similarity[index1])

            num_models = len(model_graphs)

            for rank in range(1,min(21, num_models)):
                print("\t", end = '')
                print(similarity[index1][indices[rank]],
                        model_graphs[indices[rank]].model_name)

        return model_graphs, embeddings

if __name__ == "__main__":
    INSTANCE_ID = 'ml-models-characterization-db'
    DATABASE_ID = 'models_db'
    vectorize = Vectorize(INSTANCE_ID, DATABASE_ID)
    model_graphs, embeddings = vectorize.get_graph2vec_embeddings()
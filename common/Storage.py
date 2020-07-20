"""Module with Storage class to store Graph object into a spanner database"""

from google.cloud import spanner
from queue import Queue
import os

from common import Edge
from common import Graph
from common import Node

class Storage:
    """ Storage class to store graph into spanner database

    Stores the Graph object attributes into the Models table, 
    Node object into Operators table and Edge object into the Tensors table.

    Attributes:
        spanner_client (cloud spanner Client object) : Instance to access 
            spanner API for python.
        instance (cloud spanner Instance object): Instance to access the 
            required spanner instance.
        database (cloud spanner Database object): Instance to access the
            database where the data needs to be pushed.

    Args:
        instance_id (str) : Id of the spanner instance.
        database_id (str) : Id of the database within the spanner instance.

    """

    _MAX_MUTATIONS = 20000 # Number of allowed spanner mutations per commit

    def __init__(self, instance_id, database_id):
        self.spanner_client = spanner.Client()
        self.instance = self.spanner_client.instance(instance_id)
        self.database = self.instance.database(database_id)

    def _load_model(self, graph, is_canonical):
        """Internal method to store data into the Models table

        Stores Graph instance attributes pertaining to model metadata into 
        Models table.

        Args:
            graph (Graph object) : The intance of Graph to be stores in 
                the Database.
            is_canonical (str) : String to separate unique architectures 
                from duplicates, The first model to be inserted into database
                with a specific architecture will have this to be "True", the other
                models with same architecture will have this to be "False". 
    
        Returns:
            A boolean, True if commit into database is succesful, False otherwise
        """ 

        # Graph attributes that are not pushed to the database
        NOT_STORED_ATTR = ['nodes', 'start_node_indices', 'edges', 'adj_list'] 

        try:
            # To store the database column names and their values to be inserted
            column_names = ['is_canonical']
            values = [is_canonical]

            attrs = vars(graph)
            for item in attrs.items():
                if (item[0] not in NOT_STORED_ATTR):
                    column_names.append(item[0])
                    values.append(item[1])

            column_names = tuple(column_names)
            values = tuple(values)

            with self.database.batch() as batch:
                batch.insert(
                    table = 'Models',
                    columns = column_names,
                    values = [values]
                )

            return True
        except Exception as e:
            print(e)
            return False 

    def _load_operators(self, graph):
        """Internal method to store data into the Operators table

        Stores attributes of Node instances of given graph, representing 
        operators, into Operators table. Inserts in batches due to mutation 
        restriction in spanner.
        If any of the data batches raises an exception due to an unsuccesful 
        commit, the model being inserted is deleted from the Models table and 
        subsequently all children are deleted from Operators table due to
        ON DELETE CASCADE.

        Args:
            graph (Graph object) : The intance of Graph to be stored in the 
                Database.
    
        Returns:
            A boolean, True if commit into database is succesful, False otherwise
        """ 
        if len(graph.nodes) == 0:
            return True

        # Node attributes that are not pushed to the database
        NOT_STORED_ATTR = ['label', 'value'] 

        try:
            # Surrogate Id for operators and iterating index
            operator_id = 0

            if len(graph.nodes) == 0:
                return

            # Number of mutations per row is the number of attributes being 
            # pushed to database
            # 3 additional attributes, 'model_name' and 'operator_id', 
            # 'is_input' present in db other than the class attributes
            num_attributes = len(vars(graph.nodes[0])) + 3

            # Number of nodes to be processed per batch i.e.
            # floor(max mutations per batch / number of mutations per row)
            num_nodes_per_batch = self._MAX_MUTATIONS // num_attributes

            num_nodes = len(graph.nodes)

            # TO-DO : Add retry logic if a batch fails
            while operator_id < num_nodes:
                with self.database.batch() as batch:
                    for _ in range(num_nodes_per_batch):
                        if operator_id == num_nodes:
                            break

                        # Boolean to denote if operator is input
                        is_input = False
                        if operator_id in graph.start_node_indices:
                            is_input = True

                        node = graph.nodes[operator_id]

                        # To store the database column names and their values 
                        # to be inserted
                        column_names = ['model_name', 'operator_id', 'is_input']
                        values = [graph.model_name, operator_id + 1, is_input]

                        attrs = vars(node)
                        for item in attrs.items():
                            if item[0] not in NOT_STORED_ATTR:
                                column_names.append(item[0])
                                values.append(item[1])

                        column_names = tuple(column_names)
                        values = tuple(values)

                        batch.insert(
                            table = 'Operators',
                            columns = column_names,
                            values =  [values]
                        )
                        operator_id += 1
            return True
        except Exception as e:
            print(e)

            query = "DELETE FROM Models WHERE model_name = \'" + graph.model_name + "\'"
            deleted_rows = self.database.execute_partitioned_dml(
                query
            )

            return False

    def _load_tensors(self, graph):
        """Internal method to store data into the Tensors table

        Stores attributes of Edge instances of given graph, representing 
        tensors, into Tensors table. Inserts in batches due to mutation 
        restriction in spanner.
        Only those edges are commited to database which can be reached by 
        traversal, assuming that unreachable edges represent weight/bias tensors
        etc.
        If any of the data batches raises an exception due to an unsuccesful 
        commit, the model being inserted is deleted from the Models table and 
        subsequently all children are deleted from Operators and Tensors table 
        due to ON DELETE CASCADE.

        Args:
            graph (Graph object) : The intance of Graph to be stored in the 
                Database.
    
        Returns:
            A boolean, True if commit into database is succesful, False otherwise
        """ 
        if len(graph.edges) == 0:
            return True

        # Dictionary to store source nodes and destination nodes of edges
        # Also ensures only traversable edges are pushed to database
        to_nodes = dict()
        from_nodes = dict()

        # Populating to_nodes and from_nodes using BFS
        adj_list = graph.adj_list
        start_node_indices = graph.start_node_indices

        # Queue and visit status of nodes for BFS
        queue = Queue()
        vis = [False] * len(graph.nodes)

        for start_node_index in start_node_indices:
            if not vis[start_node_index]:
                vis[start_node_index] = True
                queue.put(start_node_index)

                # BFS
                while not queue.empty():
                    src_node_index = queue.get()

                    if src_node_index not in adj_list:
                        continue

                    for [edge_index, dest_node_index] in adj_list[src_node_index]:

                        if edge_index not in from_nodes:
                            from_nodes.update({edge_index : set()})
                        
                        from_nodes[edge_index].add(src_node_index + 1)

                        if edge_index not in to_nodes:
                            to_nodes.update({edge_index : set()})

                        to_nodes[edge_index].add(dest_node_index + 1)

                        if not vis[dest_node_index]:
                            vis[dest_node_index] = True
                            queue.put(dest_node_index)

        # Edge attributes that are not pushed to the database
        NOT_STORED_ATTR = ['label', 'value'] 

        try:
            # Surrogate Id for tensors
            tensor_id = 0

            # Number of mutations per row is the number of attributes being 
            # pushed to database.
            # 4 additional attributes, 'model_name', 'tensor_id', 
            # 'from_operator_ids' and 'to_operator_ids', present in db other 
            # than the class attributes.
            num_attributes = len(vars(graph.edges[0])) + 4

            # Number of nodes to be processed per batch i.e.
            # floor(max mutations per batch / number of mutations per row)
            num_edges_per_batch = self._MAX_MUTATIONS // num_attributes
            
            edge_indices = list(to_nodes.keys())
            num_edges = len(edge_indices)

            # TO-DO : Add retry logic if a batch fails
            while tensor_id < num_edges:
                with self.database.batch() as batch:
                    for _ in range(num_edges_per_batch):
                        if tensor_id == num_edges:
                            break

                        edge_index = edge_indices[tensor_id]
                        edge = graph.edges[edge_index]
                        to_operator_ids = list(to_nodes[edge_index])
                        from_operator_ids = list(from_nodes[edge_index])

                        # To store the database column names and their 
                        # values to be inserted
                        column_names = [
                            'model_name', 'tensor_id', 
                            'from_operator_ids', 'to_operator_ids'
                            ]
                        values = [
                            graph.model_name, tensor_id + 1, 
                            from_operator_ids, to_operator_ids
                            ]

                        attrs = vars(edge)
                        for item in attrs.items():
                            if item[0] not in NOT_STORED_ATTR:
                                column_names.append(item[0])
                                values.append(item[1])

                        column_names = tuple(column_names)
                        values = tuple(values)

                        batch.insert(
                            table = 'Tensors',
                            columns = column_names,
                            values = [values]
                        )
                        tensor_id += 1
            return True
        except Exception as e:
            print(e)
            query = "DELETE FROM Models WHERE model_name = \'" + graph.model_name + "\'"
            deleted_rows = self.database.execute_partitioned_dml(
                query
            )
            return False

    def load_data(self, graph, is_canonical):
        """Method to commit data into spanner database

        Stores data for given graph into three tables using 3 helper internal
        methods. Prints a log if data load fails.

        Args:
            graph (Graph object) : The intance of Graph to be stored in the Database.
            is_canonical (str) : String to separate unique architectures 
                from duplicates, The first model to be inserted into database
                with a specific architecture will have this to be "True", the other
                models with same architecture will have this to be "False". 
        """ 

        loaded = self._load_model(graph, is_canonical)
        if not loaded:
            print('Data Loading Failed in _load_model')
            return 
        
        loaded = self._load_operators(graph)
        if not loaded:
            print('Data Loading Failed in _load_operators')
            return 
        
        loaded = self._load_tensors(graph)
        if not loaded:
            print('Data Loading Failed in _load_tensors')
            return 

        print("Model", graph.model_name, "succesfuly loaded into database")

    def load_embeddings(self, model_graphs, embeddings):
        """Function to load model graph embeddings to Embeddings table

        Args:
            model_graphs (list of Graph objects) : The model graphs for which 
                the embeddings are to be loaded.
            embeddings (list of vectors) : The embeddings for model graphs with 
                index correspondence.
        """

        if len(embeddings) == 0:
            return
        
        for index, model_graph in enumerate(model_graphs):
            try:
                query = (
                    "UPDATE Models SET embeddings = " 
                    + str(list(embeddings[index])) + " WHERE model_name = '" 
                    + model_graph.model_name + "'"
                    )
                updated_rows = self.database.execute_partitioned_dml(
                    query
                )

            except Exception as e:
                print(e)
                print("Model", model_graph.model_name, "embeddings not loaded.")

    def parse_models(self, parse_stateful = False):
        """ Function to query and read data from database

        Function to query database and read models into Graph objects.

        Args:
            parse_stateful (bool) : Boolean to indicate whether graphs with 
                stateful partitioned call should be parsed, these graphs do not
                contain a graph structure or tensors. Defaults to False.

        Returns:
            List of Graph objects corresponding to the graph objects the models
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

            if row[4] == 0 and not parse_stateful:
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
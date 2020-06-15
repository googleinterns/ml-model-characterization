import os
from queue import Queue
from google.cloud import spanner

# Module to store Graph data into database
# Node object attributes are stored in 'Operators' table
# Edge object attributes are stored in 'Tensors' table
# Graph object attributes are stored in 'Models' table

# The attribute names in each case must correspond to a column name in 
# the database. If any attribute is added in the Class files, 
# the Storage class will automatically try to add that attribute to the 
# database the next time it is run with any model

# If an attribute is added to the class file and not to the database,
# this will throw and error unless explicitly changed

# Spanner allows only 20000 mutations per commit, so _load_operators
# and _load_tensors implement batching
class Storage:

    # Initialize spanner client with instance and database
    def __init__(self, instance_id, database_id):
        self.spanner_client = spanner.Client()
        self.instance = self.spanner_client.instance(instance_id)
        self.database = self.instance.database(database_id)

    # Internal function to load graph metadata into the Models table
    def _load_model(self, graph):
        # To store the database column names and their values to be inserted
        columns = list()
        values = list()

        # Adding Graph attributes and their values to columns and values
        attrs = vars(graph)
        for item in attrs.items():
            if (item[0] != 'nodes' and item[0] != 'start_node_indices' 
                and item[0] != 'edges' and item[0] != 'adj_list'):
                columns.append(item[0])
                values.append(item[1])

        columns = tuple(columns)
        values = tuple(values)

        # Inserting into database
        with self.database.batch() as batch:
            batch.insert(
                table = 'Models',
                columns = columns,
                values = [values]
            )

            

    # Internal function to load operator(Node) data into Operators table     
    def _load_operators(self, graph):
        # Surrogate Id for operators and iterating index
        operator_id = 0

        if len(graph.nodes) == 0:
            return

        # Number of nodes to be processed per batch 
        num_attributes = len(vars(graph.nodes[0]))
        num_nodes_per_batch = 20000 // num_attributes

        num_nodes = len(graph.nodes)

        # Inserting data into the Operators table

        # Loop till all nodes are inserted
        while operator_id < num_nodes:
            # Creating and inserting a batch
            with self.database.batch() as batch:
                for _ in range(num_nodes_per_batch):
                    if operator_id == num_nodes:
                        break

                    node = graph.nodes[operator_id]

                    # To store the database column names and their values 
                    # to be inserted
                    columns = ['model_name', 'operator_id']
                    values = [graph.model_name, operator_id + 1]

                    # Adding Node attributes and their values to 
                    # columns and values
                    attrs = vars(node)
                    for item in attrs.items():
                        if item[0] != 'label' and item[0] != 'value':
                            columns.append(item[0])
                            values.append(item[1])

                    columns = tuple(columns)
                    values = tuple(values)

                    # Insert
                    batch.insert(
                        table = 'Operators',
                        columns = columns,
                        values=  [values]
                    )
                    operator_id += 1

    # Internal function to load tensor(Edge) data into Tensors table
    def _load_tensors(self, graph):

        if len(graph.edges) == 0:
            return

        # Dictionary to store source nodes and destination nodes of edges
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

        # Surrogate Id for tensors
        tensor_id = 0

        # Number of edges to be processed per batch
        num_attributes = len(vars(graph.edges[0]))
        num_edges_per_batch = 20000 // num_attributes
        
        edge_indices = list(to_nodes.keys())
        num_edges = len(edge_indices)

        # Inserting data into the Tensors table
        # Looping till all edges are inserted
        while tensor_id < num_edges:
            # Creating and inserting a batch
            with self.database.batch() as batch:
                for _ in range(num_edges_per_batch):
                    if tensor_id == num_edges:
                        return

                    edge_index = edge_indices[tensor_id]
                    edge = graph.edges[edge_index]
                    to_operator_ids = list(to_nodes[edge_index])
                    from_operator_ids = list(from_nodes[edge_index])

                    # To store the database column names and their 
                    # values to be inserted
                    columns = [
                        'model_name', 'tensor_id', 
                        'from_operator_ids', 'to_operator_ids'
                        ]
                    values = [
                        graph.model_name, tensor_id + 1, 
                        from_operator_ids, to_operator_ids
                        ]

                    # Adding Edge attributes and their values to columns and values
                    attrs = vars(edge)
                    for item in attrs.items():
                        if item[0] != 'label' and item[0] != 'value':
                            columns.append(item[0])
                            values.append(item[1])

                    columns = tuple(columns)
                    values = tuple(values)

                    # Insert 
                    batch.insert(
                        table = 'Tensors',
                        columns = columns,
                        values = [values]
                    )
                    tensor_id += 1

    # Function to load data for given Graph to Spanner Database
    def load_data(self, graph):
        self._load_model(graph)
        self._load_operators(graph)
        self._load_tensors(graph)
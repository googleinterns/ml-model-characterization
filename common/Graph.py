"""Module with Graph class to store models

Graph class stores model graph structure and metadata,
also contains helper functions to print graph, nodes and edges.

"""

from queue import Queue

class Graph:
    """ Graph class to store model graph and metadata

        Stores model graphs as an adjacency list, operators as nodes
        and tensors as edges. Also stores metadata related to the model

        Attributes:
            nodes (list[Node objects]) : The nodes used in the graph
            start_node_indices (list of int) : Indexes referencing to nodes which
                are the starting nodes of the graph
            edges (list[Edge objects]) : The edges used in the graph 
            adj_list (dict{int : list of [int, int]}) : The adjacency list 
                of the graph which is a dictionary storing the edges as follows,
                {from_node_index : list[[edge_index, to_node_index]]}.
                All indexes are referenced to nodes.
            source (str, optional) : The file format the model is read from,
                currently only "TF" and "TFLite" are supported.
            category (str) : Problem category model falls under
            model_name (str) : Name of the model.
            num_inputs (int) : Number of inputs to the model.
            num_outputs (int) : Number of outputs from the model.
            max_fan_in (int) : Maximum number of incoming edges to any node.
            max_fan_out (int) : Maximum number of outgoing edges from any node.
    """

    def __init__(self, nodes, start_node_indices, edges, adj_list, model_name,
                    category):
        self.nodes = nodes
        self.start_node_indices = start_node_indices
        self.edges = edges
        self.adj_list = adj_list
        self.source = None
        self.category = category
        self.model_name = model_name
        self.num_inputs = self.calc_num_inputs()
        self.num_outputs = self.calc_num_outputs()
        self.max_fan_in = self.calc_max_fan_in()
        self.max_fan_out = self.calc_max_fan_out()

    def calc_num_inputs(self):
        """ Method to calculate num_inputs

        Iterates through nodes and counts the number of nodes labelled
        'Input_Placeholder'

        Returns:
            An integer corresponding to the number of inputs to the model
        """

        ret = 0

        for node in self.nodes:
            if node.label == 'Input_Placeholder':
                ret += 1

        return ret

    def calc_num_outputs(self):
        """ Method to calculate num_outputs

        Iterates through nodes and counts the number of nodes labelled
        'Output_Placeholder'

        Returns:
            An integer corresponding to the number of outputs from the model
        """

        ret = 0

        for node in self.nodes:
            if node.label == 'Output_Placeholder':
                ret += 1

        return ret

    def calc_max_fan_out(self):
        """ Method to calculate max_fan_out

        Calculated the maximum number of outgoing edges from a node.

        Returns:
            An integer corresponding to max_fan_out
        """

        ret = 0

        # len(adj_list[node_index]) is out degree.
        for src_node_index in range(len(self.nodes)):
            if src_node_index in self.adj_list:
                ret = max(ret, len(self.adj_list[src_node_index]))

        return ret

    def calc_max_fan_in(self):
        """ Method to calculate max_fan_in

        Calculated the maximum number of incoming edges to a node.

        Returns:
            An integer corresponding to max_fan_in
        """

        in_degree = [0] * len(self.nodes)
        ret = 0

        # calculating indegree and updating ret value
        for src_node_index in range(len(self.nodes)):
            if src_node_index in self.adj_list:
                for (_, dest_node_index) in self.adj_list[src_node_index]:
                    in_degree[dest_node_index] += 1
                    ret = max(ret, in_degree[dest_node_index])
 
        return ret    

    def print_graph(self):
        """ Helper Method to print graph

        Prints the graph in BFS order using the node and edge labels.
        """

        # List to store visit status of a node
        visited = [False] * (len(self.nodes))
        queue = Queue()

        # For each start node, if not already visited, start a new traversal
        for node_index in self.start_node_indices:
            queue.put(node_index)
            visited[node_index] = True
            
            # BFS
            while not queue.empty():
                src_node_index = queue.get()

                if src_node_index not in self.adj_list:
                    continue

                for [edge_index, dest_node_index] in self.adj_list[src_node_index]:
                    print(
                        self.nodes[src_node_index].label, 
                        self.edges[edge_index].label, 
                        self.nodes[dest_node_index].label,
                    )

                    if not visited[dest_node_index]:
                        visited[dest_node_index] = True
                        queue.put(dest_node_index)

    def print_nodes(self):
        """ Helper Method to print nodes of a graph

        Prints the attributes of nodes of the graph in the format,
        attr1 : val1 attr2 : val2 ...
        Does not print attributes whos value is None or the string 'NONE'.
        """

        print("\nOperators\n")
        for node in self.nodes:
            attrs = vars(node)
            
            for item in attrs.items():
                if item[1] != None and item[1] != "NONE":
                    print(str(item[0]) + ": " + str(item[1]), end=' ')
            
            print()

    def print_edges(self):
        """ Helper Method to print edges of a graph

        Prints the attributes of edges of the graph in the format,
        attr1 : val1 attr2 : val2 ...
        Does not print attributes whos value is None or the string 'NONE'.
        """

        print("\nTensors\n")
        for edge in self.edges:
            attrs = vars(edge)
            
            for item in attrs.items():
                if item[1] != None and item[1] != "NONE":
                    print(str(item[0]) + ": " + str(item[1]), end=' ')
            
            print()

    def get_traversed_edges(self):
        """ Method to get edges which are come across during traversal

        Traverses the graph in BFS order from start_node_indices and keeps
        track of the edges visited.

        Returns:
            A list of Edge objects which are encountered during 
            traversal of graph.
        """
        
        visited = dict()
        queue = Queue()

        # set to maintain unique edges encountered
        traversed_edges = set()
        
        # List to store visit status of a node
        for node_index in range(len(self.nodes)):
            visited[node_index] = False

        # For each start node, if not already visited, start a new traversal
        for node_index in self.start_node_indices:
            queue.put(node_index)
            
            # BFS
            while not queue.empty():
                source_node_index = queue.get()

                if source_node_index not in self.adj_list:
                    continue

                for [edge_index, dest_node_index] in self.adj_list[source_node_index]:
                    traversed_edges.add(self.edges[edge_index])

                    if not visited[dest_node_index]:
                        visited[dest_node_index] = True
                        queue.put(dest_node_index)
        
        return list(traversed_edges)
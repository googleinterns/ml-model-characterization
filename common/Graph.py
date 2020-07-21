"""Module with Graph class to store models.

Graph class stores model graph structure and metadata,
also contains helper functions to print graph, nodes and edges.

"""

from queue import Queue

class Graph:
    """Graph class to store model graph and metadata.

    Stores model graphs as an adjacency list, operators as nodes
    and tensors as edges. Also stores metadata related to the model.

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
        category (str) : Problem category model falls under, i.e. 
            Text, Image etc.
        sub_category (str) : Problem sub category the model falls under i.e.
            ImageClassification, TextEmbedding etc.
        model_name (str) : Name of the model.
        num_inputs (int) : Number of inputs to the model.
        num_outputs (int) : Number of outputs from the model.
        max_fan_in (int) : Maximum number of incoming edges to any node.
        max_fan_out (int) : Maximum number of outgoing edges from any node.
    """

    def __init__(self, nodes, start_node_indices, edges, adj_list, model_name,
                    category, sub_category):
        self.nodes = nodes
        self.start_node_indices = start_node_indices
        self.edges = edges
        self.adj_list = adj_list
        self.source = None
        self.category = category
        self.sub_category = sub_category
        self.model_name = model_name
        self.num_inputs = self.calc_num_inputs()
        self.num_outputs = self.calc_num_outputs()
        self.max_fan_in = self.calc_max_fan_in()
        self.max_fan_out = self.calc_max_fan_out()

    def process_nodes(self):
        """Function to delete nodes not traversable from start.

        Function to traverse the graph from inputs and delete the ones
        that are not visited.
        Nodes with certain labels are discarded from output.
        Assumption is that nodes that are not reachable from input are
        not of semantic use.
        """

        NOT_OUTPUT = [
        "Assert", "Unpack", "Placeholder", "StridedSlice", "Less", 
        "StopGradient", "Exit", "ExpandDims", "Shape", "Merge",
        "ApplyAdam", "AssignSub", "BiasAddGrad", "Conv2DBackpropFilter"
        ] # List of operations which cannot be output nodes
        # Tentative list based on graph analysis of models.

        adj_list = self.adj_list
        nodes = self.nodes
        start_node_indices = self.start_node_indices

        # List to store visit status of node, 0 is visited and 1 otherwise
        visited = [1] * len(nodes)

        queue = Queue()

        # For each start node, if not already visited, start a new traversal
        for node_index in start_node_indices:
            queue.put(node_index)
            visited[node_index] = 0
            # BFS
            while not queue.empty():
                src_node_index = queue.get()

                # If no outgoing edges, assign label as output
                if src_node_index not in adj_list:
                    op_type = nodes[src_node_index].operator_type
                    if ("VariableOp" not in op_type 
                        and op_type not in NOT_OUTPUT):
                        # print(op_type, operation.name)
                        nodes[src_node_index].label = "Output_Placeholder"
                    continue
                
                for [edge_index, dest_node_index] in adj_list[src_node_index]:
                    if visited[dest_node_index] == 1:
                        visited[dest_node_index] = 0
                        queue.put(dest_node_index)

        # Calculate cumulative sum of visited and remove all unvisited nodes
        # After summing, value stored in current index of visited will tell 
        # how many nodes have been deleted with index <= current index
        new_nodes = list()
        for index in range(len(nodes)):
            if visited[index] == 0:
                new_nodes.append(nodes[index])
            if index != 0:
                visited[index] += visited[index - 1]

        nodes = new_nodes
        del new_nodes

        # Creating a new adjacency list since node indices have been updated
        # Using visited, new index of a node will be old index - visited[old index]
        new_adj_list = dict()
        for item in adj_list.items():
            src_node_index = item[0]

            if src_node_index == 0 and visited[src_node_index] == 1:
                continue

            if visited[src_node_index] - visited[src_node_index - 1] == 1:
                continue

            new_src_node_index = (src_node_index 
                                - visited[src_node_index])

            new_adj_list.update({new_src_node_index : []})
            for [edge_index, dest_node_index] in item[1]:
                
                if dest_node_index == 0 and visited[dest_node_index] == 1:
                    continue

                if visited[dest_node_index] - visited[dest_node_index - 1] == 1:
                    continue

                new_dest_node_index = (dest_node_index -
                                        visited[dest_node_index])
                
                new_adj_list[new_src_node_index].append(
                    [edge_index, new_dest_node_index])

        for index in range(len(start_node_indices)):
            start_node_indices[index] -= visited[start_node_indices[index]]

        adj_list = new_adj_list
        del new_adj_list

        # Updating nodes, adj_list and start_node_indices
        self.nodes = nodes
        self.adj_list = adj_list
        self.start_node_indices = start_node_indices

        # Recalculating graph attributes
        self.max_fan_in = self.calc_max_fan_in()
        self.max_fan_out = self.calc_max_fan_out()
        self.num_inputs = self.calc_num_inputs()
        self.num_outputs = self.calc_num_outputs()

    def calc_num_inputs(self):
        """Method to calculate num_inputs.

        Iterates through nodes and counts the number of nodes labelled
        'Input_Placeholder'.

        Returns:
            An integer corresponding to the number of inputs to the model.
        """

        ret = 0

        for node in self.nodes:
            if node.label == 'Input_Placeholder':
                ret += 1

        return ret

    def calc_num_outputs(self):
        """Method to calculate num_outputs.

        Iterates through nodes and counts the number of nodes labelled
        'Output_Placeholder'.

        Returns:
            An integer corresponding to the number of outputs from the model.
        """

        ret = 0

        for node in self.nodes:
            if node.label == 'Output_Placeholder':
                ret += 1

        return ret

    def calc_max_fan_out(self):
        """Method to calculate max_fan_out.

        Calculated the maximum number of outgoing edges from a node.

        Returns:
            An integer corresponding to max_fan_out.
        """

        ret = 0

        # len(adj_list[node_index]) is out degree.
        for src_node_index in range(len(self.nodes)):
            if src_node_index in self.adj_list:
                ret = max(ret, len(self.adj_list[src_node_index]))

        return ret

    def calc_max_fan_in(self):
        """Method to calculate max_fan_in.

        Calculated the maximum number of incoming edges to a node.

        Returns:
            An integer corresponding to max_fan_in.
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
        """Helper Method to print graph.

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
        """Helper Method to print nodes of a graph.

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
        """Helper Method to print edges of a graph.

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
        """Method to get edges which are come across during traversal.

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
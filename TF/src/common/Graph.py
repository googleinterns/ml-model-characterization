
from queue import Queue

# Graph Module to store adjacency list and graph related attributed
# Nodes will be a list of operators, edges a list of tensors
# start_nodes is a list of indices referencing nodes and edges, denoting entry nodes
# adj_list is a dictionary of the form {src node : list of ([edge, dest_node])} 
# where src_node, edge and dest_node and indices referencing nodes and edges
class Graph:
    def __init__(self, nodes, start_node_indices, edges, adj_list, model_name, category):
        self.nodes = nodes
        self.start_node_indices = start_node_indices
        self.edges = edges
        self.adj_list = adj_list
        self.category = category
        self.model_name = model_name
        self.num_inputs = len(self.start_node_indices)
        self.num_outputs = self.calc_num_outputs()
        self.max_fan_in = self.calc_max_fan_in()
        self.max_fan_out = self.calc_max_fan_out()

    def calc_num_outputs(self):
        ret = 0

        # Number of proxy nodes for output
        for node in self.nodes:
            if node.label == 'Output_Placeholder':
                ret += 1

        return ret

    def calc_max_fan_out(self):
        ret = 0

        for src_node_index in range(len(self.nodes)):
            if src_node_index in self.adj_list:
                ret = max(ret, len(self.adj_list[src_node_index]))

        return ret

    def calc_max_fan_in(self):
        in_degree = [0] * len(self.nodes)
        ret = 0

        for src_node_index in range(len(self.nodes)):
            if src_node_index in self.adj_list:
                for (_, dest_node_index) in self.adj_list[src_node_index]:
                    in_degree[dest_node_index] += 1
                    ret = max(ret, in_degree[dest_node_index])
 
        return ret    

    # Print graph in BFS order
    def print_graph(self):
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

    # Print nodes with attributes
    def print_nodes(self):
        print("\nOperators\n")
        for node in self.nodes:
            attrs = vars(node)
            
            for item in attrs.items():
                if item[1] != None and item[1] != "NONE":
                    print(str(item[0]) + ": " + str(item[1]), end=' ')
            
            print()

    # Print edges with attributes
    def print_edges(self):
        print("\nTensors\n")
        for edge in self.edges:
            attrs = vars(edge)
            
            for item in attrs.items():
                if item[1] != None and item[1] != "NONE":
                    print(str(item[0]) + ": " + str(item[1]), end=' ')
            
            print()

    # Function to get those edges which are traversed during BFS
    # Assumption is that the tensors not reached by traversal are weights, bias etc.
    def get_traversed_edges(self):
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
# Graph Module to store adjacency list and graph related attributed
# Nodes will be a list of operators, edges a list of tensors
# adj_list, start_nodes will be in the form of indices referencing nodes and edges

from queue import Queue

class Graph:
    def __init__(self, nodes, start_node_indices, edges, adj_list):
        self.nodes = nodes
        self.start_node_indices = start_node_indices
        self.edges = edges
        self.adj_list = adj_list
        self.num_inputs = len(self.start_node_indices)
        self.num_outputs = self.calc_num_outputs()
        self.max_fan_out = self.calc_max_fanout()
        self.max_fan_in = self.calc_max_fanin()

    def calc_num_outputs(self):
        ret = 0

        for node in self.nodes:
            if node.label == 'Output_Placeholder':
                ret += 1

        return ret

    def calc_max_fanout(self):
        ret = 0

        for src_node_index in range(len(self.nodes)):
            if src_node_index in self.adj_list:
                ret = max(ret, len(self.adj_list[src_node_index]))

        return ret

    def calc_max_fanin(self):
        in_count = [0] * len(self.nodes)
        ret = 0

        for src_node_index in range(len(self.nodes)):
            if src_node_index in self.adj_list:
                for (_, dest_node_index) in self.adj_list[src_node_index]:
                    in_count[dest_node_index] += 1
                    ret = max(ret, in_count[dest_node_index])
 
        return ret    

    # Print graph in BFS order
    def print_graph(self):
        visited = dict()
        queue = Queue()
        
        for node_index in range(len(self.nodes)):
            visited[node_index] = False

        for node_index in self.start_node_indices:
            print('\nStart of new subgraph\n')
            queue.put(node_index)
            
            while not queue.empty():
                source_node_index = queue.get()

                if source_node_index not in self.adj_list:
                    continue

                for [edge_index, dest_node_index] in self.adj_list[source_node_index]:
                    print(
                        self.nodes[source_node_index].label, 
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


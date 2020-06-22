import tensorflow as tf
import tensorflow_text
from common import Graph
from common import Node
import OpToNode
import TensorToEdge
from queue import Queue

# Module wiith methods to parse a .pb file (Frozen Graph) and
# return a Graph object with populated node and edge attributes
class TFParser:

    # Class instances to convert ops to nodes and tensors to edges
    _OP_TO_NODE = OpToNode.OpToNode()
    _TENSOR_TO_EDGE = TensorToEdge.TensorToEdge()

    # Operations that cannot be output nodes and will not be marked
    # as Ouptut_Placeholder even if no outgoing edges are present
    _NOT_OUTPUT = [
        "Assert", "Unpack", "Placeholder", "StridedSlice", "Less", 
        "StopGradient", "Mean", "Exit", "ExpandDims", "Shape", "Merge",
        ]

    def __init__(self):
        pass

    def parse_graph(self, file_path, model_name, category, is_saved_model, input_operation_names):
        if is_saved_model == "True":
            saved_model = tf.core.protobuf.saved_model_pb2.SavedModel()
            with tf.io.gfile.GFile(file_path, "rb") as f:
                saved_model.ParseFromString(f.read())

            meta_graph = saved_model.meta_graphs[0]
            graph_def = meta_graph.graph_def

            tf.io.write_graph(graph_def, "/home/shobhitbehl/GraphDef", model_name + ".pb")

        else:
            with tf.io.gfile.GFile(file_path, "rb") as f:
                graph_def = tf.compat.v1.GraphDef()
                graph_def.ParseFromString(f.read())
        
        with tf.Graph().as_default() as graph:
            tf.import_graph_def(graph_def, name="")

            # Dictionary to store origin and destination nodes for each edge
            # Nodes are operations, edges are tensors
            to_nodes = dict()
            from_nodes = dict()

            # Lists to store Nodes and Edges
            edges = list()
            nodes = list()
            start_node_indices = list()

            # tensor and operation to index mapping
            tensor_to_index = dict()

            # Loop to populate to_nodes and from_nodes
            for operation in graph.get_operations():

                # Leaving out const operations
                if operation.node_def.op == "Const":
                    continue

                # Converting operation to nodes
                new_node = self._OP_TO_NODE.convert(operation)
                node_index = len(nodes)
                nodes.append(new_node)

                # Add input_operation_names to start_node_indices
                if operation.name in input_operation_names:
                    start_node_indices.append(node_index)

                # Input node, also the start node to the graph
                if operation.node_def.op == "Placeholder":
                    new_node.label = "Input_Placeholder"
                    start_node_indices.append(node_index)

                # Input and output edges to the node, 
                # populating from_nodes and to_nodes
                for in_tensor in list(operation.inputs):
                    if in_tensor not in tensor_to_index:
                        tensor_to_index[in_tensor] = len(edges)
                        new_edge = self._TENSOR_TO_EDGE.convert(in_tensor)
                        edges.append(new_edge)
                    
                    edge_index = tensor_to_index[in_tensor]
                    if edge_index not in to_nodes:
                        to_nodes.update({edge_index : []})

                    to_nodes[edge_index].append(node_index)
                
                for out_tensor in list(operation.outputs):
                    if out_tensor not in tensor_to_index :
                        tensor_to_index[out_tensor] = len(edges)
                        new_edge = self._TENSOR_TO_EDGE.convert(out_tensor)
                        edges.append(new_edge)

                    edge_index = tensor_to_index[out_tensor]
                    if edge_index not in from_nodes:
                        from_nodes.update({edge_index : []})
                    
                    from_nodes[edge_index].append(node_index)

            # Creating and adjacency list using from_nodes and to_nodes
            adj_list = dict()
            for edge_index in range(len(edges)):
                if edge_index not in from_nodes or edge_index not in to_nodes:
                    continue

                for node1_index in from_nodes[edge_index]:
                    for node2_index in to_nodes[edge_index]:
                        if node1_index not in adj_list:
                            adj_list.update({node1_index : list()})

                        adj_list[node1_index].append([edge_index, node2_index])

            # List of nodes contains nodes that are never visited by a traversal.
            # Assuming these are weights and biases,
            # Following code discards them
            # Also calculated output nodes

            # List to store visit status of node, 
            # but 0 means visited and 1 means unvisited
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
                            and op_type not in self._NOT_OUTPUT):
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

            # new_nodes to store only nodes reached by traversal

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

            graph = Graph.Graph(nodes, start_node_indices, edges, adj_list,
                                    model_name, category)
            graph.source = "TF"
            return graph


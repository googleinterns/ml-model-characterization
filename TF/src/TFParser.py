"""Module with TFParser class to parse TF files"""

from queue import Queue
import tensorflow as tf
import tensorflow_text

from common import Graph
from common import Node
import OpToNode
import TensorToEdge

class TFParser:
    """Class to parse TF files

    Contains parsing for SavedModel and FrozenGraph formats.
    """

    _OP_TO_NODE = OpToNode.OpToNode() # For converting operations to nodes
    _TENSOR_TO_EDGE = TensorToEdge.TensorToEdge() # For converting tensors to edges

    _NOT_OUTPUT = [
        "Assert", "Unpack", "Placeholder", "StridedSlice", "Less", 
        "StopGradient", "Exit", "ExpandDims", "Shape", "Merge",
        "ApplyAdam", "AssignSub", "BiasAddGrad", "Conv2DBackpropFilter"
        ] # List of operations which cannot be output nodes
        # Tentative list based on graph analysis of models.

    def __init__(self):
        pass

    def parse_graph(self, file_path, model_name, category, sub_category,
                        is_saved_model, input_operation_names):
        """Method to parse file and Create a corresponding Graph object

        Reads a GraphDef from SavedModel or FrozenGraph file and extracts 
        operations, tensors, graph structure and metadata and stores it 
        into a Graph, Node and Edge objects. Nodes are operations 
        and edges are tensors.

        If graph contains a 'StatefulPartitionedCall' operation,
        all operations are extracted and pushed into the database without tensor
        information or graph structure.

        Args:
            file_path (str): path of the file to parse
            model_name (str): unique model name of the model being parsed.
            category (str): problem category of the model.
            sub_category (str) : problem sub category of the model.
            is_saved_model (str, optional): "True" if file is in SavedModel format, 
                defaults to "True".
            input_operation_names (list of str, optional) : Names of the operations 
                that are inputs to the model, defaults to [].

        Returns:
            The Graph object created for the file.
        """
        
        if is_saved_model == "True":
            saved_model = tf.core.protobuf.saved_model_pb2.SavedModel()
            with tf.io.gfile.GFile(file_path, "rb") as f:
                saved_model.ParseFromString(f.read())

            meta_graph = saved_model.meta_graphs[0]
            graph_def = meta_graph.graph_def

        else:
            with tf.io.gfile.GFile(file_path, "rb") as f:
                graph_def = tf.compat.v1.GraphDef()
                graph_def.ParseFromString(f.read())
        
        with tf.Graph().as_default() as graph:
            tf.import_graph_def(graph_def, name = "")

            # Dictionary to store origin and destination nodes for each edge
            to_nodes = dict()
            from_nodes = dict()

            edges = list()
            nodes = list()
            start_node_indices = list()

            tensor_to_index = dict()

            # Loop to populate to_nodes and from_nodes
            for operation in graph.get_operations():

                # If graph contains StatefulPartitionedCall operation,
                # only extracting the operations and returning empty graph
                if operation.node_def.op == "StatefulPartitionedCall":
                    print(
                        "Graphs with operation 'StatefulPartitionedCall' are " 
                        "not fully supported for parsing, graph or tensor " 
                        "information not supported, only operators will be "
                        "loaded into database."
                        )

                    NODES_DISCARDED = [
                        "Const", "VarHandleOp", "StatefulPartitionedCall", 
                        "NoOp", "Identity"
                        ] # List of operations to not be considered, not of semantic use.

                    nodes.clear()

                    # Looping over all ops in the graph
                    for node_def in graph_def.node:
                        op = node_def.op
                        if op in NODES_DISCARDED or "VariableOp" in op:
                            continue
                        new_node = self._OP_TO_NODE.convert(None, node_def)
                        nodes.append(new_node)

                    # Looping over operations that occur within functions
                    for func in graph_def.library.function:
                        for node_def in func.node_def:
                            op = node_def.op
                            if op in NODES_DISCARDED or "VariableOp" in op:
                                continue

                            new_node = self._OP_TO_NODE.convert(None, node_def)
                            nodes.append(new_node)

                    # Discarding unwanted nodes
                    for index, node in enumerate(nodes):
                        if (node.operator_type in NODES_DISCARDED or 
                            "VariableOp" in node.operator_type):
                            nodes.pop(index)
                            
                    new_graph = Graph.Graph(nodes, [], [], {}, model_name, category, sub_category)
                    new_graph.source = "TF"

                    return new_graph

                if operation.node_def.op == "Const":
                    continue

                # Converting operation to nodes
                new_node = self._OP_TO_NODE.convert(operation, operation.node_def)
                node_index = len(nodes)
                nodes.append(new_node)

                # Add input_operation_names to start_node_indices
                if operation.name in input_operation_names:
                    start_node_indices.append(node_index)

                # Input node, also the start node to the graph
                if operation.node_def.op == "Placeholder":
                    new_node.label = "Input_Placeholder"
                    start_node_indices.append(node_index)

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
            # following code discards them and also calculates output nodes

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
                                    model_name, category, sub_category)
            graph.source = "TF"
            return graph
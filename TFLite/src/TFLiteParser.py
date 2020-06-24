"""Module with TFLiteParser class to parse tflite files"""

import flatbuffers
import inspect


import OpToNode
import TensorToEdge
from common import Edge
from common import Graph
from common import Node
from tflite import BuiltinOperator
from tflite import Model

class TFLiteParser:
    """Class to parse TF files

    Contains parsing for SavedModel and FrozenGraph formats.
    """

    _OP_TO_NODE = OpToNode.OpToNode() # For converting operations to nodes
    _TENSOR_TO_EDGE = TensorToEdge.TensorToEdge() # For converting tensors to edges

    def __init__(self):

        self._builtin_optype = dict() # Dict for enum value to name mapping
        for member in inspect.getmembers(
                BuiltinOperator.BuiltinOperator):
            if not member[0].startswith('_'):
                if not inspect.ismethod(member[1]):
                    self._builtin_optype[member[1]] = member[0]

    # Reading tflite file onto Model object
    def parse(self, file_path):
        """Method to parse tflite file into a tflite Model object

        Args:
            file_path (str) : path to the file to be parsed.

        Returns:
            tflite Model object containing the tflite model information.
        """
        
        with open(file_path, "rb") as file:
            model = Model.Model.GetRootAsModel(file.read(), 0)
        
        return model

    def parse_graph(self, file_path, model_name, category):
        """Method to parse file and Create a corresponding Graph object

        Reads a tflite file into a tflite/Model Object and then extracts 
        operators, tensors, graph structure and metadata and stores it 
        into a Graph, Node and Edge objects. Nodes are operations and 
        edges are tensors.

        Args:
            file_path (str): path of the file to parse
            model_name (str): unique model name of the model being parsed.
            category (str): problem category of the model.

        Returns:
            The Graph object created for the file.
        """

        model = self.parse(file_path)

        nodes = list()
        edges = list()
        adj_list = dict()
        start_node_indices = list()

        # Global list of opcodes in the model, referenced by Operators
        opcodes = list()
        for opcode_index in range(model.OperatorCodesLength()):
            opcodes.append(model.OperatorCodes(opcode_index))

        # Only considering the main model
        subgraph = model.Subgraphs(0)

        # Dictionary to store origin and destination nodes for each edge
        to_nodes = dict()
        from_nodes = dict()

        for tensor_index in range(subgraph.TensorsLength()):
            tensor = subgraph.Tensors(tensor_index)
            # Converting tensor to an Edge object
            new_edge = self._TENSOR_TO_EDGE.convert(tensor)
            edges.append(new_edge)
        
        # Populating to_nodes, from_nodes
        # Add proxy nodes for Input and Output of the model

        for input_index in range(subgraph.InputsLength()):
            new_node = Node.Node(label = "Input_Placeholder", operator_type = "Input_Placeholder")
            nodes.append(new_node)
            
            node_index = len(nodes) - 1
            start_node_indices.append(node_index)
            edge_index = subgraph.Inputs(input_index)
            
            if edge_index not in from_nodes:
                from_nodes.update({edge_index : []})
            from_nodes[edge_index].append(node_index)

        for operator_index in range(subgraph.OperatorsLength()):
            operator = subgraph.Operators(operator_index)
            builtin_opcode = opcodes[operator.OpcodeIndex()].BuiltinCode()
            opname =  self._builtin_optype[builtin_opcode]

            new_node = self._OP_TO_NODE.convert(operator, opname)

            # Condition to extract Conv 2D filter sizes and
            # input and output channels as it is contained in tensors 
            # and not in operators
            if new_node.label == "CONV_2D":
                weight_tensor = subgraph.Tensors(operator.Inputs(1))
                new_node.filter_height = weight_tensor.Shape(1)
                new_node.filter_width = weight_tensor.Shape(2)

            nodes.append(new_node)
            node_index = len(nodes) - 1

            
            for input_index in range(operator.InputsLength()):
                edge_index = operator.Inputs(input_index)
                if edge_index not in to_nodes:
                    to_nodes.update({edge_index : list()})

                to_nodes[edge_index].append(node_index)
            
            for output_index in range(operator.OutputsLength()):
                edge_index = operator.Outputs(output_index)
                if edge_index not in from_nodes:
                    from_nodes.update({edge_index : list()})

                from_nodes[edge_index].append(node_index)

        for output_index in range(subgraph.OutputsLength()):
            new_node = Node.Node(label = "Output_Placeholder", operator_type = "Output_Placeholder")
            nodes.append(new_node)
            
            node_index = len(nodes) - 1
            edge_index = subgraph.Outputs(output_index)
            
            if edge_index not in to_nodes:
                to_nodes.update({edge_index : []})
            to_nodes[edge_index].append(node_index)

        # Constructing adjacency List from to_nodes, from_nodes
        for edge_index in range(len(edges)):

            if edge_index not in from_nodes or edge_index not in to_nodes:
                continue

            for node1_index in from_nodes[edge_index]:
                for node2_index in to_nodes[edge_index]:
                    if node1_index not in adj_list:
                        adj_list.update({node1_index : list()})

                    adj_list[node1_index].append([edge_index, node2_index])
            
        graph = Graph.Graph(nodes, start_node_indices, edges, adj_list, model_name, category)
        graph.source = "TFLite"
        return graph             
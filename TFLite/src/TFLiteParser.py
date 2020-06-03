import flatbuffers
import inspect
import tflite.BuiltinOperator
import tflite.Model
import Node
import Edge
import Graph
import OpToNode
import TensorToEdge


class TFLiteParser:

    _OP_TO_NODE = OpToNode.OpToNode()
    _TENSOR_TO_EDGE = TensorToEdge.TensorToEdge()

    def __init__(self):

        # Dictionary for enum value to enum name mapping
        self._builtin_optype = dict()
        for member in inspect.getmembers(
                tflite.BuiltinOperator.BuiltinOperator):
            if not member[0].startswith('_'):
                if not inspect.ismethod(member[1]):
                    self._builtin_optype[member[1]] = member[0]

    # Reading tflite file onto Model object
    def parse(self, file_path):
        with open(file_path, "rb") as file:
            model = tflite.Model.Model.GetRootAsModel(file.read(), 0)
        
        return model

    # Generate Graph from Model with tensors as edges and operators as nodes
    # Adjacency List is integers indirected to global list of edges and nodes
    def parse_graph(self, file_path):
        model = self.parse(file_path)

        nodes = list()
        edges = list()
        adj_list = dict()
        start_node_indices = list()

        # Global list of opcodes in the model, referenced by Operators
        opcodes = list()
        for opcode_index in range(model.OperatorCodesLength()):
            opcodes.append(model.OperatorCodes(opcode_index))

        for subgraph_index in range(model.SubgraphsLength()):
            subgraph = model.Subgraphs(subgraph_index)

            # Dictionary to store origin and destination nodes for each edge
            to_nodes = dict()
            from_nodes = dict()
            
            # Adding tensors to list of edges
            offset_edges = len(edges) 

            for tensor_index in range(subgraph.TensorsLength()):
                tensor = subgraph.Tensors(tensor_index)
                new_edge = Edge.Edge(label = tensor.Name(), value = tensor)
                new_edge = self._TENSOR_TO_EDGE.convert(tensor)
                edges.append(new_edge)
            
            # Populating to_nodes, from_nodes
            # Add proxy nodes for Input and Output of the model
            for input_index in range(subgraph.InputsLength()):
                new_node = Node.Node(label = "Input_Placeholder")
                nodes.append(new_node)
                
                node_index = len(nodes) - 1
                start_node_indices.append(node_index)
                edge_index = subgraph.Inputs(input_index) + offset_edges
                
                if edge_index not in from_nodes:
                    from_nodes.update({edge_index : []})
                from_nodes[edge_index].append(node_index)

            for operator_index in range(subgraph.OperatorsLength()):
                operator = subgraph.Operators(operator_index)
                builtin_opcode = opcodes[operator.OpcodeIndex()].BuiltinCode()
                opname =  self._builtin_optype[builtin_opcode]

                new_node = self._OP_TO_NODE.convert(operator, opname)

                nodes.append(new_node)
                node_index = len(nodes) - 1

                
                for input_index in range(operator.InputsLength()):
                    edge_index = operator.Inputs(input_index) + offset_edges
                    if edge_index not in to_nodes:
                        to_nodes.update({edge_index : list()})

                    to_nodes[edge_index].append(node_index)
                
                for output_index in range(operator.OutputsLength()):
                    edge_index = operator.Outputs(output_index) + offset_edges
                    if edge_index not in from_nodes:
                        from_nodes.update({edge_index : list()})

                    from_nodes[edge_index].append(node_index)

            for output_index in range(subgraph.OutputsLength()):
                new_node = Node.Node(label = "Output_Placeholder")
                nodes.append(new_node)
                
                node_index = len(nodes) - 1
                edge_index = subgraph.Outputs(output_index) + offset_edges
                
                if edge_index not in to_nodes:
                    to_nodes.update({edge_index : []})
                to_nodes[edge_index].append(node_index)

            # Constructing Adjacency List from to_nodes, from_nodes
            for edge_index in range(offset_edges, len(edges)):

                if edge_index not in from_nodes or edge_index not in to_nodes:
                    continue

                for node1_index in from_nodes[edge_index]:
                    for node2_index in to_nodes[edge_index]:
                        if node1_index not in adj_list:
                            adj_list.update({node1_index : list()})

                        adj_list[node1_index].append([edge_index, node2_index])
            
        graph = Graph.Graph(nodes, start_node_indices, edges, adj_list)
        return graph             






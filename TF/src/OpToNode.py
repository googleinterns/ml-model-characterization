"""Module with OpToNode class to create Node objects for operations"""

import tensorflow as tf

from common import Node

class OpToNode:
    """Class to convert operation to Node object"""

    def __init__(self):
        pass
    
    # Internal helper function to extract attributes from different op types.
    def _attr_from_node_def(self, operation, node_def, node):
        """Internal function to add attribute values to node

        Adds relevant attribute values to node depending on the operation.
        For semantic information on Node attributes for TF,
        https://www.tensorflow.org/api_docs/python/tf/raw_ops

        Args:
            operation (GraphDef operation object) : The operation that the 
                node represents, is None in case of constructing nodes 
                from the node_def of a function.
            node_def (NodeDef object) : The node_def of the operation/function.
            node (Node object) : The Node object to add attribute values to.

        Returns:
            The modified Node object with added attribute values.
        """

        attr = node_def.attr
        op = node_def.op

        if op == "Conv2D" or op == "DepthwiseConv2dNative":
            node.padding = attr['padding'].s.decode('utf-8')
            node.data_format = attr['data_format'].s.decode('utf-8')

            # If empty string, assume default value
            if node.data_format == '':
                node.data_format = "NHWC"

            for index in range(len(node.data_format)):
                if node.data_format[index] == 'H':
                    h_index = index
                if node.data_format[index] == 'W':
                    w_index = index
            
            node.stride_h = list(attr['strides'].list.i)[h_index]
            node.stride_w = list(attr['strides'].list.i)[w_index]

            # If dilations are empty list, assume default value
            if list(attr['dilations'].list.i) == []:
                node.dilation_h_factor = 1
                node.dilation_w_factor = 1
            else:
                node.dilation_w_factor = list(attr['dilations'].list.i)[w_index]
                node.dilation_h_factor = list(attr['dilations'].list.i)[h_index]

            if operation != None:
                node.filter_height = operation.inputs[1].get_shape().as_list()[0]
                node.filter_width = operation.inputs[1].get_shape().as_list()[1]
        
        elif op == "MaxPool" or op == "AvgPool":
            node.padding = attr['padding'].s.decode('utf-8')
            node.data_format = attr['data_format'].s.decode('utf-8')

            # If empty string, assume default value
            if node.data_format == '':
                node.data_format = "NHWC"

            for index in range(len(node.data_format)):
                if node.data_format[index] == 'H':
                    h_index = index
                if node.data_format[index] == 'W':
                    w_index = index

            node.stride_h = list(attr['strides'].list.i)[h_index]
            node.stride_w = list(attr['strides'].list.i)[w_index]
            node.filter_height = list(attr['ksize'].list.i)[h_index]
            node.filter_width = list(attr['ksize'].list.i)[w_index]

        elif (op == "FusedBatchNorm" or op == "FusedBatchNormV2" or 
                op == "FusedBatchNormV3"):
            node.is_training = attr['is_training'].b

        elif op == "CudnnRNN":
            node.rnn_mode = attr['rnn_mode'].s.decode('utf-8')
            node.input_mode = attr['input_mode'].s.decode('utf-8')
            node.direction = attr['direction'].s.decode('utf-8')
            
            # If empty strings, assume default values
            if node.rnn_mode == '':
                node.rnn_mode = 'lstm'

            if node.input_mode == '':
                node.input_mode = 'linear_input'

            if node.direction == '':
                node.direction = 'unidirectional'

            node.is_training = attr['is_training'].b

        elif op == "ConcatV2":
            if operation != None:
                axis_tensor = operation.inputs[len(operation.inputs) - 1]
                if tf.get_static_value(axis_tensor) != None:
                    node.axis = int(tf.get_static_value(axis_tensor))

        elif op == "Cast":
            node.in_data_type = str(attr['SrcT'])[9:-1]
            node.out_data_type = str(attr['DstT'])[9:-1]

        return node

    def convert(self, operation, node_def):
        """Function to create Node object representing given operation

        Creates a new Node object to represent the given operation.

        Args:
            operation (GraphDef operation object) : The operation to create 
                a node for, is None in case of constructing nodes for functions.
            node_def : The node_def of the operation/function.

        Returns:
            The created Node object instance representing the operation.
        """

        node = Node.Node(label = node_def.name, 
                            operator_type = node_def.op ,value = None)

        node = self._attr_from_node_def(operation, node_def, node)
        return node
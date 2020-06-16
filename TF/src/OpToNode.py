from common import Node
import tensorflow as tf

# Module to convert and operation into a Node object, for more information 
# on ops and their attributes refer to,
# https://www.tensorflow.org/api_docs/python/tf/raw_ops
class OpToNode:
    def __init__(self):
        pass
    
    # Internal helper function to extract attributes from different op types.
    def _attr_from_node_def(self, operation, node):
        attr = operation.node_def.attr
        op = operation.node_def.op


        if op == "Conv2D" or op == "DepthwiseConv2dNative":
            # print(operation.inputs)
            node.padding = attr['padding'].s.decode('utf-8')
            node.data_format = attr['data_format'].s.decode('utf-8')

            for index in range(len(node.data_format)):
                if node.data_format[index] == 'H':
                    h_index = index
                if node.data_format[index] == 'W':
                    w_index = index
            
            node.stride_h = list(attr['strides'].list.i)[h_index]
            node.stride_w = list(attr['strides'].list.i)[w_index]
            node.dilation_w_factor = list(attr['dilations'].list.i)[w_index]
            node.dilation_h_factor = list(attr['dilations'].list.i)[h_index]

            node.filter_height = operation.inputs[1].get_shape().as_list()[0]
            node.filter_width = operation.inputs[1].get_shape().as_list()[1]
        
        elif op == "MaxPool" or op == "AvgPool":

            node.padding = attr['padding'].s.decode('utf-8')
            node.data_format = attr['data_format'].s.decode('utf-8')

            for index in range(len(node.data_format)):
                if node.data_format[index] == 'H':
                    h_index = index
                if node.data_format[index] == 'W':
                    w_index = index

            node.stride_h = list(attr['strides'].list.i)[h_index]
            node.stride_w = list(attr['strides'].list.i)[w_index]
            node.filter_height = list(attr['ksize'].list.i)[h_index]
            node.filter_width = list(attr['ksize'].list.i)[w_index]

        elif op == "FusedBatchNorm":
            node.is_training = attr['is_training'].b

        elif op == "CudnnRNN":
            node.rnn_mode = attr['rnn_mode'].s.decode('utf-8')
            node.input_mode = attr['input_mode'].s.decode('utf-8')
            node.direction = attr['direction'].s.decode('utf-8')
            node.is_training = attr['is_training'].b

        elif op == "ConcatV2":
            axis_tensor = operation.inputs[len(operation.inputs) - 1]
            node.axis = int(tf.get_static_value(axis_tensor))

        elif op == "Cast":
            node.in_data_type = str(attr['SrcT'])[9:-1]
            node.out_data_type = str(attr['DstT'])[9:-1]

        return node

    def convert(self, operation):
        new_node = Node.Node(label = operation.name, value = None)
        new_node.operator_type = operation.node_def.op

        new_node = self._attr_from_node_def(operation, new_node)
        return new_node

from common import Edge
import tensorflow as tf

class TensorToEdge:
    def __init__(self):
        pass

    def convert(self, tensor):

        new_edge = Edge.Edge(label = tensor.op.name, value = None)
        
        new_edge.tensor_type = str(tensor.dtype)[9:-2].upper()
        if tensor.shape:
            new_edge.tensor_shape = tensor.get_shape().as_list()
        return new_edge


"""Module with TensorToEdge class to create Edge objects for tensors"""

import tensorflow as tf

from common import Edge

#Module to convert Tensor to Edge
class TensorToEdge:
    """Class to convert tensor to Edge object"""

    def __init__(self):
        pass

    def convert(self, tensor):
        """Function to create an Edge object representing given Tensor

        Creates a new Edge object to represent the given tensor and populates
        its attributes.

        Args:
            tensor (TF tensor object) : The tensor to create an edge for.

        Returns:
            The created Edge object intance representing the tensor.
        """
        # Creating edge and extracting features
        new_edge = Edge.Edge(label = tensor.name, 
                                tensor_label = tensor.name, value = None)

        new_edge.tensor_type = str(tensor.dtype)[9:-2].upper()
        if tensor.shape:
            new_edge.tensor_shape = tensor.get_shape().as_list()
        return new_edge
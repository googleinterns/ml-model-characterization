"""Module with TensorToEdge class to create Edge objects for tensors"""

import inspect

from common import Edge
from tflite import TensorType

class TensorToEdge:
    """Class to convert tensor to Edge object"""
    
    def __init__(self):
        typeclass = TensorType.TensorType
        self._tensor_type = dict() # Dictionary to map enum value to enum name
        for member in inspect.getmembers(typeclass):
            if not member[0].startswith('_'):
                if not inspect.ismethod(member[1]):
                    self._tensor_type[member[1]] = member[0]

    def convert(self, tensor):
        """Function to create an Edge object representing given Tensor

        Creates a new Edge object to represent the given tensor and populates
        its attributes.

        Args:
            tensor (tflite/Tensor object) : The tensor to create an edge for.

        Returns:
            The created Edge object instance representing the tensor.
        """

        new_edge = Edge.Edge(label = tensor.Name(), 
                                tensor_label = tensor.Name(), value = tensor)
        new_edge.tensor_type = self._tensor_type[tensor.Type()]

        shape = list()
        for i in range(tensor.ShapeLength()):
            shape.append(tensor.Shape(i))
        new_edge.tensor_shape = shape

        return new_edge

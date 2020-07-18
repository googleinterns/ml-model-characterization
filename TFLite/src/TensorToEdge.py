from common import Edge
from tflite import TensorType
import inspect

#Module to convert Tensor to Edge
class TensorToEdge:
    
    def __init__(self):

        # Dictionary to map enum value to enum name
        typeclass = TensorType.TensorType
        self._tensor_type = dict()
        for member in inspect.getmembers(typeclass):
            if not member[0].startswith('_'):
                if not inspect.ismethod(member[1]):
                    self._tensor_type[member[1]] = member[0]

    def convert(self, tensor):

        # Creating edge and extracting features
        new_edge = Edge.Edge(label = tensor.Name(), value = tensor)
        new_edge.tensor_type = self._tensor_type[tensor.Type()]

        shape = list()
        for i in range(tensor.ShapeLength()):
            shape.append(tensor.Shape(i))
        new_edge.tensor_shape = shape

        return new_edge

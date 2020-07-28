"""Module with Edge class to store Tensor attributes."""

class Edge:
    """Edge class to represent tensors.
    
    Stores tensors and their extracted attributes.

    Attributes:
        label (str): Label for the instance, intended for programmer's use.
        value (tensor): The complete tensor object in data type as 
            extracted from the file format.
        tensor_label (str): The name if the tensor as extracted from the file format.
        tensor_shape (list of int, optional): Shape of the tensor.
        tensor_type (str, optional): Data type of the tensor.
    """

    def __init__(self, label, tensor_label, value = None):
        self.label = label
        self.value = value
        self.tensor_label = tensor_label
        
        self.tensor_shape = None
        self.tensor_type = None

    def serialize(self):
        """Helper method to serialize instance.

        Serializes attributes of the instance to a string, except 'value' and
        'label' attributes.

        Returns:
            A string which contains space separated attributes values of the
            instance.

        """
        ret_str = ""
        attrs = vars(self)
        for item in attrs.items():
            if item[0] != 'value' and item[0] != 'label':
                ret_str += str(item[1]) + " "

        return ret_str
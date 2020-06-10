# Edge module to store tensor attributes
class Edge:
    def __init__(self, label, value = None):
        self.label = label
        self.value = value
        
        self.tensor_label = label
        self.tensor_shape = None
        self.tensor_type = None

    # helper function to convert all attributes to a string except value and label
    def serialize(self):
        ret_str = ""
        attrs = vars(self)
        for item in attrs.items():
            if item[0] != 'value' and item[0] != 'label':
                ret_str += str(item[1]) + " "

        return ret_str
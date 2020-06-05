# Edge module to store tensor attributes

class Edge:
    def __init__(self, label, value = None):
        self.label = label
        self.value = value
        self.shape = None
        self.type = None
"""Module with Node class to store Operator attributes."""

class Node:
    """ Node class to represent operators. 

    Stores tensors and their extracted attributes.

    Attributes:
        label (str): label for the instance, intended for programmer's use.
        value (operator): The complete operator object in data type 
            as extracted from the file format.
        operator_type (str) : Layer name, for example: Conv2D

        Following attributes are extracted from OpToNode.py 
        file of specific file formats, refer to the link in the same for
        sematic meaning

        padding (str, optional)
        activation_function (str, optional)
        stride_h (int, optional)
        stride_w (int, optional)
        dilation_h_factor (int, optional)
        dilation_w_factor (int, optional)
        depth_multiplier (int, optional)
        num_channels (int, optional)
        lsh_projection_type (str, optional)
        filter_width (int, optional)
        filter_height (int, optional)
        asymmetric_quantize_inputs (bool, optional)
        weights_format (str, optional)
        keep_num_dims (bool, optional)
        axis (int, optional)
        lstm_kernel_type (str, optional)
        ngram_size (int, optional)
        max_skip_size (int, optional)
        include_all_ngrams (bool, optional)
        combiner (str, optional)
        num_splits (int, optional)
        in_data_type (str, optional)
        out_data_type (str, optional)
        merge_outputs (bool, optional)
        mirror_pad_mode (str, optional)
        data_format (str, optional)
        is_training (bool, optional)
        rnn_mode (str, optional)
        input_mode (str, optional)
        direction (str, optional)
    """
    
    def __init__(self, label, operator_type, value = None):

        self.label = label
        self.value = value
        self.operator_type = operator_type

        self.padding = None
        self.activation_function = None
        self.stride_h = None
        self.stride_w = None
        self.dilation_h_factor = None
        self.dilation_w_factor = None
        self.depth_multiplier = None
        self.num_channels = None
        self.lsh_projection_type = None
        self.filter_width = None
        self.filter_height = None
        self.asymmetric_quantize_inputs = None
        self.weights_format = None
        self.keep_num_dims = None
        self.axis = None
        self.lstm_kernel_type = None
        self.ngram_size = None
        self.max_skip_size = None
        self.include_all_ngrams = None
        self.combiner = None
        self.num_splits = None
        self.in_data_type = None
        self.out_data_type = None
        self.merge_outputs = None
        self.mirror_pad_mode = None
        self.data_format = None
        self.is_training = None
        self.rnn_mode = None
        self.input_mode = None
        self.direction = None

    def serialize(self):
        """ Helper method to serialize instance.

        Serializes attributes of the instance to a string, except 'value'
        and 'label' attributes.

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
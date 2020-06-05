# Node module to store operator attributes

class Node:
    def __init__(self, label, value = None):
        self.label = label
        self.value = value
        self.padding = None
        self.activation_function = None
        self.stride_h = None
        self.stride_w = None
        self.dilation_h_factor = None
        self.dilation_w_factor = None
        self.depth_multiplier = None
        self.num_channels = None
        self.lsh_projection = None
        self.filter_width = None
        self.filter_height = None
        self.rank = None
        self.asymmetric_quantize_inputs = None
        self.weights_format = None
        self.keep_num_dims = None
        self.axis = None
        self.kernel_type = None
        self.ngram_size = None
        self.max_skip_size = None
        self.include_all_ngrams = None
        self.combiner = None
        self.keep_dims = None
        self.num_splits = None
        self.in_data_type = None
        self.out_data_type = None
        self.min = None
        self.max = None
        self.merge_outputs = None
        self.mirror_pad_mode = None
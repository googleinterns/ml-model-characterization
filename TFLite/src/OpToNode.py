"""Module with OpToNode class to create Node objects for operators"""

import inspect

from common import Node
from tflite import *

class OpToNode:
    """Class to convert operation to Node object"""

    _OP_TFLITE_TO_TF = {
        "ADD" : "Add", "AddN" : "ADD_N", "DIV" : "Div", "SUB" : "Sub",
        "AVERAGE_POOL_2D" : "AveragePool", "SpaceToBatchND" : "SPACE_TO_BATCH_ND",
        "BATCH_TO_SPACE_ND" : "BatchToSpaceND", "CONCATENATION" : "Concat", 
        "CONV_2D" : "Conv2D", "DEPTHWISE_CONV_2D" : "DepthwiseConv2dNative",
        "DEQUANTIZE" : "Dequantize", "GATHER" : "Gather", 
        "GATHER_ND" : "GatherNd", "L2_NORMALIZATION" : "L2_NORMALIZATION", 
        "L2_POOL_2D" : "L2_POOL_2D", "LOCAL_RESPONSE_NORMALIZATION" : "LRN", 
        "MAX_POOL_2D" : "MaxPool", "MUL" : "Mul",  "PAD" : "Pad",
        "PADV2" : "PadV2", "RESHAPE" : "Reshape", "SOFTMAX" : "Softmax",
        "SPACE_TO_DEPTH" : "SpaceToDepth", "DEPTH_TO_SPACE" : "DepthToSpace",
        "TRANSPOSE" : "Transpose", "MEAN" : "Mean", "SUM" : "Sum",
        "REDUCE_PROD" : "Prod", "REDUCE_MAX" : "Max", "REDUCE_MIN" : "Min",
        "REDUCE_ANY" : "Any", "RESIZE_BILINEAR" : "ResizeBilinear",
        "RESIZE_NEAREST_NEIGHBOR" : "ResizeNearestNeighbor", "SQUEEZE" : "Squeeze",
        "SPLIT" : "Split", "SPLIT_V" : "SplitV", "STRIDED_SLICE" : "StridedSlice",
        "TOPK_V2" : "TopKV2", "CAST" : "Cast", "ARG_MAX" : "ArgMax", 
        "ARG_MIN" : "ArgMin", "TILE" : "Tile", "EXPAND_DIMS" : "ExpandDims",
        "TRANSPOSE_CONV" : "Conv2DBackpropInput", 
        "SPARSE_TO_DENSE" : "SparseToDense", "SHAPE" : "Shape",
        "PACK" : "Pack", "ONE_HOT" : "OneHot", "UNPACK" : "Unpack",
        "LEAKY_RELU" : "LeakyRelu", "SQUARED_DIFFERENCE" : "SquaredDifference",
        "MIRROR_PAD" : "MirrorPad", "UNIQUE" : "Unique", "WHERE" : "Where",
        "REVERSE_SEQUENCE" : "ReverseSequence", "MATRIX_DIAG" : "MatrixDiag",
        "MATRIX_SET_DIAG" : "MatrixSetDiag", "FLOOR" : "Floor", "CEIL" : "Ceil",
        "ELU" : "Elu", "ROUND" : "Round", "RELU" : "Relu", "RELU6" : "Relu6",
        "LOGISTIC" : "Sigmoid", "TANH" : "Tanh", "EXP" : "Exp", "COS" : "Cos",
        "LOG_SOFTMAX" : "LogSoftmax", "MAXIMUM" : "Maximum", "MINIMUM" : "Minimum",
        "GREATER" : "Greater", "GREATER_EQUAL" : "GreaterEqual", "LESS" : "Less",
        "LESS_EQUAL" : "LessEqual", "EQUAL" : "Equal", "NOT_EQUAL" : "NotEqual",
        "NEG" : "Neg", "SELECT" : "Select", "SLICE" : "Slice", "POW" : "Pow",
        "LOGICAL_OR" : "LogicalOr", "LOGICAL_AND" : "LogicalAnd",
        "LOGICAL_NOT" : "LogicalNot", "FLOOR_DIV" : "FloorDiv", 
        "FLOOR_MOD" : "FloorMod", "RANGE" : "Range", "SIN" : "Sin", 
        "LOG" : "Log", "SQRT" : "Sqrt", "RSQRT" : "Rsqrt", "SQUARE" : "Square",
        "ZEROS_LIKE" : "ZerosLike", "ABS" : "Abs", "HARD_SWISH" : "HardSwish",
        "FILL" : "Fill", "REVERSE_V2" : "ReverseV2", "RANK" : "Rank",
        "SEGMENT_SUM" : "SegmentSum", "SCATTER_ND" : "ScatterNd"
    } # Mapping from TFLite to TF operations,
    # contains only those ops which have a one-one mapping


    def __init__(self):

        # Dictionaries for enum value to enum name mapping
        class_type = Padding.Padding
        self._padding = dict()
        self._fill_dict(class_type, self._padding)


        class_type = ActivationFunctionType.ActivationFunctionType
        self._activation_function_type = dict()
        self._fill_dict(class_type, self._activation_function_type)


        class_type = LSHProjectionType.LSHProjectionType
        self._lsh_projection_type = dict()
        self._fill_dict(class_type, self._lsh_projection_type)
        

        class_type = FullyConnectedOptionsWeightsFormat.FullyConnectedOptionsWeightsFormat
        self._weights_format = dict()
        self._fill_dict(class_type, self._weights_format)


        class_type = LSTMKernelType.LSTMKernelType
        self._lstm_kernel_type = dict()
        self._fill_dict(class_type, self._lstm_kernel_type)


        class_type = CombinerType.CombinerType
        self._combiner_type = dict()
        self._fill_dict(class_type, self._combiner_type)

        class_type = TensorType.TensorType
        self._tensor_type = dict()
        self._fill_dict(class_type, self._tensor_type)


        class_type = MirrorPadMode.MirrorPadMode
        self._mirror_pad_mode = dict()
        self._fill_dict(class_type, self._mirror_pad_mode)

    # Function to fill dictionary with inverse 
    # enum mappings of class 'class_type'
    def _fill_dict(self, class_type, val_to_name):
        """ Internal method to fill dictionary

        Fills dictionary with inverse enum mappings from value to name
        
        Args:
            class_type (tflite/* Object) : The class for which inverse enum 
                mapping is done.
            val_to_name (dict of int to str) : The dictionary to fill.

        """
        for member in inspect.getmembers(class_type):
            if not member[0].startswith('_'):
                if not inspect.ismethod(member[1]):
                    val_to_name[member[1]] = member[0]

    """ Internal methods to cast BuiltinOptions at runtime and extract options

    Internal methods ending with '_options' cast the BuiltinOptions of operator
    at runtime and extracts the relevant options and adds them as attribute 
    values to node.
    Dummy methods exist for future improvements.
    For semantic information on Node attributes for TFLite,
    https://www.tensorflow.org/lite/guide/ops_compatibility#unsupported_operations


    Args:
        operator (tflite/Operator object) :  Operator that the node represents
        node (Node object) : The Node object to add attribute values to.

    Returns:
        The modified Node object with added attribute values.
    
    """
    def _none_options(self, operator, node):
        return node
        
    def _conv2d_options(self, operator, node):
        # Casting into required Class
        options = Conv2DOptions.Conv2DOptions() 
        buf = operator.BuiltinOptions().Bytes
        pos = operator.BuiltinOptions().Pos
        options.Init(buf, pos) 
        
        # Extracting attributes
        node.padding = self._padding[options.Padding()] 
        
        val = options.FusedActivationFunction()
        node.activation_function = self._activation_function_type[val]
        
        node.stride_h = options.StrideH() 
        node.stride_w = options.StrideW() 
        node.dilation_h_factor = options.DilationHFactor() 
        node.dilation_w_factor = options.DilationWFactor()

        return node
    
    def _depthwise_conv2d_options(self, operator, node):
        options = DepthwiseConv2DOptions.DepthwiseConv2DOptions()
        buf = operator.BuiltinOptions().Bytes
        pos = operator.BuiltinOptions().Pos
        options.Init(buf, pos) 
        
        node.padding = self._padding[options.Padding()] 
        
        val = options.FusedActivationFunction()
        node.activation_function = self._activation_function_type[val]
        
        node.depth_multiplier = options.DepthMultiplier()
        node.stride_h = options.StrideH() 
        node.stride_w = options.StrideW() 
        node.dilation_h_factor = options.DilationHFactor()
        node.dilation_w_factor = options.DilationWFactor()

        return node
    
    def _concat_embeddding_options(self, operator, node):
        options = ConcatEmbeddingsOptions.ConcatEmbeddingsOptions()
        buf = operator.BuiltinOptions().Bytes
        pos = operator.BuiltinOptions().Pos
        options.Init(buf, pos) 

        node.num_channels = options.NumChannels()

        return node
    
    def _lsh_projection_options(self, operator, node):
        options = LSHProjectionOptions.LSHProjectionOptions()
        buf = operator.BuiltinOptions().Bytes
        pos = operator.BuiltinOptions().Pos
        options.Init(buf, pos) 

        node.lsh_projection_type = self._lsh_projection_type [options.Type()]

        return node
    
    def _pool2d_options(self, operator, node):
        options = Pool2DOptions.Pool2DOptions()
        buf = operator.BuiltinOptions().Bytes
        pos = operator.BuiltinOptions().Pos
        options.Init(buf, pos)

        node.padding = self._padding[options.Padding()] 
        
        val = options.FusedActivationFunction()
        node.activation_function = self._activation_function_type[val]
        
        node.stride_h = options.StrideH() 
        node.stride_w = options.StrideW() 
        node.filter_width = options.FilterWidth()
        node.filter_height = options.FilterHeight()

        return node
    
    def _svdf_options(self, operator, node):
        options = SVDFOptions.SVDFOptions()
        buf = operator.BuiltinOptions().Bytes
        pos = operator.BuiltinOptions().Pos
        options.Init(buf, pos)
        
        val = options.FusedActivationFunction()
        node.activation_function = self._activation_function_type[val]
        
        node.asymmetric_quantize_inputs = options.AsymmetricQuantizeInputs()

        return node

    def _rnn_options(self, operator, node):
        options = RNNOptions.RNNOptions()
        buf = operator.BuiltinOptions().Bytes
        pos = operator.BuiltinOptions().Pos
        options.Init(buf, pos)

        
        val = options.FusedActivationFunction()
        node.activation_function = self._activation_function_type[val]
        
        node.asymmetric_quantize_inputs = options.AsymmetricQuantizeInputs()

        return node
    
    def _fully_connected_options(self, operator, node):
        options = FullyConnectedOptions.FullyConnectedOptions()
        buf = operator.BuiltinOptions().Bytes
        pos = operator.BuiltinOptions().Pos
        options.Init(buf, pos)

        
        val = options.FusedActivationFunction()
        node.activation_function = self._activation_function_type[val]
        
        node.weights_format = self._weights_format[options.WeightsFormat()]
        node.keep_num_dims = options.KeepNumDims()
        node.asymmetric_quantize_inputs = options.AsymmetricQuantizeInputs()

        return node

    def _softmax_options(self, operator, node):
        return node

    def _concatenation_options(self, operator, node):
        options = ConcatenationOptions.ConcatenationOptions()
        buf = operator.BuiltinOptions().Bytes
        pos = operator.BuiltinOptions().Pos
        options.Init(buf, pos)

        node.axis = options.Axis()
        
        val = options.FusedActivationFunction()
        node.activation_function = self._activation_function_type[val]


        return node
            
    def _add_options(self, operator, node):
        options = AddOptions.AddOptions()
        buf = operator.BuiltinOptions().Bytes
        pos = operator.BuiltinOptions().Pos
        options.Init(buf, pos)

        
        val = options.FusedActivationFunction()
        node.activation_function = self._activation_function_type[val]


        return node
    
    def _l2norm_options(self, operator, node):
        options = L2NormOptions.L2NormOptions()
        buf = operator.BuiltinOptions().Bytes
        pos = operator.BuiltinOptions().Pos
        options.Init(buf, pos)

        
        val = options.FusedActivationFunction()
        node.activation_function = self._activation_function_type[val]


        return node
    
    def _local_response_norm_options(self, operator, node):
        return node
    
    def _lstm_options(self, operator, node):
        options = LSTMOptions.LSTMOptions()
        buf = operator.BuiltinOptions().Bytes
        pos = operator.BuiltinOptions().Pos
        options.Init(buf, pos)

        
        val = options.FusedActivationFunction()
        node.activation_function = self._activation_function_type[val]
        
        node.lstm_kernel_type = self._lstm_kernel_type[options.KernelType()]
        node.asymmetric_quantize_inputs = options.AsymmetricQuantizeInputs()

        return node
    
    def _resize_bilinear_options(self, operator, node):
        return node
    
    def _call_options(self, operator, node):
        return node
    
    def _reshape_options(self, operator, node):
        return node

    def _skipgram_options(self, operator, node):
        options = SkipGramOptions.SkipGramOptions()
        buf = operator.BuiltinOptions().Bytes
        pos = operator.BuiltinOptions().Pos
        options.Init(buf, pos)

        node.ngram_size = options.NgramSize()
        node.max_skip_size = options.MaxSkipSize()
        node.include_all_ngrams = options.IncludeAllNgrams()

        return node
    
    def _space_to_depth_options(self, operator, node):
        return node
        
    def _embedding_lookup_sparse_options(self, operator, node):
        options = EmbeddingLookupSparseOptions.EmbeddingLookupSparseOptions()
        buf = operator.BuiltinOptions().Bytes
        pos = operator.BuiltinOptions().Pos
        options.Init(buf, pos)

        node.combiner = self._combiner_type[options.Combiner()]

        return node
    
    def _mul_options(self, operator, node):
        options = MulOptions.MulOptions()
        buf = operator.BuiltinOptions().Bytes
        pos = operator.BuiltinOptions().Pos
        options.Init(buf, pos)

        
        val = options.FusedActivationFunction()
        node.activation_function = self._activation_function_type[val]


        return node
    
    def _pad_options(self, operator, node):
        return node
    
    def _gather_options(self, operator, node):
        options = GatherOptions.GatherOptions()
        buf = operator.BuiltinOptions().Bytes
        pos = operator.BuiltinOptions().Pos
        options.Init(buf, pos)

        node.axis = options.Axis()

        return node
    
    def _batch_to_space_nd_options(self, operator, node):
        return node
    
    def _space_to_batch_nd_options(self, operator, node):
        return node
    
    def _transpose_options(self, operator, node):
        return node
    
    def _reducer_options(self, operator, node):
        options = ReducerOptions.ReducerOptions()
        buf = operator.BuiltinOptions().Bytes
        pos = operator.BuiltinOptions().Pos
        options.Init(buf, pos)

        return node
    
    def _sub_options(self, operator, node):
        options = SubOptions.SubOptions()
        buf = operator.BuiltinOptions().Bytes
        pos = operator.BuiltinOptions().Pos
        options.Init(buf, pos)

        
        val = options.FusedActivationFunction()
        node.activation_function = self._activation_function_type[val]


        return node
    
    def _div_options(self, operator, node):
        options = DivOptions.DivOptions()
        buf = operator.BuiltinOptions().Bytes
        pos = operator.BuiltinOptions().Pos
        options.Init(buf, pos)

        
        val = options.FusedActivationFunction()
        node.activation_function = self._activation_function_type[val]


        return node
    
    def _squeeze_options(self, operator, node):
        return node
    
    def _sequence_rnn_options(self, operator, node):
        options = SequenceRNNOptions.SequenceRNNOptions()
        buf = operator.BuiltinOptions().Bytes
        pos = operator.BuiltinOptions().Pos
        options.Init(buf, pos)

        
        val = options.FusedActivationFunction()
        node.activation_function = self._activation_function_type[val]
        
        node.asymmetric_quantize_inputs = options.AsymmetricQuantizeInputs()

        return node
    
    def _strided_slice_options(self, operator, node):
        return node
    
    def _exp_options(self, operator, node):
        return node
    
    def _top_kv2_options(self, operator, node):
        return node
    
    def _split_options(self, operator, node):
        options = SplitOptions.SplitOptions()
        buf = operator.BuiltinOptions().Bytes
        pos = operator.BuiltinOptions().Pos
        options.Init(buf, pos)

        node.num_splits = options.NumSplits()

        return node
    
    def _log_softmax_options(self, operator, node):
        return node
    
    def _cast_options(self, operator, node):
        options = CastOptions.CastOptions()
        buf = operator.BuiltinOptions().Bytes
        pos = operator.BuiltinOptions().Pos
        options.Init(buf, pos)

        node.in_data_type = self._tensor_type[options.InDataType()]
        node.out_data_type = self._tensor_type[options.OutDataType()]

        return node
    
    def _dequantize_options(self, operator, node):
        return node
    
    def _maximum_minimum_options(self, operator, node):
        return node
    
    def _argmax_options(self, operator, node):
        return node
    
    def _less_options(self, operator, node):
        return node
    
    def _neg_options(self, operator, node):
        return node
    
    def _padv2_options(self, operator, node):
        return node
    
    def _greater_options(self, operator, node):
        return node
    
    def _greater_equal_options(self, operator, node):
        return node
    
    def _less_equal_options(self, operator, node):
        return node
    
    def _select_options(self, operator, node):
        return node
    
    def _slice_options(self, operator, node):
        return node
    
    def _transpose_conv_options(self, operator, node):
        options = TransposeConvOptions.TransposeConvOptions()
        buf = operator.BuiltinOptions().Bytes
        pos = operator.BuiltinOptions().Pos
        options.Init(buf, pos)

        node.padding = self._padding[options.Padding()]
        node.stride_h = options.StrideH()
        node.stride_w = options.StrideW()

        return node
    
    def _sparse_to_dense_options(self, operator, node):
        return node
    
    def _tile_options(self, operator, node):
        return node
    
    def _expand_dims_options(self, operator, node):
        return node
    
    def _equal_options(self, operator, node):
        return node
    
    def _not_equal_options(self, operator, node):
        return node
    
    def _shape_options(self, operator, node):
        return node

    def _pow_options(self, operator, node):
        return node
        
    def _argmin_options(self, operator, node):
        return node
    
    def _fake_quant_options(self, operator, node):        
        return node
    
    def _pack_options(self, operator, node):
        options = PackOptions.PackOptions()
        buf = operator.BuiltinOptions().Bytes
        pos = operator.BuiltinOptions().Pos
        options.Init(buf, pos)

        node.axis = options.Axis()

        return node
    
    def _logical_or_options(self, operator, node):
        return node
    
    def _onehot_options(self, operator, node):
        options = OneHotOptions.OneHotOptions()
        buf = operator.BuiltinOptions().Bytes
        pos = operator.BuiltinOptions().Pos
        options.Init(buf, pos)

        node.axis = options.Axis()

        return node
    
    def _logical_and_options(self, operator, node):
        return node
    
    def _logical_not_options(self, operator, node):
        return node
    
    def _unpack_options(self, operator, node):
        options = UnpackOptions.UnpackOptions()
        buf = operator.BuiltinOptions().Bytes
        pos = operator.BuiltinOptions().Pos
        options.Init(buf, pos)

        node.axis = options.Axis()

        return node
    
    def _floor_div_options(self, operator, node):
        return node
    
    def _square_options(self, operator, node):
        return node   
    
    def _zeros_like_options(self, operator, node):
        return node 
    
    def _fill_options(self, operator, node):
        return node
    
    def _bidirectional_sequence_lstm_options(self, operator, node):
        options = BidirectionalSequenceLSTMOptions.BidirectionalSequenceLSTMOptions()
        buf = operator.BuiltinOptions().Bytes
        pos = operator.BuiltinOptions().Pos
        options.Init(buf, pos)

        
        val = options.FusedActivationFunction()
        node.activation_function = self._activation_function_type[val]
        
        node.asymmetric_quantize_inputs = options.AsymmetricQuantizeInputs()
        node.merge_outputs = options.MergeOutputs()

        return node
    
    def _bidirectional_sequence_rnn_options(self, operator, node):
        options = BidirectionalSequenceRNNOptions.BidirectionalSequenceRNNOptions()
        buf = operator.BuiltinOptions().Bytes
        pos = operator.BuiltinOptions().Pos
        options.Init(buf, pos)

        
        val = options.FusedActivationFunction()
        node.activation_function = self._activation_function_type[val]
        
        node.asymmetric_quantize_inputs = options.AsymmetricQuantizeInputs()
        node.merge_outputs = options.MergeOutputs()

        return node
    
    def _unidirectional_sequence_lstm_options(self, operator, node):
        options = UnidirectionalSequenceLSTMOptions.UnidirectionalSequenceLSTMOptions()
        buf = operator.BuiltinOptions().Bytes
        pos = operator.BuiltinOptions().Pos
        options.Init(buf, pos)

        
        val = options.FusedActivationFunction()
        node.activation_function = self._activation_function_type[val]
        
        node.asymmetric_quantize_inputs = options.AsymmetricQuantizeInputs()

        return node
    
    def _floor_mod_options(self, operator, node):
        return node
    
    def _range_options(self, operator, node):
        return node
    
    def _resize_nearest_neighbor_options(self, operator, node):
        return node
        
    def _leaky_relu_options(self, operator, node):
        return node
    
    def _squared_difference_options(self, operator, node):
        return node
    
    def _mirror_pad_options(self, operator, node):
        options = MirrorPadOptions.MirrorPadOptions()
        buf = operator.BuiltinOptions().Bytes
        pos = operator.BuiltinOptions().Pos
        options.Init(buf, pos)

        node.mirror_pad_mode = self._mirror_pad_mode[options.Mode()]

        return node
    
    def _abs_options(self, operator, node):
        return node
    
    def _splitv_options(self, operator, node):
        options = SplitVOptions.SplitVOptions()
        buf = operator.BuiltinOptions().Bytes
        pos = operator.BuiltinOptions().Pos
        options.Init(buf, pos)

        node.num_splits = options.NumSplits()

        return node
    
    def _unique_options(self, operator, node):
        return node
    
    def _reversev2_options(self, operator, node):
        return node
    
    def _addn_options(self, operator, node):
        return node
    
    def _gathernd_options(self, operator, node):
        return node
    
    def _cos_options(self, operator, node):
        return node
    
    def _where_options(self, operator, node):
        return node
    
    def _rank_options(self, operator, node):
        return node
    
    def _reverse_sequence_options(self, operator, node):
        return node
    
    def _matrix_diag_options(self, operator, node):
        return node
    
    def _quantize_options(self, operator, node):
        return node
    
    def _matrix_set_diag_options(self, operator, node):
        return node
    
    def _hard_swish_options(self, operator, node):
        return node
    
    def _if_options(self, operator, node):
        return node
    
    def _while_options(self, operator, node):
        return node
    
    def _depth_to_space_options(self, operator, node):
        return node
    
    def _nonmax_suppresionv4_options(self, operator, node):
        return node
        
    def _nonmax_suppresionv5_options(self, operator, node):
        return node
        
    def _scatternd_options(self, operator, node):
        return node
    
    def _selectv2_options(self, operator, node):
        return node
    
    def _densify_options(self, operator, node):
        return node
    
    def _segment_sum_options(self, operator, node):
        return node
    
    def _batch_matmul_options(self, operator, node):
        return node
  
    def convert(self, operator, opname):
        """Function to create Node object representing given operator

        Creates a new Node object to represent the given operator.

        Args:
            operator (tflite/Operator object) : The operator to create 
                a node for.
            opname (str) : Name of the operator.

        Returns:
            The created Node object instance representing the operator
        """

        node = Node.Node(label = opname, operator_type = opname, 
                            value = operator)

        # List of internal methods of class for casting BuiltinOptions
        methods = list()
        for method in inspect.getmembers(OpToNode, predicate = inspect.isroutine):
            if method[0].endswith('_options'):
                methods.append(method)

        # methods with the name *_options sorted in accordance 
        # with enum value for efficient calling
        methods.sort(key = lambda mem : mem[1].__code__.co_firstlineno)

        node_options = list()
        for method in methods:
            node_options.append(method[1])

        type_val = operator.BuiltinOptionsType()
        node = node_options[type_val](self, operator, node)
        if node.operator_type in self._OP_TFLITE_TO_TF:
            node.operator_type = self._OP_TFLITE_TO_TF[node.operator_type]

        return node
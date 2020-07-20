# Builds tf_main, tfite_main and tflite_init
# tf_main parses a .pb(Frozen graph and Saved model) file into a graph
# tflite_init compiles schema.fbs with flatc to generate python code files
# tflite_init to be run before tflite_main
# tflite_main parser a .tflite file into a graph

# Common code

py_library(
    name = 'Node',
    srcs = ['common/Node.py'],
)

py_library(
    name = 'Edge',
    srcs = ['common/Edge.py'],
)

py_library(
    name = 'Graph',
    srcs = ['common/Graph.py'],
    deps = [':Node', ':Edge'],
)

py_library(
    name = 'Vectorize',
    srcs = ['common/Vectorize.py'],
)

py_library(
    name = 'Storage',
    srcs = ['common/Storage.py'],
    deps = [':Graph'],
)

# TF
py_library(
    name = 'TFOpToNode',
    srcs = ['TF/src/OpToNode.py'],
    deps = [':Node'],
    imports = ['common/'],
)

py_library(
    name = 'TFTensorToEdge',
    srcs = ['TF/src/TensorToEdge.py'],
    deps = [':Edge'],
)

py_library(
    name = 'TFParser',
    srcs = ['TF/src/TFParser.py'], 
    deps = [':Node', ':Edge', ':Graph', ':TFOpToNode', ':TFTensorToEdge'],
)

py_binary(
    name = 'tf_main',
    srcs = ['TF/src/main.py'],
    deps = [':TFParser', ':Storage'],
    main = 'TF/src/main.py',
)

# TFLite
filegroup(
    name = 'tflitefiles',
    srcs = glob(['TFLite/src/tflite/*.py'])
)

py_library(
    name = 'TFLiteOpToNode',
    srcs = ['TFLite/src/OpToNode.py'],
    deps = [':Node'],
    data = [':tflitefiles'],
)

py_library(
    name = 'TFLiteTensorToEdge',
    srcs = ['TFLite/src/TensorToEdge.py'],
    deps = [':Edge'],
    data = [':tflitefiles'],
)

py_library(
    name = 'TFLiteParser',
    srcs = ['TFLite/src/TFLiteParser.py'], 
    deps = [':Node', ':Edge', ':Graph', ':TFLiteOpToNode', ':TFLiteTensorToEdge'],
    data = [':tflitefiles'],
)

py_binary(
    name = 'tflite_init',
    srcs = ['TFLite/src/init.py'],
    main = 'TFLite/src/init.py',
)

py_binary(
    name = 'tflite_main',
    srcs = ['TFLite/src/main.py'],
    deps = [':TFLiteParser', ':Storage'],
    main = 'TFLite/src/main.py',
)

py_binary(
    name = 'similarity',
    srcs = ['common/similarity.py'],
    deps = [':Storage', ':Vectorize'],
)

py_binary(
    name = 'compare',
    srcs = ['common/compare.py'],
    deps = [':Storage', ':Vectorize', ':TFParser', ':TFLiteParser'],
    imports = ['./TFLite/src/']
)
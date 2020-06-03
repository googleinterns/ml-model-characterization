
py_library(
    name = 'Node',
    srcs = ['src/Node.py'],
)

py_library(
    name = 'Edge',
    srcs = ['src/Edge.py'],
)

py_library(
    name = 'Graph',
    srcs = ['src/Graph.py'],
    deps = [':Node', ':Edge'],
)

py_library(
    name = 'OpToNode',
    srcs = ['src/OpToNode.py'],
    deps = [':Node'],
)

py_library(
    name = 'TensorToEdge',
    srcs = ['src/TensorToEdge.py'],
    deps = [':Edge'],
)

py_library(
    name = 'TFLiteParser',
    srcs = ['src/TFLiteParser.py'], 
    deps = [':Node', ':Edge', ':Graph', ':OpToNode', ':TensorToEdge'],
)

py_binary(
    name = 'main',
    srcs = ['src/main.py'],
    deps = [':TFLiteParser'],
)
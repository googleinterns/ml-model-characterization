# Builds main
# main parses a .pb file into a graph and prints it

py_library(
    name = 'Node',
    srcs = ['src/common/Node.py'],
)

py_library(
    name = 'Edge',
    srcs = ['src/common/Edge.py'],
)

py_library(
    name = 'Graph',
    srcs = ['src/common/Graph.py'],
    deps = [':Node', ':Edge'],
)

py_library(
    name = 'Storage',
    srcs = ['src/common/Storage.py'],
    deps = [':Graph']
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
    name = 'TFParser',
    srcs = ['src/TFParser.py'], 
    deps = [':Node', ':Edge', ':Graph', ':OpToNode', ':TensorToEdge'],
)

py_binary(
    name = 'main',
    srcs = ['src/main.py'],
    deps = [':TFParser'],
)
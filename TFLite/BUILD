# Builds main and init
# init compiles schema.fbs with flatc to generate python code files
# main parses a tflite file into a graph and prints it

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

filegroup(
    name = "tflitefiles",
    srcs = glob(['src/tflite/*.py'])
)

py_library(
    name = 'OpToNode',
    srcs = ['src/OpToNode.py'],
    deps = [':Node'],
    data = [':tflitefiles'],
)

py_library(
    name = 'TensorToEdge',
    srcs = ['src/TensorToEdge.py'],
    deps = [':Edge'],
    data = [':tflitefiles'],
)

py_library(
    name = 'TFLiteParser',
    srcs = ['src/TFLiteParser.py'], 
    deps = [':Node', ':Edge', ':Graph', ':OpToNode', ':TensorToEdge'],
    data = [':tflitefiles'],
)

py_binary(
    name = "init",
    srcs = ['src/init.py'],
)

py_binary(
    name = 'main',
    srcs = ['src/main.py'],
    deps = [':TFLiteParser'],
)
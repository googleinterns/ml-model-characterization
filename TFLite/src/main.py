import TFLiteParser
import os

if __name__ == "__main__":

    tflite_parser = TFLiteParser.TFLiteParser()

    graph = tflite_parser.parse_graph("models/01062020-193418-e2.tflite")

    print("Number of inputs:", graph.num_inputs)
    print("Number of outputs:", graph.num_outputs)
    print("Max fan-in:", graph.max_fan_in)
    print("Max fan-out:", graph.max_fan_out)
    graph.print_graph()
    graph.print_nodes()
    graph.print_edges()
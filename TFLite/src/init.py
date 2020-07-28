"""Module to initialise and generate code for parsing tflite files.

Attributes:
    SCHEMA_PATH (str) : Path to schema.fbs downloaded from
        https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/schema/schema.fbs.
    OUT_PATH (str) : Path to store generated code files.
"""

import os

SCHEMA_PATH = "./TFLite/schema.fbs"
OUT_PATH = "./TFLite/src/"

def init_tflite_files():
    """Function to generate code files from flatbuffer schema.

    Compiles flatbuffer schema using flatc compiler and generates code files
    for reading the .tflite file.
    """
    print("Creating tflite files")

    # Compiling schema.fbs into python code
    os.system('flatc --python -o ' + OUT_PATH + ' ' + SCHEMA_PATH)

    #Echoing code to __init__.py for corect funtioning of 'from tflite import *'
    os.system(
        'echo "from os import path\n'
        'import glob\n\n'
        'modules = glob.glob(path.join(path.dirname(__file__), \'*.py\'))\n'
        'str_modules = str()\n'
        'for module in modules:\n'
        '   if path.isfile(module) and not module.endswith(\'__init__.py\'):\n'
        '       str_modules += path.basename(module)[:-3] + \' \'\n\n'
        'str_modules = str_modules[:-1]\n'
        '__all__ = str_modules.split(\' \')\n" '
        '> ' + OUT_PATH + 'tflite/__init__.py'
    )

if __name__ == "__main__":

    init_tflite_files()
import os

def init_tflite_files():

    # Absolute path to schema.fbs and the path to store the generated code
    schema_path = '/home/shobhitbehl/ml-model-characterization/TFLite/schema.fbs'
    out_path = '/home/shobhitbehl/ml-model-characterization/TFLite/src/'

    print("Creating tflite files")

    # Compiling schema.fbs into python code
    os.system('flatc --python -o ' + out_path + ' ' + schema_path)

    #Echoing code to __init__.py for corect funtioning of 'from tflite import *'
    os.system(
        'echo "import os.path\n'
        'import glob\n'
        'modules = glob.glob(os.path.join(os.path.dirname(__file__), "*.py"))\n'
        '__all__ = list()\n'
        'for module in modules:\n'
        '   if isfile(module) and not module.endswith(\'__init__.py\'):\n'
        '       __all__.append(basename(module)[:-3]\n"'
        '> ' + out_path + 'tflite/__init__.py'
    )

if __name__ == "__main__":
    init_tflite_files()
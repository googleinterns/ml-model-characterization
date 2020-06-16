
# ML Model Characterization

  

## TF Models (.pb files)

Enter the following commands to run Inferencing <br>

-  `bazel build tf_main`  <br>

-  `bazel-bin/tf_main [--filename FILENAME] [--model_name MODEL_NAME] [--category CATEGORY] [--is_saved_model IS_SAVED_MODEL] [--input_operation_names [INPUT_OPERATION_NAMES]]`  <br>

  

## TFLite Models (.tflite files)

Enter the following commands to run Inferencing <br>

-  `bazel build tflite_init`  <br>

-  `bazel-bin/tflite_init`  <br>

-  `bazel build tflite_main`  <br>

-  `bazel-bin/tflite_main [--filename FILENAME] [--model_name MODEL_NAME] [--category CATEGORY] `  <br>
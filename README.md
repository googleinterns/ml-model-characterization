
# ML Model Characterization

  

## TF Models (.pb files)

Enter the following commands to run Inferencing <br>

-  `bazel build tf_main`  <br>

-  `bazel-bin/tf_main "filepath" "model_name" "category" "is_saved_model"`  <br>

  

## TFLite Models (.tflite files)

Enter the following commands to run Inferencing <br>

-  `bazel build tflite_init`  <br>

-  `bazel-bin/tflite_init`  <br>

-  `bazel build tflite_main`  <br>

-  `bazel-bin/tflite_main "filepath" "model_name" "category" `  <br>

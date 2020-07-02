
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

##  Load models into Spanner DB from scratch

- Download the [data](https://drive.google.com/file/d/1DxtiCvxDLJ950g-2gd__Avu1dgupZLv6/view?usp=sharing)
- Copy the contents of _models_tf_ to _TF/models_ and _models_tflite_ to _TFLite/models_
- In _main<span>.py_ files of _TF/src_ and _TFLite/src_, call the _load_data_ function.
- Run the following commands
	- `chmod +x load_data.sh`
	- `./load_data.sh`
# ML Model Characterization
This repository contains code for parsing TF and TFLite models into a graph structure and loading layer information and tensor information into a database for querying.
Also contains code to generate graph embeddings for further structural learning on the models in the database.

## TF Models (.pb files)

Enter the following commands to run inference or data loading.
To switch between inference and data loading, uncomment the respective function from _TF/src/main.py_.

- `bazel build tf_main`  
- `bazel-bin/tf_main`, this takes the following CLAs,
	- \-\-filename [filename] : Name of the file, must be present in _MODELS_DIR_ of _TF/src/main.py_
	- \-\-model_name [model_name] : Name of the model, must be unique as it is the primary key in the database
	- \-\-category [category] (optional) : Problem category of the model, defaults to "None"
	- \-\-sub_category [sub_category] (optional) : Problem sub-category of the model, defaults to "None"
	- \-\-is_saved_model [is_saved_model] (optional) : "True" to denote the .pb file is in SavedModel format and "False" for FrozenGraph format, defaults to "True".  
	- \-\-input_operation_names [input_operation_names] (optional) : Names of the input operations to the model graph, defaults to [].
	- \-\-is_canonical [is_canonical] (optional) : "True" if the model is canonical i.e. the first of its architecture, else "False", defaults to "False".


## TFLite Models (.tflite files)
Enter the following commands to run inference or data loading.
To switch between inference and data loading, uncomment the respective function from _TFLite/src/main.py_.

- `bazel build tflite_init` 
- `bazel-bin/tflite_init` 
- `bazel build tflite_main` 
- `bazel-bin/tflite_main`, this takes the following CLAs
	- \-\-filename [filename] : Name of the file, must be present in _MODELS_DIR_ of _TFLite/src/main.py_
	- \-\-model_name [model_name] : Name of the model, must be unique as it is the primary key in the database
	- \-\-category [category] (optional) : Problem category of the model, defaults to "None"
	- \-\-sub_category [sub_category] (optional) : Problem sub-category of the model, defaults to "None"
	- \-\-is_canonical [is_canonical] (optional) : "True" if the model is canonical i.e. the first of its architecture, else "False", defaults to "False".

## Loading Data to DB from scratch
Download the data folder from [Google drive](https://drive.google.com/drive/folders/1i6aUbCB0XTEsYXlyxMGpXEv6ydmukzQF?usp=sharing).
The file _load_data.py_ loads models from _models_ directory in _TF/_ and _TFLite/_ with a specific value for _is_canonical_ in the database, hence the data loading will be done in two phases.
### Canonical Models
- Copy the models in **canonical_[file_type].zip** to _[file_type]/models_ directory.
- Set the _IS_CANONICAL_ constant in _load_data,py_ to "True" and run the command `python3 load_data.py`
### Non Canonical Models
- Copy the models in **additional_[file_type].zip** to _[file_type]/models_ directory.
- Set the _IS_CANONICAL_ constant in _load_data,py_ to "False" and run the command `python3 load_data.py`

## Graph Embeddings
To run graph2vec for graph embeddings and printing the _TOP_K_ (defaults to 20) models most similar to every model, run the following commands.
To change the number of models being printed, change _TOP_K_ value in _common/Vectorize.py_.

- `bazel build vectorize`
- `bazel-bin/vectorize`, this takes the following CLAs,
	- \-\-include_edge_attrs [include_edge_attrs] (optional) : "True" if edge attributes are to be included in feature building, else "False", defaults to "False".
	- \-\-include_node_attrs [include_node_attrs] (optional) : "False" if node attributes are to be included in feature building, else "False", defaults to "True".

For further control on which attributes of Node and Edge are to be used, vary the __NODE_ATTRS_ and __EDGE_ATTRS_ in _common/Storage.py_.

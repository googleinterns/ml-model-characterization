# ML Model Characterization
This repository contains code for parsing TF and TFLite models into a graph structure and loading layer information and tensor information into a database for querying.<br>
Also contains code to generate graph embeddings for further structural learning on the models in the database.

## TF Models (.pb files)

Enter the following commands to run inference or data loading. <br>
To switch between inference and data loading, uncomment the respective function from _TF/src/main.py_.

- `bazel build tf_main`  
- `bazel-bin/tf_main`, this takes the following CLAs,
	- \-\-filename [filename] (required) : Name of the file, must be present in _MODELS_DIR_ of _TF/src/main.py_
	- \-\-model_name [model_name] (required) : Name of the model, must be unique as it is the primary key in the database
	- \-\-category [category] (optional) : Problem category of the model, defaults to "None"
	- \-\-sub_category [sub_category] (optional) : Problem sub-category of the model, defaults to "None"
	- \-\-is_saved_model [is_saved_model] (optional) : "True" to denote the .pb file is in SavedModel format and "False" for FrozenGraph format, defaults to "True".  
	- \-\-input_operation_names [input_operation_names] (optional) : Names of the input operations to the model graph, defaults to [].
	- \-\-is_canonical [is_canonical] (optional) : "True" if the model is canonical i.e. the first of its architecture, else "False", defaults to "False".


## TFLite Models (.tflite files)
Enter the following commands to run inference or data loading.<br>
To switch between inference and data loading, uncomment the respective function from _TFLite/src/main.py_.

- `bazel build tflite_init` 
- `bazel-bin/tflite_init` 
- `bazel build tflite_main` 
- `bazel-bin/tflite_main`, this takes the following CLAs
	- \-\-filename [filename] (required) : Name of the file, must be present in _MODELS_DIR_ of _TFLite/src/main.py_
	- \-\-model_name [model_name] (required) : Name of the model, must be unique as it is the primary key in the database
	- \-\-category [category] (optional) : Problem category of the model, defaults to "None"
	- \-\-sub_category [sub_category] (optional) : Problem sub-category of the model, defaults to "None"
	- \-\-is_canonical [is_canonical] (optional) : "True" if the model is canonical i.e. the first of its architecture, else "False", defaults to "False".

## Loading Data to DB from scratch
Download the data folder from [Google drive](https://drive.google.com/drive/folders/1i6aUbCB0XTEsYXlyxMGpXEv6ydmukzQF?usp=sharing).<br>
The file _load_data.py_ loads models from _models_ directory in _TF/_ and _TFLite/_ with a specific value for _is_canonical_ in the database, hence the data loading will be done in two phases.
### Canonical Models
- Copy the models in **canonical_[file_type].zip** to _[file_type]/models_ directory.
- Set the _IS_CANONICAL_ constant in _load_data,py_ to "True" and run the command `python3 load_data.py`
### Non Canonical Models
- Copy the models in **additional_[file_type].zip** to _[file_type]/models_ directory.
- Set the _IS_CANONICAL_ constant in _load_data,py_ to "False" and run the command `python3 load_data.py`

## Graph Embeddings

### Printing similarity within database
To run graph2vec for graph embeddings and printing the _TOPK_ (defaults to 20) models most similar to every model or module, run the following commands. <br>
To change the number of models being printed, change _TOPK_ value in _common/similarity.py_. <br>
To switch between printing for model and modules, uncomment the respective function in _common/similarity.py_. <br>

- `bazel build similarity`
- `bazel-bin/similarity`, this takes the following CLAs,
	- \-\-include_edge_attrs [include_edge_attrs] (optional) : "True" if edge attributes are to be included in feature building, else "False", defaults to "False".
	- \-\-include_node_attrs [include_node_attrs] (optional) : "False" if node attributes are to be included in feature building, else "False", defaults to "True".
	- \-\-wl_iterations [wl_iterations] (optional) : Depth of sub-graph rooted at every node to be considered for feature building in graph2vec, defaults to 3.

### Printing similar models in database to a given input model not present in database
To run graph2vec for graph embeddings and printing the _TOPK_ (defaults to 20) models most similar to an input model/module run the following commands. <br> 

To change the number of models being printed, change _TOPK_ value in _common/compare.py_. <br>

- `bazel build compare`
- `bazel-bin/compare`, this takes the following CLAs,
	- \-\-file_path [file_path] (required) : Full path to the input model/module
	- \-\-file_format [file_format] (required) : File format, one of "TFFrozenGraph", "TFSavedModel" and "TFLite".
	- \-\-model_name [model_name] (optional) : Name of the input model/module, defaults to "input_file"
	- \-\-include_edge_attrs [include_edge_attrs] (optional) : "True" if edge attributes are to be included in feature building, else "False", defaults to "False".
	- \-\-include_node_attrs [include_node_attrs] (optional) : "False" if node attributes are to be included in feature building, else "False", defaults to "True".
	- \-\-wl_iterations [wl_iterations] (optional) : Depth of sub-graph rooted at every node to be considered for feature building in graph2vec, defaults to 3.


For further control on which attributes of Node and Edge are to be used, vary the __NODE_ATTRS_ and __EDGE_ATTRS_ in _common/Storage.py_.

# ML-Workbench

## What is it?
Lorem Ipsum is simply dummy text of the printing and typesetting industry. Lorem Ipsum has been the industry's standard dummy text ever since the 1500s, when an unknown printer took a galley of type and scrambled it to make a type specimen book. It has survived not only five centuries, but also the leap into electronic typesetting, remaining essentially unchanged. It was popularised in the 1960s with the release of Letraset sheets containing Lorem Ipsum passages, and more recently with desktop publishing software like Aldus PageMaker including versions of Lorem Ipsum.

## Who should use it?
It is a long established fact that a reader will be distracted by the readable content of a page when looking at its layout. The point of using Lorem Ipsum is that it has a more-or-less normal distribution of letters, as opposed to using 'Content here, content here', making it look like readable English. Many desktop publishing packages and web page editors now use Lorem Ipsum as their default model text, and a search for 'lorem ipsum' will uncover many web sites still in their infancy. Various versions have evolved over the years, sometimes by accident, sometimes on purpose (injected humour and the like).

## Guiding principles
* **Easy to initiate**: ML workbench will provide a ready-made library and documentation, that can enable even novice users to readily write new applications from scratch. 
* **Highly customizable**: The library will ensure that solutions are highly customizable, as the user can play and experiment with input parameters of APIs. It should enable addition, deletion or modification of intermediate steps.
* **Extensible**: ML Workbench library will allow users to add their own custom libraries, which would comply with the specified guidelines and conventions.
* **Automatically deployable**: The ML workbench will support creation of models and configuration files that can be directly used for deployment in production environment without further human intervention.
* **Scalable**: The ML workbench will enable creation of an end-to-end ML application, which would work on large scales of data, with high performance.
* **Repeatable**: The ML workbench will enable creation of applications, which are robust and consistent, i.e. given an identical dataset, they would produce identical results, without failures.

## Installation

### Requirements
- python 2.7 and pip
- [virtualenv](https://virtualenv.pypa.io/en/stable/installation/)

### Binaries
1. Create virtual environment - `virtualenv run_env`
2. Activate virtual environment - `source run_env/bin/activate`
3. Install - `pip install  ML-Workbench/install/mlworkbench-0.0.1.tar.gz`

### Source
1. Run `bash build.sh` in 'framework' folder
2. Binaries are created in 'ML-Workbench/install' folder
3. Follow the instruction in 'Binaries' section to install binaries

## Getting started

### Running your first example
1. Activate virtual environment in which mlworkbench is installed - `source run_env/bin/activate` - skip if already in the virtual environment
2. Optionally declare a working directory for MLWB  `export MLWB_HOME=some/location/` (default: ~/MLWB_HOME)
3. `mlworkbench run -dag location/example.yaml` to submit the DAG for execution (to run Iris Classification example -  `mlworkbench run -dag examples/Iris_Classification/iris_experiment.yaml`)


### Examples
Examples can be found in 'examples' folder. Detailed explantion can of these examples can be found on wiki.

1. [Iris Classification](https://github.com/ekstep/ML-Workbench/wiki/Tutorial-1:-Iris-Classification)
2. [MovieLens - Item Similarity](https://github.com/ekstep/ML-Workbench/wiki/Tutorial-2:-MovieLens---Item-Item-Similarity)
3. [MovieLens - User Recommendation](https://github.com/ekstep/ML-Workbench/wiki/Tutorial-3:-MovieLens---User-Recommendation)


### Additional notes
+ `localhost:8080` has airflow webserver visualization of the DAG
+ `Ctrl + C` will kill the mlworkbench run process (This will work only after the DAG has been submitted)  
+ Graph inputs, outputs and experiment directory locations are defined relative to the yaml file location




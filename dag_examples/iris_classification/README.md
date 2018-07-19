# Overview
## Objective
This example deals with a  multi-class classification task that involves  identifying 3 different types of iris flowers (Setosa, Versicolour, and Virginica) from their petal and sepal length and width

## Input
* Iris Data: Labeled data in csv format
  +  Schema of Iris Data - `Id | SepalLength | SepalWidth | PetalLength | PetalWidth | Species`
  +  Target to be predicted:  `Species` - Takes in three values - `Setosa; Versicolour; Virginica` 
  +  Number of rows: 150
  +  The data is clean with no missing values or other data quality issues

## Output
* Classification models learned from three different data splits 
* Evaluation reports corresponding to the models


# Process outline

split --> model_logistic_regression

This current example demonstrates a very simple supervised learning pipeline.  Since the data set only has numeric features with no missing values and data quality issues, we can directly use it to learn a classifier.  To ensure that the classification model can generalize to new data, it is typical to learn the model from a subset of data (called the train split) and evaluate it on the rest (called the test split). 

# Implementation
The overall inputs and outputs and each of the data processing steps are discussed in more detail below.
* **Graph Inputs:**   As mentioned earlier, there is a single input which is a csv file containing labeled data with input features and the species label.  The variable “raw_data”  is mapped to the file name
```
raw_data: "inputs/Iris.csv"
```
**Raw Data Structure**
```

Id    | SepalLengthCm | SepalWidthCm | PetalLengthCm | PetalWidthCm | Species        
1     | 5.1           | 3.5          | 1.4           | 0.2          | Iris-setosa    
2     | 4.9           | 3.0          | 1.4           | 0.2          | Iris-setosa
94    | 5.0           | 2.3          | 3.3           | 1.0          | Iris-versicolor
95    | 5.6           | 2.7          | 4.2           | 1.3          | Iris-versicolor
102   | 5.8           | 2.7          | 5.1           | 1.9          | Iris-virginica
103   | 7.1           | 3.0          | 5.9           | 2.1          | Iris-virginica

```

* **Graph Outputs:** Since we are just training a model on Iris data, we get a model file and a model_report with model score.
```
report: "outputs/model_report"
prediction_file_2: "outputs/model"

```

# Node Specifications

* **Step 1: Split data into test and train.**
The first step involves splitting the data frame “raw_data” into train and test dataframes  “[train, test]”.  This task involves invoking the operator “test_train_split”. The  arguments “test_size” determines the ratio of test dataframe when compared to the original dataset.

```
  - node_name: split
    inputs: raw_data
    outputs: [train, test]
    operation: sklearn_lib.Splitters
    arguments:
      model_args:
        name: train_test_split
        arguments:
          test_size: 0.2
    imports:
      - [sklearn.model_selection, train_test_split]

```

* **Step 2: Learn a Logistic Regression Model with Cross validation.**
The third step is the most important one which train a Logistic regression model on the Iris data.  The node carries out cross validation, where the number of splits is determined by `cv_args` in the `arguments` section. the model learnt is a `sklearn.LogisticRegression` model.

```
  - node_name: model_logistic_regression
    inputs: split.train
    outputs: [report, model]
    operation: sklearn_lib.CrossValidate
    arguments:
      model_args:
        name: LogisticRegression
        arguments:
          max_iter: 200
      target_variable: Species
      ignore_variables: Id
      cv_args:
        cv: 5
    imports:
      - [sklearn.linear_model, LogisticRegression]

```


# Outputs

* **Fitted model:** The fitted models are exported as pickle files which could be reused to perform classification on more data.

* **Model report:** The model report file contains the model accuracy metric. For the three models, three model reports are generated.
```
The accuracy of <class 'sklearn.linear_model.logistic.LogisticRegression'> model is 0.94
```

* **Run-logs:** Log files are generated for all the nodes separately in the airflow home folder

```
[2018-03-23 14:10:49,433] {models.py:167} INFO - Filling up the DagBag from /Users/mayankkumar/MLWB_HOME/airflow/dags/airflow_dag.py
[2018-03-23 14:10:51,044] {base_task_runner.py:112} INFO - Running: ['bash', '-c', u'airflow run Iris_Classification split 2018-03-21T00:00:00 --job_id 13 --raw -sd DAGS_FOLDER/airflow_dag.py']
[2018-03-23 14:10:51,394] {base_task_runner.py:95} INFO - Subtask: [2018-03-23 14:10:51,393] {__init__.py:57} INFO - Using executor SequentialExecutor
[2018-03-23 14:10:51,706] {base_task_runner.py:95} INFO - Subtask: [2018-03-23 14:10:51,706] {models.py:167} INFO - Filling up the DagBag from /Users/mayankkumar/MLWB_HOME/airflow/dags/airflow_dag.py
[2018-03-23 14:10:53,325] {base_task_runner.py:95} INFO - Subtask: [2018-03-23 14:10:53,324] {models.py:1126} INFO - Dependencies all met for <TaskInstance: Iris_Classification.split 2018-03-21 00:00:00 [queued]>
[2018-03-23 14:10:53,330] {base_task_runner.py:95} INFO - Subtask: [2018-03-23 14:10:53,330] {models.py:1126} INFO - Dependencies all met for <TaskInstance: Iris_Classification.split 2018-03-21 00:00:00 [queued]>
[2018-03-23 14:10:53,331] {base_task_runner.py:95} INFO - Subtask: [2018-03-23 14:10:53,330] {models.py:1318} INFO - 
[2018-03-23 14:10:53,331] {base_task_runner.py:95} INFO - Subtask: --------------------------------------------------------------------------------
[2018-03-23 14:10:53,331] {base_task_runner.py:95} INFO - Subtask: Starting attempt 1 of 1
[2018-03-23 14:10:53,331] {base_task_runner.py:95} INFO - Subtask: --------------------------------------------------------------------------------
[2018-03-23 14:10:53,331] {base_task_runner.py:95} INFO - Subtask: 
[2018-03-23 14:10:53,338] {base_task_runner.py:95} INFO - Subtask: [2018-03-23 14:10:53,338] {models.py:1342} INFO - Executing <Task(PythonOperator): split> on 2018-03-21 00:00:00
[2018-03-23 14:10:53,519] {base_task_runner.py:95} INFO - Subtask: [2018-03-23 14:10:53,518] {python_operator.py:81} INFO - Done. Returned value was: None
[2018-03-23 14:10:53,526] {base_task_runner.py:95} INFO - Subtask: Closing remaining open files:/Users/mayankkumar/ML-Workbench/examples/Iris_Classification/MLWBdemo1/experiment/read_csv/read_csv.h5...done
[2018-03-23 14:10:56,052] {jobs.py:2083} INFO - Task exited with return code 0

```

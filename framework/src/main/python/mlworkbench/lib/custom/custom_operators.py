from __future__ import division
## THIS CAN BE FURTHER SPLIT UP

import matplotlib
matplotlib.use('Agg')

import findspark

import sys
from collections import defaultdict
from surprise import dump
from surprise import Reader, Dataset, KNNBasic, NormalPredictor, KNNBasic, KNNWithMeans, KNNWithZScore, KNNBaseline, SVD, SVDpp, NMF
from surprise.model_selection import GridSearchCV
import pickle
import pandas as pd
from sklearn.pipeline import Pipeline
import urllib
import zipfile
import os
import shutil
from sklearn.base import TransformerMixin
import matplotlib.pyplot as plt

from mlworkbench.lib.feature_engineering.dataframe_transformers import SeriesToDF, QBinning, DFToSeries
from mlworkbench.lib.feature_engineering.pipeline_builder import get_pipeline
from mlworkbench.lib.operation_definition import NodeOperation


sys.setrecursionlimit(100000)


# to make sure splits are consistent
import random
import numpy as np

my_seed = 0
random.seed(my_seed)
np.random.seed(my_seed)

###

def custom_pyspark_reader(node):
    findspark.init()

    from pyspark.sql import SparkSession
    from pyspark.sql.functions import col, monotonically_increasing_id, from_unixtime
    from pyspark.sql.types import TimestampType

    reader = NodeOperation(node) # initiate
    reader.io_check(len(reader.inputs) == len(reader.outputs)) # check if i/o expectations are met
    file_loc = reader.get_location(reader.inputs[0], 'folder')
    ss_df = dataagg(basefolder=file_loc, **reader.arguments)
    reader.put_dataframes([ss_df], reader.outputs) # pass function to read files

def movielens_data_fetcher(node):
    reader = NodeOperation(node)
    reader.io_check(len(reader.inputs) == len(reader.outputs)) # check if i/o expectations are met
    downloadurl = reader.graph_inputs[reader.inputs[0]]
    print '********************************************* downloadurl: ', downloadurl
    cwd = os.getcwd()
    path_to_zip_file = cwd+"/ml-100k.zip"
    directory_to_extract_to = cwd+"/ml-100k"
    urllib.urlretrieve(downloadurl, path_to_zip_file)
    zip_ref = zipfile.ZipFile(path_to_zip_file, 'r')
    zip_ref.extractall(directory_to_extract_to)
    zip_ref.close()
    sep = reader.arguments['separator']
    data = pd.read_csv(directory_to_extract_to+'/ml-100k/u.data' , sep=sep, header=None, names= ['user', 'item', 'rating', 'timestamp'])
    os.remove(path_to_zip_file)
    shutil.rmtree(directory_to_extract_to)
    reader.put_dataframes([data], reader.outputs) # pass function to read files




## Helper functions
def _convertEpochToDatetime(df, columns, timeunit = 'ms'):
    for c in columns:
        # df[c] = pd.to_datetime(df[c], unit = timeunit)
        df = df.withColumn(c, from_unixtime((col(c)/1000), "yyyy-MM-dd HH:mm:ss").cast(TimestampType()))
    return df

def _import_spark_df_from_json_exported_from_dfa(basefolder):
    spark = SparkSession.builder.appName("Python Spark SQL basic example").config("spark.some.config.option",
                                                                                  "some-value").getOrCreate()
    sparkdf = spark.read.json('file://'+basefolder)
    df_list = {}
    df_list['ss'] = sparkdf
    # objects = list(os.walk(basefolder))[0][1]
    # df_list = {}
    # for fol in objects:
    #     sparkdf = spark.read.json('file://'+basefolder)
    #     df_list[fol] = sparkdf
    return df_list

def dataagg(basefolder, eventstartdate, eventenddate):
    df_list = _import_spark_df_from_json_exported_from_dfa(basefolder)
    ss = df_list['ss'].select('dimensions_gdata_id_ss', 'dimensions_did_ss', 'context_date_range_to_ss',
                          'context_date_range_from_ss', 'syncts_ss', 'ets_ss')
    ss = ss.withColumn('timediff', (col('context_date_range_to_ss') - col('context_date_range_from_ss')) / 1000)
    ss = _convertEpochToDatetime(ss, ['ets_ss', 'syncts_ss', 'context_date_range_from_ss', 'context_date_range_to_ss'])
    ss = ss.where((col("context_date_range_from_ss") >= eventstartdate) & (col("context_date_range_from_ss") <= eventenddate))
    ss = ss.withColumn("session_id", monotonically_increasing_id())
    ss = ss.toPandas()
    for c in ss.columns:
        if(ss[c].dtypes == 'O'):
            ss[c] = ss[c].astype(str)
    return ss

###
def _test_train_split(df, splitcolumn, splitdate):
    traindf = df[(df[splitcolumn] <= splitdate)]
    valdf = df[(df[splitcolumn] > splitdate)]
    return [traindf, valdf]


def custom_splitter(node):
    splitter = NodeOperation(node)  # initiate
    # check if i/o expectations are met
    splitter.io_check((len(splitter.inputs) == 1) & (len(splitter.outputs) == 2))
    data = splitter.get_dataframes(splitter.inputs)[0]  # get input data
    df_list = _test_train_split(data, **splitter.arguments)
    # store output
    splitter.put_dataframes(df_list, splitter.outputs)

###

def independent_column_preprocessing(node):
    icp = NodeOperation(node)
    icp.io_check(len(icp.inputs) == len(icp.outputs))
    data = icp.get_dataframes(icp.inputs)
    icp_pipeline = get_pipeline(**icp.arguments)
    # for i in range(0,len(data)):
    #     df_list.append(icp_pipeline.fit_transform(data[i]))
    df_1 = icp_pipeline.fit_transform(data[0])
    df_2 = icp_pipeline.transform(data[1])
    df_list = [df_1, df_2]
    icp.put_dataframes(df_list,icp.outputs)


###

class AddBias(TransformerMixin):

    def __init__(self, c=4.5):
        self.c = c

    def fit(self, X, y=None):
        # assumes X is a DataFrame
        self.avg_rating = X.values.mean()
        return self

    def transform(self, X):
        # assumes X is a DataFrame
        Xc = X + (self.c - self.avg_rating)
        return Xc

def custom_column_preprocessing(df_list, max_rating=10):
    binning = QBinning(q= max_rating,labels=False, duplicates='drop')
    seriesdf = SeriesToDF()
    dfseries = DFToSeries()
    bias = AddBias()
    cpp_pipeline = Pipeline([("SeriesToDF",seriesdf),("QBinning",binning), ('AddBias', bias),("DFtoSeries",dfseries)])
    training_df = df_list[0]
    validation_df = df_list[1]
    training_df['target'] = training_df.groupby(['dimensions_gdata_id_ss'])['timediff'].transform(cpp_pipeline.fit_transform)
    validation_df['target'] = validation_df.groupby(['dimensions_gdata_id_ss'])['timediff'].transform(cpp_pipeline.transform)
    training_df = training_df.drop(columns=['timediff'])
    validation_df = validation_df.drop(columns=['timediff'])
    training_df['target'].fillna(0, inplace=True)
    validation_df['target'].fillna(0, inplace=True)
    final_df_train = training_df.groupby(['dimensions_did_ss','dimensions_gdata_id_ss'],as_index=False).mean()
    final_df_val = validation_df.groupby(['dimensions_did_ss','dimensions_gdata_id_ss'],as_index=False).mean()

    return [final_df_train, final_df_val]

def custom_preprocessing(node):
    cp = NodeOperation(node)
    cp.io_check((len(cp.inputs) == 2) & (len(cp.outputs) == 2))
    data = cp.get_dataframes(cp.inputs)
    df_list = custom_column_preprocessing(data, max_rating=cp.arguments['max_rating'])
    cp.put_dataframes(df_list,cp.outputs)


###

surprise_models = {'NormalPredictor': NormalPredictor,
                   'KNNBasic': KNNBasic,
                   'KNNWithMeans': KNNWithMeans,
                   'KNNWithZScore': KNNWithZScore,
                   'KNNBaseline': KNNBaseline,
                   'SVD': SVD,
                   'SVDpp': SVDpp,
                   'NMF': NMF
                   }


def convert_all_columns_to_string(df):
    for c in df.columns:
        if (df[c].dtypes == 'O'):
            df[c] = df[c].astype(str)
    return df



def KNNBasic_model_trainer(node):
    model = NodeOperation(node)
    model.io_check((len(model.inputs) == 1) & (len(model.outputs) == 2))
    data = model.get_dataframes(model.inputs)

    rating_scale = (data[0].iloc[:,2].min(), data[0].iloc[:,2].max())
    reader = Reader(rating_scale=rating_scale)
    training_data = Dataset.load_from_df(data[0].iloc[:,0:3], reader)
    trainingSet = training_data.build_full_trainset()
    testSet = trainingSet.build_anti_testset()
    modeltype = model.arguments['algo_class']
    best_model_selection_metric = model.arguments['best_model_selection_metric']
    model_args = dict(model.arguments.copy())
    del model_args['algo_class']
    del model_args['best_model_selection_metric']
    gs = GridSearchCV(algo_class=surprise_models[modeltype], **model_args)

    gs.fit(training_data)
    
    model_1 = gs.best_estimator[best_model_selection_metric]
    model_1.fit(trainingSet)

    model_output_file_1 = model.get_location(model.outputs[0], 'file')
    dump.dump(file_name=model_output_file_1, algo=model_1)
    
    gs_report = convert_all_columns_to_string(pd.DataFrame(gs.cv_results))
    model.put_dataframes([gs_report], [model.outputs[1]])


def get_top_n(predictions, n=10):
    '''Return the top-N recommendation for each user from a set of predictions.
    Args:
        predictions(list of Prediction objects): The list of predictions, as
            returned by the test method of an algorithm.
        n(int): The number of recommendation to output for each user. Default
            is 10.
    Returns:
    A dict where keys are user (raw) ids and values are lists of tuples:
        [(raw item id, rating estimation), ...] of size n.
    '''

    # First map the predictions to each user.
    top_n = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        top_n[uid].append((iid, est))

    # Then sort the predictions for each user and retrieve the k highest ones.
    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_n[uid] = user_ratings[:n]

    return top_n


def convert_dict_to_df(inputdict, keycolumn, othercolumns):
    initdf = pd.DataFrame(inputdict)
    all_cols = [keycolumn] + othercolumns
    finaldf = pd.DataFrame(columns=all_cols)
    tempdf = pd.DataFrame()
    for c in initdf.columns:
        tempdf[othercolumns] = initdf[c].apply(pd.Series)
        tempdf[keycolumn] = c
        finaldf = finaldf.append(tempdf, ignore_index=True)
    return finaldf


def convert_object_columns_to_string(df):
    for c in df.columns:
        if (df[c].dtypes == 'O'):
            df[c] = df[c].astype(str)
    return df



###

def spark_data_prep(node):
    #construct string for bash operator
    dataprep = NodeOperation(node)
    dataprep.io_check((len(dataprep.inputs)==0) & (len(dataprep.outputs)==1))
    output_folder = dataprep.get_location(dataprep.outputs[0],'folder')
    configuration = dataprep.arguments['configuration']
    classpath = dataprep.arguments['classpath']
    jar_location = dataprep.arguments['jar_location']
    output_format = dataprep.arguments['output_format']

    bash_script = """$SPARK_HOME/bin/spark-submit --class {} '{}' "{}" "{}" {}""".format(classpath,jar_location
                                                                                         ,configuration
                                                                                         ,output_folder,output_format)
    return  bash_script



### Used for Item Item Similarity

def compute_item_item_similarity(node):
    sim_compute = NodeOperation(node)
    sim_compute.io_check(
        (len(sim_compute.inputs) == 1) & (len(sim_compute.outputs) == 1))  # check if i/o expectations are met
    data = sim_compute.get_dataframes(sim_compute.inputs)[0]

    ## Training the knn model
    rating_scale = (data['rating'].min(), data['rating'].max())
    reader = Reader(rating_scale=rating_scale)
    training_data = Dataset.load_from_df(data.iloc[:, 0:3], reader)  # column number to be used
    trainingSet = training_data.build_full_trainset()
    knn = KNNBasic(sim_options=sim_compute.arguments['sim_options'])
    knn.fit(trainingSet)

    sim_matrix = knn.compute_similarities()
    rawids = map(knn.trainset.to_raw_iid, knn.trainset.all_items())
    final_dict = {'matrix': sim_matrix, 'indices': rawids}

    file_loc = sim_compute.get_location(sim_compute.outputs[0], 'file')
    open(file_loc, "wb")
    pickle.dump(file=open(file_loc, "wb"), obj=final_dict)



def top_k_neighbors_computation(node):
    compute_neighbors = NodeOperation(node)
    compute_neighbors.io_check((len(compute_neighbors.inputs) == 1) & (
                len(compute_neighbors.outputs) == 1))  # check if i/o expectations are met
    file_location = compute_neighbors.get_location(compute_neighbors.inputs[0], 'file')
    topK = compute_neighbors.arguments['topK']
    # knn = dump.load(model_location)[1]
    input_dict = pickle.load(file=open(file_location, "rb"))

    sim_mat = input_dict['matrix']
    indices = input_dict['indices']

    sim_df = pd.DataFrame(sim_mat, columns=indices, index=indices)

    for index, row in sim_df.iterrows():
        item_in_focus = index
        sim_items = row.sort_values(ascending=False).index[0:topK]
        temp_df = pd.DataFrame(sim_items).T
        temp_df.index = [item_in_focus]
        try:
            predictions
        except NameError:
            predictions = temp_df.copy()
        else:
            predictions = predictions.append(temp_df)

    #     sim_mat = input_dict
    #     rawids = map(knn.trainset.to_raw_iid, knn.trainset.all_items())
    #     iids = knn.trainset.all_items()
    #     iids_pred = [knn.get_neighbors(x, k=compute_neighbors.arguments['topK']) for x in iids]
    #     pred = [map(knn.trainset.to_raw_iid, x) for x in iids_pred]
    #     predictions = pd.DataFrame(pred)
    #     predictions.index = rawids
    compute_neighbors.put_dataframes([predictions], compute_neighbors.outputs)



def item_similarity_model_evaluation(node):
    evaluate = NodeOperation(node)
    evaluate.io_check((len(evaluate.inputs) == 2) & (len(evaluate.outputs) == 1))  # check if i/o expectations are met
    predictions = evaluate.get_dataframes([evaluate.inputs[0]])[0]

    val_set = evaluate.get_dataframes([evaluate.inputs[1]])[0]
    report_location = evaluate.get_location(evaluate.outputs[0], 'file')

    K = len(predictions.columns)
    evaldf = pd.DataFrame(columns=['k', 'precision_at_k'])
    for k in range(K):
        hits = 0
        total = 0
        for i in predictions.index:
            if (i in val_set.index):
                pred = list(predictions.loc[i])[0:k + 1]
                val = list(val_set.loc[i].dropna())[0:k + 1]  ## add condition to check if the length of valida
                hits += len(list(set(pred).intersection(val)))
                total += len(pred)
        precision = float(hits / total) * 100
        evaldf = evaldf.append(pd.DataFrame([(k + 1, precision)], columns=['k', 'precision_at_k']), ignore_index=True)
    plt.plot(evaldf['k'], evaldf['precision_at_k'])
    plt.xlabel('K value')
    plt.ylabel('Precision %')
    plt.title('Similarity evaluation: K vs Precision')
    plt.savefig(report_location + '.png')



### for MovieLens Example

def model_predictor(node):
    predict = NodeOperation(node)
    predict.io_check((len(predict.inputs) == 2) & (len(predict.outputs) == 1))
    model_location = predict.get_location(predict.inputs[0], 'file')
    prediction_set = predict.get_dataframes([predict.inputs[1]])[0
    ]
    knn = dump.load(model_location)[1]
    raw_testset = knn.trainset.build_anti_testset()
    testdf = pd.DataFrame(raw_testset, columns=['users', 'items', 'rating'])

    iids = map(knn.trainset.to_inner_iid, prediction_set.iloc[:, 0])
    testsetdf = testdf[testdf['users'].isin(iids)]
    testset = [tuple(x) for x in testsetdf.values]

    predictions = knn.test(testset)

    top_10_predictions = get_top_n(predictions, n=predict.arguments['topK'])
    df = convert_dict_to_df(top_10_predictions, keycolumn='user',
                            othercolumns=['item', 'rating'])
    df_list = [convert_object_columns_to_string(df)]

    predict.put_dataframes(df_list, predict.outputs)



def model_evaluation(node):
    evaluation = NodeOperation(node)
    evaluation.io_check((len(evaluation.inputs) == 2) & (len(evaluation.outputs) == 1))
    file_location = evaluation.get_location(evaluation.outputs[0], 'file')
    sys.stdout = open(file_location, 'w')
    model_location = evaluation.get_location(evaluation.inputs[0], 'file')
    validation_data = evaluation.get_dataframes([evaluation.inputs[1]])[0]
    knn = dump.load(model_location)[1]
    iids = knn.trainset.all_items()
    rawiids = map(knn.trainset.to_raw_iid, iids)
    uids = knn.trainset.all_users()
    rawuids = map(knn.trainset.to_raw_uid, uids)
    actual_validation_data = validation_data[validation_data.iloc[:,0].isin(rawuids)&validation_data.iloc[:,1].isin(rawiids)]
    validation_set = [tuple(x) for x in actual_validation_data.values]
    predictions = [knn.predict(iid=x[1], uid=x[0], r_ui=x[2]) for x in validation_set]
    predicted_df = pd.DataFrame(predictions)
    print "Mean Absolute Error: ", sum(abs(predicted_df['r_ui']-predicted_df['est']))/(predicted_df.shape[0])

    print "Root Mean Squared Error: ",sum((predicted_df['r_ui']-predicted_df['est'])**2)/(predicted_df.shape[0])



## Keras Models





## Node that creates the user and item indices - Converts the list of user ids and item ids to indices from 0 to n
def add_user_item_index(node):
    index_node = NodeOperation(node)
    index_node.io_check(
        (len(index_node.inputs) == 2) & (len(index_node.outputs) == 2))  # check if i/o expectations are met
    data = index_node.get_dataframes(index_node.inputs)

    training_data = data[0]
    validation_data = data[1]

    training_data['user_emb'] = pd.Categorical(training_data.iloc[:, 0]).codes
    training_data['item_emb'] = pd.Categorical(training_data.iloc[:, 1]).codes

    validation_data = pd.merge(validation_data, training_data[[training_data.columns[0], 'user_emb']].drop_duplicates(),
                               on=[training_data.columns[0]], how='left')
    validation_data = pd.merge(validation_data, training_data[[training_data.columns[1], 'item_emb']].drop_duplicates(),
                               on=[training_data.columns[1]], how='left').dropna()

    training_data = training_data[['user_emb', 'item_emb', 'rating', 'user', 'item']]
    validation_data = validation_data[['user_emb', 'item_emb', 'rating', 'user', 'item']]
    df_list = [training_data, validation_data]

    index_node.put_dataframes(df_list, index_node.outputs)



def CF_keras_model_fitting(node):
    from keras.models import Sequential, model_from_json
    from keras.wrappers.scikit_learn import KerasClassifier
    from keras.layers import Dense, Dropout, Activation, Embedding, Reshape, Merge
    from keras.utils import np_utils
    from sklearn.preprocessing import LabelEncoder
    model_fitting = NodeOperation(node)
    model_fitting.io_check((len(model_fitting.inputs) == 1) & (len(model_fitting.outputs) == 2))

    training_data = model_fitting.get_dataframes(model_fitting.inputs)[0]

    Users = training_data.iloc[:, 0].values
    Items = training_data.iloc[:, 1].values
    Ratings = training_data.iloc[:, 2]
    n_users = len(training_data.iloc[:, 0].drop_duplicates().values)
    m_items = len(training_data.iloc[:, 1].drop_duplicates().values)
    k_factors = model_fitting.arguments['k_factors']

    user_nw = Sequential()
    user_nw.add(Embedding(n_users, k_factors, input_length=1))
    user_nw.add(Reshape((k_factors,)))

    item_nw = Sequential()
    item_nw.add(Embedding(m_items, k_factors, input_length=1))
    item_nw.add(Reshape((k_factors,)))

    merged = Merge([user_nw, item_nw], mode='dot', dot_axes=1)

    CF_model = Sequential()
    CF_model.add(merged)
    CF_model.compile(loss=model_fitting.arguments['loss'], optimizer=model_fitting.arguments['optimizer'],
                     metrics=model_fitting.arguments['metrics'])

    CF_model.fit([Users, Items], Ratings, epochs=5, verbose=2)

    model_location = model_fitting.get_location(model_fitting.outputs[0], 'file')
    model_weights_location = model_fitting.get_location(model_fitting.outputs[1], 'file')

    model_json = CF_model.to_json()
    with open(model_location, "w") as json_file:
        json_file.write(model_json)

    CF_model.save_weights(filepath=model_weights_location)

## Node used to evaluate a keras model
def evaluate_keras_model(node):
    from keras.models import Sequential, model_from_json
    from keras.wrappers.scikit_learn import KerasClassifier
    from keras.layers import Dense, Dropout, Activation, Embedding, Reshape, Merge
    from keras.utils import np_utils
    from sklearn.preprocessing import LabelEncoder
    model_evaluation = NodeOperation(node)
    model_evaluation.io_check((len(model_evaluation.inputs) == 3) & (len(model_evaluation.outputs) == 1))

    validation_data = model_evaluation.get_dataframes([model_evaluation.inputs[2]])[0]
    file_location = model_evaluation.get_location(model_evaluation.outputs[0], 'file')

    model_path = model_evaluation.get_location(model_evaluation.inputs[0], 'file')
    model_weights_path = model_evaluation.get_location(model_evaluation.inputs[1], 'file')

    json_file = open(model_path, 'r')
    loaded_model_json = json_file.read()
    json_file.close()

    model = model_from_json(loaded_model_json)
    # load weights into new model
    model.load_weights(model_weights_path)

    model.compile(loss='mse', optimizer='adam', metrics=['mae'])

    Users_for_pred = validation_data['user_emb']
    Items_for_pred = validation_data['item_emb']

    validation_data['pred_rating'] = (model.predict([Users_for_pred, Items_for_pred]))
    pred_rating_list = [tuple(x) for x in validation_data[['rating', 'pred_rating']].values]

    rmse = sum([(a - b) ** 2 for (a, b) in pred_rating_list]) / len(pred_rating_list)
    mae = sum([abs(a - b) for (a, b) in pred_rating_list]) / len(pred_rating_list)

    output = "Model RMSE: " + str(rmse) + "\n" + "Model MAE: " + str(mae)

    with open(file_location, 'w') as outputfile:
        outputfile.write(output)

## Predict using keras model
def predict_keras_model(node):
    from keras.models import Sequential, model_from_json
    from keras.wrappers.scikit_learn import KerasClassifier
    from keras.layers import Dense, Dropout, Activation, Embedding, Reshape, Merge
    from keras.utils import np_utils
    from sklearn.preprocessing import LabelEncoder
    model_prediction = NodeOperation(node)
    model_prediction.io_check((len(model_prediction.inputs) == 4) & (len(model_prediction.outputs) == 1))

    df_list = model_prediction.get_dataframes(model_prediction.inputs[0:2])
    model_location = model_prediction.get_location(model_prediction.inputs[2], 'file')
    model_weights_location = model_prediction.get_location(model_prediction.inputs[3], 'file')
    training_data = df_list[0]
    prediction_set = df_list[1]

    pred_set = prediction_set.iloc[:, 0].drop_duplicates().values
    pred_set_encoded = training_data[training_data['user'].isin(pred_set)]['user_emb'].drop_duplicates().values

    all_items = training_data['item_emb'].drop_duplicates().values
    prediction_df = pd.DataFrame([(x, y) for x in pred_set_encoded for y in all_items],
                                 columns=['user_emb', 'item_emb'])
    prediction_df = prediction_df[~prediction_df.isin(training_data[['user_emb', 'item_emb']])].dropna()

    Users_for_pred = prediction_df['user_emb'].values
    Items_for_pred = prediction_df['item_emb'].values

    json_file = open(model_location, 'r')
    loaded_model_json = json_file.read()
    json_file.close()

    model = model_from_json(json_string=loaded_model_json)

    predictions = model.predict([Users_for_pred, Items_for_pred])
    predictions_list = pd.DataFrame(predictions).iloc[:, 0].values

    predicted_df = pd.DataFrame(
        {'user_emb': Users_for_pred, 'item_emb': Items_for_pred, 'predictions': predictions_list})
    predicted_df = predicted_df.sort_values(['user_emb', 'predictions'], ascending=[True, False])
    predicted_df_topK = predicted_df.groupby('user_emb').head(model_prediction.arguments['topK']).reset_index(drop=True)
    predicted_df_topK['idx'] = predicted_df_topK.groupby('user_emb').cumcount()
    final_df = predicted_df_topK[['user_emb', 'idx', 'item_emb']].pivot(index='user_emb', columns='idx')[['item_emb']]
    final_df['user_emb'] = final_df.index
    final_df.columns = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'user_emb']

    output = pd.merge(final_df, training_data[['user', 'user_emb']].drop_duplicates(), on=['user_emb'],
                      how='left').drop('user_emb', 1)

    model_prediction.put_dataframes([output], model_prediction.outputs)


## Model generation using keras and sklearn
def baseline_model_svm():
    from keras.models import Sequential, model_from_json
    from keras.wrappers.scikit_learn import KerasClassifier
    from keras.layers import Dense, Dropout, Activation, Embedding, Reshape, Merge
    from keras.utils import np_utils
    from sklearn.preprocessing import LabelEncoder
    # create model
    model = Sequential()
    model.add(Dense(8, input_dim=4, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(16, input_dim=8, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(3, activation='softmax'))
    # Compile model
    # model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.compile(loss='categorical_hinge', optimizer='adam', metrics=['accuracy'])
    return model

## Train a keras svm model
def train_keras_svm_model(node):
    from keras.models import Sequential, model_from_json
    from keras.wrappers.scikit_learn import KerasClassifier
    from keras.layers import Dense, Dropout, Activation, Embedding, Reshape, Merge
    from keras.utils import np_utils
    from sklearn.preprocessing import LabelEncoder
    model_training = NodeOperation(node)
    model_training.io_check(
        (len(model_training.inputs) == 2) & (len(model_training.outputs) == 3))  # check if i/o expectations are met

    df_list = model_training.get_dataframes(model_training.inputs)

    trainset = df_list[0]
    testset = df_list[1]

    estimator = KerasClassifier(build_fn=baseline_model_svm, epochs=200, batch_size=5, verbose=0)

    X = trainset.values[:, 1:5].astype(float)
    Y = trainset.values[:, 5]

    # encode class values as integers
    encoder = LabelEncoder()
    encoder.fit(Y)
    encoded_Y = encoder.transform(Y)
    # convert integers to dummy variables (i.e. one hot encoded)
    dummy_y = np_utils.to_categorical(encoded_Y)

    X_test = testset.values[:, 1:5].astype(float)
    Y_test = testset.values[:, 5]

    # encode class values as integers
    encoded_Y_test = encoder.transform(Y_test)
    # convert integers to dummy variables (i.e. one hot encoded)
    dummy_y_test = np_utils.to_categorical(encoded_Y_test)

    estimator.fit(x=X, y=dummy_y)

    model_score = estimator.score(x=X_test, y=dummy_y_test)

    pred = estimator.predict(x=X_test)
    pred_df = pd.DataFrame(encoder.inverse_transform(pred), columns=['predictions'])

    model_location = model_training.get_location(model_training.outputs[0], 'file')
    model_report_location = model_training.get_location(model_training.outputs[2], 'file')
    model_training.put_dataframes([pred_df], [model_training.outputs[1]])
    output = 'The accuracy of ' + model_training.node_name + ' model is: ' + str(model_score)
    with open(model_report_location, 'w') as outputfile:
        outputfile.write(output)

    model_file = open(model_location, 'wb')
    pickle.dump(estimator, model_file)



# -*- coding: utf-8 -*-
#
# Copyright (C) 2016  EkStep Foundation
# 
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as
# published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
# 
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

from custom_operators import custom_pyspark_reader, custom_splitter, independent_column_preprocessing,custom_preprocessing,\
    KNNBasic_model_trainer, compute_item_item_similarity, top_k_neighbors_computation, item_similarity_model_evaluation, \
    model_evaluation, model_predictor, add_user_item_index, CF_keras_model_fitting, evaluate_keras_model, \
    spark_data_prep, movielens_data_fetcher, predict_keras_model, train_keras_svm_model

from nlp_custom_operators import read_text_file_corpus, compute_doc_similarity


custom_python_callables = {
    'custom_pyspark_reader': custom_pyspark_reader,
    'movielens_data_fetcher': movielens_data_fetcher,
    'custom_splitter': custom_splitter,
    'independent_column_preprocessing': independent_column_preprocessing,
    'custom_preprocessing': custom_preprocessing,
    'KNNBasic_model_trainer': KNNBasic_model_trainer,
    'compute_item_item_similarity': compute_item_item_similarity,
    'top_k_neighbors_computation': top_k_neighbors_computation,
    'item_similarity_model_evaluation': item_similarity_model_evaluation,
    'model_evaluation': model_evaluation,
    'model_predictor': model_predictor,
    'add_user_item_index': add_user_item_index,
    'CF_keras_model_fitting': CF_keras_model_fitting,
    'evaluate_keras_model': evaluate_keras_model,
    'predict_keras_model': predict_keras_model,
    'train_keras_svm_model': train_keras_svm_model,
    'read_text_file_corpus': read_text_file_corpus,
    'compute_doc_similarity': compute_doc_similarity
}

custom_bash_callables = {
    'spark_data_prep': spark_data_prep
}
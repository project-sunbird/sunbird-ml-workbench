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

from mlworkbench.utils.common import dicts_union
from mlworkbench.lib.etl.readers import csv_reader
from mlworkbench.lib.etl.writers import csv_writer
from mlworkbench.lib.etl.splitters import k_fold_splitter, df_splitter_on_column_values
from mlworkbench.lib.models.scikit import logistic_model, svm_model, decision_tree
from mlworkbench.lib.custom.custom_operators_registry import custom_python_callables, custom_bash_callables


python_callables = {
    'csv_reader': csv_reader,
    'csv_writer' : csv_writer,
    'k_fold_splitter': k_fold_splitter,
    'df_splitter_on_column_values': df_splitter_on_column_values,
    'logistic_model': logistic_model,
    'svm_model': svm_model,
    'decision_tree': decision_tree
}

bash_callables = {
}

python_callables = dicts_union(python_callables,custom_python_callables)
bash_callables = dicts_union(bash_callables, custom_bash_callables)

callables = dicts_union(python_callables, bash_callables)

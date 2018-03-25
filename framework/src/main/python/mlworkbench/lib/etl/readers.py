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

import pandas as pd
from mlworkbench.lib.operation_definition import NodeOperation


def csv_reader(node):
    reader = NodeOperation(node) # initiate
    reader.io_check(len(reader.inputs) == len(reader.outputs)) # check if i/o expectations are met
    reader.read_files_to_dataframe_storage(pd.read_csv) # pass function to read files
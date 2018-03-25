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

from mlworkbench.lib.operation_definition import NodeOperation
from mlworkbench.utils.common import create_directory, get_parent_dir

def csv_writer(node):
    writer = NodeOperation(node) # initiate
    writer.io_check(len(writer.inputs) == len(writer.outputs)) # check if i/o expectations are met
    df_list, output_loc_list = writer.get_dataframe_objects_to_write()

    for i in range(0,len(df_list)):
        create_directory(get_parent_dir(output_loc_list[i]))
        df_list[i].to_csv(output_loc_list[i],**writer.arguments) # write to file
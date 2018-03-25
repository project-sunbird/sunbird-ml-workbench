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

import json
from collections import OrderedDict
from json import JSONDecoder


def _validate_config(learningConfig):
    # Has all the keys
    top_keys = ['input_data_location', 'expt_id',
                'runs', 'output_dir', 'input_data_format']
    run_keys = ['splitter_test_size', 'independent_column_preprocessing_pipeline_filename',
                'run_id', 'hp_search_config_filename', 'custom_preprocessing_pipeline_filename',
                'output_model_name', 'feature_set_specific_learning_config_filename', "n_folds"]
    input_formats = ['csv']

    if (set(top_keys) == set(learningConfig.keys())):

        if (learningConfig["input_data_format"] not in set(input_formats)):
            # Not supported format
            raise ValueError("{input_format} is currently not supported as an input_data_format.".format(
                input_format=repr(learningConfig["input_data_format"])))

        for i in range(len(learningConfig["runs"])):
            if (set(run_keys) != set(learningConfig["runs"][i].keys())):
                # Error with format
                raise ValueError(
                    "Configuration file is not in the expected format. Check the run no. {i}.".format(i=repr(i + 1)))
    else:
        # Error with format
        raise ValueError(
            "Configuration file is not in the expected format. Check the top level keys.")


def parseConfig(learning_config_filename):
    with open(learning_config_filename) as data_file:
        learningConfig = json.load(data_file)
        # Validate structure of json
    _validate_config(learningConfig)
    return learningConfig


def _make_unique(key, dct):
    counter = 0
    unique_key = key

    while unique_key in dct:
        counter += 1
        unique_key = '{}_{}'.format(key, counter)
    return unique_key


def _parse_object_pairs(pairs):
    dct = OrderedDict()
    for key, value in pairs:
        if key in dct:
            key = _make_unique(key, dct)
        dct[key] = value

    return dct


# Public functions
def parseJSON(json_loc):
    # Create OrderedDict
    data = open(json_loc).read()
    decoder = JSONDecoder(object_pairs_hook=_parse_object_pairs)
    orderedDict = decoder.decode(data)

    return orderedDict  # Dictionary of feature-wise tasks


def script_to_pipeline(ccp_loc):
    execfile(ccp_loc, globals())
    return custom_pipeline  # THERE MUST BE A BETTER WAY!
    # Need to check if the file is a valid pipeline

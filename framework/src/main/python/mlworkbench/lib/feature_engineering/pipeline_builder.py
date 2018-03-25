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

from collections import OrderedDict
import warnings
from json import JSONDecoder
from sklearn.pipeline import Pipeline

from mlworkbench.utils.parsers import parseJSON
from mlworkbench.lib.feature_engineering.dataframe_transformers import *
from transformer_registry import TaskReference


def get_pipeline(json_loc):
    # Create a dict with datatype and tasklist
    orderedDict = parseJSON(json_loc)
    finalDict = {}
    for value in orderedDict.values()[0]:
        if value['TaskName'] != 'None':
            finalDict.setdefault(value['FeatureName']
                                 , [value['Type'], ]).append((value['TaskName'], value['Options'].replace('\n', '')))
        else:
            finalDict.setdefault(value['FeatureName'], [value['Type']])

    # Build pipeline

    featureList = []
    for feature, tasks in finalDict.iteritems():

        taskList = []

        # Add column selection
        operation = "Select~Column"
        options = "cols = \"" + feature + "\""
        try:
            taskName = TaskReference[operation]
            taskList.append("(\"{0}\", {1}({2}))".format(operation, taskName, options))
        except:
            warnings.warn("Skipped \"%s\" - %s" % (operation, feature), RuntimeWarning)

        if tasks[0] == "Numeric":
            # Add Numeric Cast
            operation = "Numeric~~Cast"
            options = ""
            try:
                taskName = TaskReference[operation]
                taskList.append("(\"{0}\", {1}({2}))".format(operation, taskName, options))
            except:
                warnings.warn("Skipped \"%s\" - %s" % (operation, feature), RuntimeWarning)
            tasks.pop(0)
            for task in tasks:
                operation = "Numeric~~" + task[0].replace(" ", "~")
                options = task[1]
                try:
                    taskName = TaskReference[operation]
                    taskList.append("(\"{0}\", {1}({2}))".format(operation, taskName, options))
                except:
                    warnings.warn("Skipped \"%s\" - %s" % (operation, feature), RuntimeWarning)

            pipe = eval(", ".join(taskList).join(("(\"%s\", Pipeline([" % feature, "]))")))
            featureList.append(pipe)

        elif tasks[0] == "String":
            # Add String Cast
            operation = "String~~Cast"
            options = ""
            try:
                taskName = TaskReference[operation]
                taskList.append("(\"{0}\", {1}({2}))".format(operation, taskName, options))
            except:
                warnings.warn("Skipped \"%s\" - %s" % (operation, feature), RuntimeWarning)
            tasks.pop(0)
            for task in tasks:
                operation = "String~~" + task[0].replace(" ", "~")
                options = task[1]
                try:
                    taskName = TaskReference[operation]
                    taskList.append("(\"{0}\", {1}({2}))".format(operation, taskName, options))
                except:
                    warnings.warn("Skipped \"%s\" - %s" % (operation, feature), RuntimeWarning)

            pipe = eval(("Pipeline([" + ", ".join(taskList) + "])"))
            pipe = (feature, pipe)
            featureList.append(pipe)

    pipeline = Pipeline([("features", DFFeatureUnion(featureList))])
    return pipeline
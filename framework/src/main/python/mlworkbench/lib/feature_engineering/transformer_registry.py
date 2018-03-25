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

TaskReference= {"Select~Column":"ColumnExtractor",
"Numeric~~Cast":"NumericCast",
"Numeric~~Missing~Imputation":"DFMissingNum",
"Numeric~~Clip":"DFClip",
"Numeric~~Find~and~Replace":"DFReplace",
"Numeric~~Unit~Conversion":"UnitConv",
"Numeric~~Standardization":"DFStandardScaler",
"Numeric~~Binning":"Binning",
"Numeric~~Quantile~Binning":"QBinning",
"String~~Cast":"StringCast",
"String~~Missing~Imputation":"DFMissingStr",
"String~~Find~and~Replace":"DFReplace",
"String~~One~hot~encoding":"DFOneHot"}

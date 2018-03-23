## THIS CAN BE FURTHER SPLIT UP

from sklearn.base import TransformerMixin
import pandas as pd
import numpy as np
from sklearn.preprocessing.data import Normalizer
from . import Q_
from sklearn.preprocessing import Imputer
from math import ceil, floor
from __builtin__ import dict


class NumericCast(TransformerMixin):
    '''
    Implementation of pandas friendly numeric cast
    '''

    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_cast = X.apply(lambda x: pd.to_numeric(x, errors='coerce'))
        return X_cast


class StringCast(TransformerMixin):
    '''
    Implementation of pandas friendly string cast
    '''

    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        Xz = X.astype(str)
        Xz = Xz.replace("nan", np.NaN)
        return Xz


class DFReplace(TransformerMixin):
    '''
    Refer https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.replace.html
    '''

    def __init__(self, to_replace=None, value=None
                 #                  , inplace=False
                 #                  , limit=None
                 , regex=False
                 #                  , method='pad'
                 ):
        self.to_replace = to_replace
        self.value = value
        self.inplace = False
        #         self.limit = limit
        self.regex = regex
        self.method = 'pad'

    def fit(self, X, y=None):
        # Do Nothing
        return self

    def transform(self, X):
        X_replaced = X.replace(to_replace=self.to_replace, value=self.value, inplace=self.inplace,
                               #                                limit=self.limit,
                               regex=self.regex, method=self.method)

        return X_replaced


def _Impute(value, S):
    return {
        'mean': S.mean(),
        'median': S.median(),
        'most_frequent': S.mode()[0]
    }[value]


class DFMissingNum(TransformerMixin):
    '''
    Replaces missing values by input value or method.Below are the methods available.
    'mean': replace missing values using the mean.
    'median': replace missing values using the median
    'most_frequent': replace missing values using the mode
    'backfill' or 'bfill': use NEXT valid observation to fill gap.
    'pad' or 'ffill': propagate last valid observation forward to next valid.
    Numeric value: Replaces with the input value
    Ex: repalce = ""mean"" for replacing with mean, replace = 0 for replacing with the numeric 0
    Note: No quotes for numeric values
    '''

    def __init__(self, replace):
        self.replace = replace
        self.imp = None
        self.statistics_ = None

    def fit(self, X, y=None):

        if type(self.replace) == dict:
            for key, value in self.replace.iteritems():
                if value in ['mean', 'median', 'most_frequent']:
                    self.replace[key] = _Impute(value=value, S=X[key])

        elif self.replace in ['mean', 'median', 'most_frequent']:
            self.imp = DFImputer(strategy=self.replace)
            self.imp.fit(X)
            self.statistics_ = pd.Series(self.imp.statistics_, index=X.columns)
        return self

    def transform(self, X):
        if self.replace in ['mean', 'median', 'most_frequent']:
            Ximp = self.imp.transform(X)
            X_replaced = pd.DataFrame(Ximp, index=X.index, columns=X.columns)

        elif self.replace in ['backfill', 'bfill', 'pad', 'ffill']:
            X_replaced = X.fillna(method=self.replace)

        elif type(self.replace) == dict:
            X_replaced = X.copy()
            for key, value in self.replace.iteritems():
                if value in ['backfill', 'bfill', 'pad', 'ffill']:
                    X_replaced[key] = X_replaced[key].fillna(method=value)
                else:
                    X_replaced[key] = X_replaced[key].fillna(value=value)
        else:
            X_replaced = X.fillna(value=self.replace)
        return X_replaced


class DFMissingStr(TransformerMixin):
    '''
    METHODS
    most_frequent:
    backfill/bfill:
    pad/ffill:
    '''

    def __init__(self, replace):
        self.replace = replace
        self.statistics_ = None

    def fit(self, X, y=None):

        if type(self.replace) == dict:
            for key, value in self.replace.iteritems():
                if value == 'most_frequent':
                    self.replace[key] = X[key].mode()[0]
        elif self.replace == 'most_frequent':
            self.statistics_ = X.mode().to_dict()
            for key, value in self.statistics_.items():
                self.statistics_[key] = value.values()[0]

        return self

    def transform(self, X):
        if self.replace == 'most_frequent':
            X_replaced = X.fillna(self.statistics_)

        elif self.replace in ['backfill', 'bfill', 'pad', 'ffill']:
            X_replaced = X.fillna(method=self.replace)

        elif type(self.replace) == dict:
            X_replaced = X.copy()
            for key, value in self.replace.iteritems():
                if value in ['backfill', 'bfill', 'pad', 'ffill']:
                    X_replaced[key] = X_replaced[key].fillna(method=value)
                else:
                    X_replaced[key] = X_replaced[key].fillna(value=value)
        else:
            X_replaced = X.fillna(value=self.replace)
        return X_replaced


class DFClip(TransformerMixin):
    '''
    Refer https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.clip.html
    '''

    def __init__(self, lower=None, upper=None):
        self.lower = lower
        self.upper = upper

    def fit(self, X, y=None):
        # Do Nothing
        return self

    def transform(self, X):
        X_clipped = X.clip(self.lower, self.upper)
        return X_clipped


class DFNormalizer(TransformerMixin):
    # Row wise transformer? - Can be removed if so
    def __init__(self, norm='l2', copy=True):
        self.norm = norm
        self.copy = copy
        self.ss_ = None

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # assumes X is a DataFrame
        self.ss_ = Normalizer()
        Xss = self.ss_.transform(X)
        Xscaled = pd.DataFrame(Xss, index=X.index, columns=X.columns)
        return Xscaled


class UnitConv(TransformerMixin):

    def __init__(self, frm, to):
        self.frm = frm
        self.to = to

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        Xz = X.apply(lambda x: Q_(x, self.frm).to(self.to))
        return Xz


class DFOneHot(TransformerMixin):
    '''
    dummy_na: Unseeen values Boolean
    reference - https://pandas.pydata.org/pandas-docs/stable/generated/pandas.get_dummies.html
    '''

    def __init__(self, dummy_na=False):
        self.dummy_na = dummy_na
        self.categories_ = {}

    def fit(self, X, y=None):
        for col in list(X):
            self.categories_[col] = X[col].unique()
        return self

    def transform(self, X):
        X_new = X.copy()
        for colname, levels in self.categories_.iteritems():
            X_new[colname][~X_new[colname].isin(levels)] = np.NaN
        X_dummy = pd.get_dummies(X_new, dummy_na=self.dummy_na)

        fit_colnames = []
        for colname, levels in self.categories_.iteritems():
            for level in levels:
                fit_colnames.append(str(colname) + '_' + str(level))

        X_dummy = X_dummy.reindex(columns=fit_colnames, fill_value=0)

        return X_dummy


class ColumnExtractor(TransformerMixin):

    def __init__(self, cols):
        self.cols = cols

    def fit(self, X, y=None):
        # stateless transformer
        return self

    def transform(self, X):
        # assumes X is a DataFrame
        Xcols = pd.DataFrame(X[self.cols])
        return Xcols


class Binning(TransformerMixin):

    # Binning - https://pandas.pydata.org/pandas-docs/stable/generated/pandas.cut.html

    def __init__(self, nBins, labels=None, right=True, include_lowest=False):
        self.nBins = nBins
        self.right = right
        self.include_lowest = include_lowest
        self.labels = labels
        self._bounds = {}

    def fit(self, X, y=None):
        for i in list(range(X.shape[1])):
            fitted = pd.cut(X.iloc[:, i], self.nBins, right=self.right, retbins=True,
                            include_lowest=self.include_lowest)
            fitted[1][0] = floor(fitted[1][0] * 1000) / 1000
            fitted[1][-1] = ceil(fitted[1][-1] * 1000) / 1000
            self._bounds[i] = fitted[1]
        return self

    def transform(self, X):
        for i in list(range(X.shape[1])):
            Xtransform_temp = pd.cut(X.iloc[:, i], self._bounds.get(i),
                                     right=self.right, labels=self.labels,
                                     include_lowest=self.include_lowest)
            if 'Xtransform' in locals():
                Xtransform = pd.concat([Xtransform.reset_index(drop=True), Xtransform_temp], axis=1)
            else:
                Xtransform = Xtransform_temp

        Xtransform = pd.DataFrame(Xtransform, index=X.index, columns=X.columns)
        return Xtransform


class QBinning(TransformerMixin):

    # Quantile binning - https://pandas.pydata.org/pandas-docs/stable/generated/pandas.qcut.html
    ## WARNING: Lower limit is -Inf and upper limit is Inf

    def __init__(self, q, labels=None, duplicates='raise'):
        self.q = q
        self.labels = labels
        self.duplicates = duplicates
        self._bounds = {}
        # print "QBinning successfully initiated."

    def fit(self, X, y=None):
        for i in list(range(X.shape[1])):
            fitted = pd.qcut(X.iloc[:, i], self.q, labels=self.labels, retbins=True, duplicates=self.duplicates)
            fitted[1][0] = float("-inf")
            fitted[1][len(fitted[1]) - 1] = float("inf")
            self._bounds[i] = fitted[1]
        # print "QBinning - successful fit."
        return self

    def transform(self, X):
        for i in list(range(X.shape[1])):
            Xtransform_temp = pd.cut(X.iloc[:, i], self._bounds.get(i),
                                     labels=self.labels)
            if 'Xtransform' in locals():
                Xtransform = pd.concat([Xtransform.reset_index(drop=True), Xtransform_temp], axis=1)
            else:
                Xtransform = Xtransform_temp
        Xtransform = pd.DataFrame(Xtransform, index=X.index, columns=X.columns)
        return Xtransform


class SeriesToDF(TransformerMixin):
    '''
    Implementation of pandas friendly numeric cast
    '''

    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_df = pd.DataFrame(X)
        return X_df


class DFToSeries(TransformerMixin):
    '''
    Implementation of pandas friendly numeric cast
    '''

    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_s = X.iloc[:, 0]
        return X_s


class DFImputer(TransformerMixin):
    # Imputer but for pandas DataFrames

    def __init__(self, strategy='mean'):
        self.strategy = strategy
        self.imp = None
        self.statistics_ = None

    def fit(self, X, y=None):
        self.imp = Imputer(strategy=self.strategy)
        self.imp.fit(X)
        self.statistics_ = pd.Series(self.imp.statistics_, index=X.columns)
        return self

    def transform(self, X):
        # assumes X is a DataFrame
        Ximp = self.imp.transform(X)
        Xfilled = pd.DataFrame(Ximp, index=X.index, columns=X.columns)
        return Xfilled


class DFFeatureUnion(TransformerMixin):
    # FeatureUnion but for pandas DataFrames

    def __init__(self, transformer_list):
        self.transformer_list = transformer_list
        print "Feature union successfully initiated."

    def fit(self, X, y=None):
        for (name, t) in self.transformer_list:
            t.fit(X, y)

        print "Feature union - successful fit."
        return self

    def transform(self, X):
        # assumes X is a DataFrame
        Xts = [t.transform(X) for _, t in self.transformer_list]
        Xunion = reduce(lambda X1, X2: pd.merge(X1, X2, left_index=True, right_index=True), Xts)
        print "Feature union - successful transform."
        return Xunion
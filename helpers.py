# a function to filter the top categories in a categorical column
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.pipeline import Pipeline

from sklearn.base import BaseEstimator, TransformerMixin

def filter_top_cats(array, t = 5, method = 'number'):
#     '''Takes the column and reduces the number of categorical values
#
# Parameters:
# -----------
# n : number of categorical values to reduce to.  Result will have n + 1 for
#     the n + other
#
# method : 'number' or 'percent'  'number' indicates that t is the minimum
#          number of records for an element to persist.  'percent' indicates
#          that t is the minumum percent of of the column a value represents.
# '''

    df = pd.DataFrame(array)

    # treat each col individually
    for col in df.columns:

        # choose reduction method and get the names of the values to keep
        if method == 'number':
            keep = df[col].value_counts()[:t].index
        elif method == 'percent':
            keep = df[col].value_counts()[df[col].value_counts(normalize = True) > t]
        else:
            warnings.warn('invalid value for "method." Keeping all values')
            return array

        # replace all values that we are not keeping with 'other'
        df[col] = df[col].map(lambda x: x if x in keep else 'other')

    return df.astype('object')


# convert text series to bool then int
def boolstring_to_int(s):
    return s.astype('bool').astype('int')

## Column Transformer does not return column names if the feature transformes
## do not provide get_feture_names().  This function gets the names from the
## input variables.  We don't need this for training the model but it is
## helpful for testing along the way.
# https://johaupt.github.io/scikit-learn/tutorial/python/data%20processing/ml%20pipeline/model%20interpretation/columnTransformer_feature_names.html
def get_feature_names(column_transformer):
    """Get feature names from all transformers.
    Returns
    -------
    feature_names : list of strings
        Names of the features produced by transform.
    """
    # Remove the internal helper function
    #check_is_fitted(column_transformer)

    # Turn loopkup into function for better handling with pipeline later
    def get_names(trans):
        # >> Original get_feature_names() method
        if trans == 'drop' or (
                hasattr(column, '__len__') and not len(column)):
            return []
        if trans == 'passthrough':
            if hasattr(column_transformer, '_df_columns'):
                if ((not isinstance(column, slice))
                        and all(isinstance(col, str) for col in column)):
                    return column
                else:
                    return column_transformer._df_columns[column]
            else:
                indices = np.arange(column_transformer._n_features)
                return ['x%d' % i for i in indices[column]]
        if not hasattr(trans, 'get_feature_names'):

        # >>> Change: Return input column names if no method avaiable
            # For transformers without a get_features_names method, use the input
            # names to the column transformer
            if column is None:
                return []
            else:
                return [name + "__" + f for f in column]

        return [name + "__" + f for f in trans.get_feature_names()]

    ### Start of processing
    feature_names = []

    # Allow transformers to be pipelines. Pipeline steps are named differently, so preprocessing is needed
    # if type(column_transformer) == sklearn.pipeline.Pipeline:
    if type(column_transformer) == Pipeline:
        l_transformers = [(name, trans, None, None) for step, name, trans in column_transformer._iter()]
    else:
        # For column transformers, follow the original method
        l_transformers = list(column_transformer._iter(fitted=True))


    for name, trans, column, _ in l_transformers:
        # if type(trans) == sklearn.pipeline.Pipeline:
        if type(trans) == Pipeline:
            # Recursive call on pipeline
            _names = get_feature_names(trans)
            # if pipeline has no transformer that returns names
            if len(_names)==0:
                _names = [name + "__" + f for f in column]
            feature_names.extend(_names)
        else:
            feature_names.extend(get_names(trans))

    return feature_names

#################################################################
#################           Classes             #################
#################################################################

# Custom Transformer for Date Field
class DatePrep(BaseEstimator, TransformerMixin):
    def __init__(self, strategy = 'median'):
        # store the srategy
        self.strategy = strategy
    def fit(self, df, y = None):
        df.replace(0, np.nan, inplace = True)
        # store the value to fill with based on the fill strategy
        self.fill_vals = eval(f'df.{self.strategy}()')
        return self

    def transform(self, df, y = None):

        # replace 0 with nan
        df.replace(0, np.nan, inplace = True)

        # fill construction value
        df['construction_year'].fillna(self.fill_vals['construction_year'],
                                       inplace = True)

        # get datetime of date recorded
        date_recorded = pd.to_datetime(df['date_recorded'])

        # calculate delta to consruction date
        years_since_construction = date_recorded.dt.year - df['construction_year']

        # convert date to days
        days_since_epoch = (date_recorded - datetime(1970,1,1)).dt.days
        return pd.concat([days_since_epoch, years_since_construction], axis = 1)

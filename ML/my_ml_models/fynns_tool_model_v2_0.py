'''
NOTE FOR VERSION 1.1:
Note time: 2023.12.14 00:37


- Encapsulate as global functions, instead of inside class methods.
Reason: Class methods are not efficient.

- Remove todf() & toseries() & raiseTypeError() & buildModelComm()
'''

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from xgboost import XGBRegressor


def cleanData(X: pd.DataFrame, y: pd.Series | None = None, numColsimputeStrategy: str = 'mean', catColsimputeStrategy: str = 'most_frequent', encoderName: str = 'oe', handle_unknown: str = 'error') -> tuple[pd.DataFrame, pd.Series]:
    '''
    For numerical data: impute missing values
    For categorical data: encode and impute missing values
    '''

    print(f'''
        numColsimputeStrategy: {numColsimputeStrategy}
        catColsimputeStrategy: {catColsimputeStrategy}
        encoderName: {encoderName} 
        handle_unknown: {handle_unknown}    
          ''')
    # find numerical and categorical columns from X
    numColsX = list(X.select_dtypes(exclude=['object']).columns)
    catColsX = list(X.select_dtypes(include=['object']).columns)
    print(f'''
        numColsX: {numColsX}
        catColsX: {catColsX}
          ''')
    if encoderName == 'oe':
        encoder = OrdinalEncoder(handle_unknown=handle_unknown)
    elif encoderName == 'ohe':
        encoder = OneHotEncoder(handle_unknown=handle_unknown)
    else:
        raise TypeError("Argument encoder should be 'oe' or 'ohe'.")
    # build column transformer
    ct = ColumnTransformer(transformers=[
        ('numCols', SimpleImputer(strategy=numColsimputeStrategy), numColsX),
        ('catCols', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy=catColsimputeStrategy)),
            ('encoder', encoder)]), catColsX)
    ])
    # fit_transform X
    X = pd.DataFrame(ct.fit_transform(X))
    X.columns = numColsX+catColsX

    if type(y) != type(None):
        yname=y.name
        if y.dtype == 'object':
            if encoderName == 'oe':
                encoder = OrdinalEncoder(handle_unknown=handle_unknown)
            elif encoderName == 'ohe':
                encoder = OneHotEncoder(handle_unknown=handle_unknown)
            else:
                raise TypeError("Argument encoder should be 'oe' or 'ohe'.")
            y = encoder.fit_transform(y.values.reshape(-1, 1))
        elif y.dtype in ['int64', 'float64']:
            # impute y
            y = SimpleImputer(strategy=numColsimputeStrategy).fit_transform(
                y.values.reshape(-1, 1))
            y=pd.DataFrame(y)
            y.columns=[yname]
        else:
            Exception(f'Unknown dtype of y: {y.dtype}')
        return X, y
    else:
        return X

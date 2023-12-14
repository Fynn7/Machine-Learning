'''
NOTE FOR VERSION 1.1:
Note time: 2023.12.14 00:37


- Encapsulate as global functions, instead of inside class methods.
Reason: Class methods are not efficient.

- Removed todf() & toseries() & raiseTypeError() & buildModelComm()
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
        try:
            yname = y.name
        except AttributeError:
            raise TypeError("Expect y as a pandas Series.")
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
            y = pd.DataFrame(y)
            y.columns = [yname]
        else:
            Exception(f'Unknown dtype of y: {y.dtype}')
        return X, y
    else:
        return X


def fitModel(tts: tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series], modelName: str = 'RandomForestRegressor', cv: int = 5, **modelArgs) -> tuple[pd.Series, dict[float | np.ndarray]]:
    # if model is not given or it is given illegally
    Xtrain, Xtest, ytrain, ytest = tts
    if type(ytrain)==np.ndarray and type(ytest)==np.ndarray:
        print("Found ytrain,ytest as np.ndarray type, reshaping -1 into pd.Series.")
        ytrain,ytest=pd.Series(ytrain.reshape(-1)),pd.Series(ytest.reshape(-1))
    try:
        model = eval(f"{modelName}(**modelArgs)")
        print(f'''
        Successfully create model: {modelName}
        ''')
    except NameError as e:  # `modelName` type incorrect or unknown model
        model = RandomForestRegressor(**modelArgs)
        print(
            f"Illegal model name or type. Model argument set to default as `RandomForestRegressor`.\nOriginal error message: {e}")
    except Exception as e:
        print(
            f"Unknow error occurs while building the model.\nOriginal error message: {e}")
    model.fit(Xtrain, ytrain)
    pred = model.predict(Xtest)
    score = model.score(Xtest, ytest)
    mae_score = mean_absolute_error(ytest, pred)
    try:
        cv_score = -1 * cross_val_score(model, Xtrain, ytrain, cv=cv,
                                        scoring='neg_mean_absolute_error')
    except ValueError as e:
        raise ValueError(
            f"Samples maybe less than argument `cv` value as in `cross_val_score`. Try to set another `cv` value or try on with more samples.\nOriginal error message: {e}")
    except Exception as e:
        print(
            f"Unknow error occurs while cross validation.\nOriginal error message: {e}")
    print(f"y_pred:{pred};\ny_true:\n{ytest}")
    list(fitModel.__code__.co_varnames)
    # for attr in list(fitModel.__code__.co_varnames):
        # print(attr,' = ',eval(attr))
    return pred, {'score': score, 'mae_score': mae_score, 'cv_score': cv_score}

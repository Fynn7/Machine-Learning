'''
NOTE FOR VERSION 1.1:
Note time: 2023.12.13.22:49

Clear out methods as `Pipeline` for step-by-step data cleaning, collect them into a whole "cleanData" method.

for all pipelines and model training, we also use only 1 method to represent them all, as we're using 1 parameter called 'model:str=...'
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

def raiseTypeError(arg: object, shouldBe: type | object, origErrMsg: str | None = None) -> None:
    errmsg = f'【{arg}】 should be 【{shouldBe}】, not 【{type(arg)}.】'
    if origErrMsg:
        errmsg += f"Original Error Message: {str(origErrMsg)}"
    raise TypeError(errmsg)


def todf(l: list | pd.DataFrame | pd.Series) -> pd.DataFrame:
    '''
    todf & checks if it's DataFrame.
    '''
    try:
        return pd.DataFrame(l)
    except ValueError:
        raiseTypeError(l, '`DataFrame-Like`')


def toseries(l: list | pd.DataFrame | pd.Series) -> pd.DataFrame:
    '''
    todf & checks if it's DataFrame.
    '''
    try:
        return pd.Series(l)
    except ValueError:
        raiseTypeError(l, '`Series-Like`')


def buildModelComm(args: dict, prefix: str = "", suffix: str = "") -> str:
    '''
    Unpack/release arguments to a single string.

    ```
    >>> # run the model
    >>> args = {
        n_estimators : 100, 
        random_state : 42,
        }
    >>> comm = buildModelComm(args, prefix = "RandomForestRegressor(", suffix = ")")
    >>> model = eval(comm)
    >>> model.__str__
    <method-wrapper '__str__' of RandomForestRegressor object at 0x000001DFFC38FA10>

    >>> comm
    'RandomForestRegressor(n_estimators=100,random_state=42)'
    ```
    '''
    if type(args) != dict:
        raiseTypeError(args, dict)
    comm = prefix
    for k, v in args.items():
        comm += f"{k}="
        if type(v) == str:  # if the value of the argument is str type, like handle_unknown='ignore'
            comm += f"\'{v}\',"
        else:
            comm += f"{v},"
    comm = comm.strip(',') + suffix
    return comm


class RegressionModel:
    def __init__(self, X: pd.DataFrame, y: pd.Series) -> None:
        self.init(X,y)

    def __str__(self) -> str:
        return self._df.to_string()

    def __repr__(self) -> str:
        return self._df.to_string()

    def _update(self, **attrs) -> None:
        '''
        Update attributes directly by their names.
        '''
        for a, v in attrs.items():
            try:
                exec(f"self.{a}")  # check if the attribute exists
            except AttributeError:
                raise AttributeError(f"Attribute {a} not found.")
            exec(f"self.{a}=v")
            print("Attribute", a, "updated.")

    def init(self, X: pd.DataFrame | None = None, y: pd.Series | None = None) -> None:
        '''
        Simplized. Only initiallize X,y.
        '''
        if type(X) != type(None):
            self._X = todf(X) # check types and also convert to DataFrame
        if type(y) != type(None):
            self._y = toseries(y) # check types and also convert to Series
        dfcopy = self._X.copy()
        dfcopy[self._y.name] = self._y
        self._df = dfcopy
        # For update log: pop out these names we don't expect: 'self','attrNames','attr', so we slice it to [1:-2]
        attrNames = list(self.init.__code__.co_varnames)[1:-3]
        for attr in attrNames:
            # if user DID given these values to the corresponding arguments in .init()/__init__() constructing/updating methode
            if type(eval(attr)) != type(None):
                print(attr, "initiallized/updated.")

    def info(self) -> None:
        for attr, v in self.__dict__.items():
            print(
                f"==========================================\n{attr.strip('_')}:\n---------------------\n{v}\n")
            print("==========================================")

    # -------------------------------------------
    # All Getter Functions
    def getdf(self) -> pd.DataFrame:
        return self._df
    
    def getXy(self) -> tuple[pd.DataFrame, pd.Series]:
        return self._X, self._y

    # -------------------------------------------
    # Data Preprocessing
    # Handling missing values (impute) & Encoding categorical columns (encoder)
    # => preprocessing (class <ColumnTransformer>)
    # => autoPipeline
    
    def cleanData(self,cleanX:bool=True,cleany:bool=True, numColsimputeStrategy:str='mean',catColsimputeStrategy: str = 'most_frequent',encoderName:str='oe',handle_unknown:str='error',save:bool=False) -> tuple[pd.DataFrame, pd.Series]:
        '''
        For numerical data: impute missing values
        For categorical data: encode and impute missing values
        '''
        X, y = self._X, self._y
        if cleanX:
        # find numerical and categorical columns from X
            numColsX=X.select_dtypes(exclude=['object']).columns
            catColsX=X.select_dtypes(include=['object']).columns
            if encoderName=='oe':
                encoder=OrdinalEncoder(handle_unknown=handle_unknown)
            elif encoderName=='ohe':
                encoder=OneHotEncoder(handle_unknown=handle_unknown)
            else:
                raiseTypeError(encoder, 'oe or ohe')
            # build column transformer
            ct=ColumnTransformer(transformers=[
                ('numCols',SimpleImputer(strategy=numColsimputeStrategy),numColsX),
                ('catCols',Pipeline(steps=[
                    ('imputer',SimpleImputer(strategy=catColsimputeStrategy)),
                    ('encoder',encoder)]),catColsX)
            ])
            # fit_transform X
            X=ct.fit_transform(X)

        if cleany:
            if y.dtype=='object':
                if encoderName=='oe':
                    encoder=OrdinalEncoder(handle_unknown=handle_unknown)
                elif encoderName=='ohe':
                    encoder=OneHotEncoder(handle_unknown=handle_unknown)
                else:
                    raiseTypeError(encoder, 'oe or ohe')
                y=encoder.fit_transform(y.values.reshape(-1,1))
            elif y.dtype in ['int64','float64']:
                # impute y
                y=SimpleImputer(strategy=numColsimputeStrategy).fit_transform(y.values.reshape(-1,1))
            else:
                Exception(f'Unknown dtype of y: {y.dtype}')
        if save:
            print("DO NOT SAVE IT. NOT YET FINISHED.")
            # self.init(X,y)
        return X,y

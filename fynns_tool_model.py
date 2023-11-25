import time
import sys
import os

from functools import wraps
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import seaborn as sns

from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder

# def getArgs(f):
#     '''Note that the function that uses this decorator DOES NOT RETURN!'''
#     @wraps(f)
#     def w(*args,**kwargs)->tuple[list[tuple],list[tuple]]:
#         if f(*args,**kwargs)!=None:
#             print("!!!Warning: This function will NOT return its original return, instead it will return argument infos, if you use this decorator!")
#         a = [(param, value) for param, value in zip(f.__code__.co_varnames, args)] # args of the original function
#         kw = [(key, value) for key, value in kwargs.items()]
#         return a,kw
#     return w


# class InvalidParameterError(Exception):
#     def __init__(self, *args: object) -> None:
#         super().__init__(*args)

# class ArrayLike(list):

# def isNumeric(o:object)->bool:
#     if type(o)==str:
#         return o.isnumeric()
#     try:
#         float(o)
#     except ValueError:
#         return False

def raiseTypeError(arg: object, shouldBe: type | object) -> None:
    '''
    Should finally switch to InvalidParameterError,
    and written inside class as a methode
    But we don't wanna focus on this right now

    See: `InvalidErrorSample.txt` under this folder
    '''
    print('''    Should finally switch to InvalidParameterError,
    and written inside class as a methode
    But we don't wanna focus on this right now
          
    See: `InvalidErrorSample.txt` under this folder''')
    raise TypeError(
        f'【{arg}】 should be 【{shouldBe}】, not 【{type(arg)}.】')


def todf(l: list | dict | pd.Series) -> pd.DataFrame:
    try:
        return pd.DataFrame(l)
    except ValueError:
        raiseTypeError(l, '`list`|`dict`|`pd.Series`')


def isFunc(o: object) -> bool:
    return callable(o)


def isType(o: object, t: type) -> bool:
    return type(o) == t


def isdf(o: object) -> bool:
    return type(o) == type(pd.DataFrame())


def isNone(o: object) -> bool:
    return type(o) == type(None)


def isEmpty(df: pd.DataFrame) -> bool:
    return df.empty if type(df) == type(pd.DataFrame()) else raiseTypeError(df, 'DataFrame')


def compare(oldData: pd.DataFrame, newData: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(oldData).compare(pd.DataFrame(newData))


def exclude(df: pd.DataFrame, exclude: str | list | type) -> pd.DataFrame:
    return df.select_dtypes(exclude=exclude) if type(df) == type(pd.DataFrame()) else raiseTypeError(df, 'DataFrame')


def include(df: pd.DataFrame, include: str | list | type) -> pd.DataFrame:
    '''
    include(df,np.float64)
    ```
    '''
    return df.select_dtypes(include=include) if type(df) == type(pd.DataFrame()) else raiseTypeError(df, 'DataFrame')


def toNumeric(s: pd.Series, errors: str = 'coerce') -> pd.Series:
    '''
    # Example:
    ```
    >>> df = pd.DataFrame({'col1': ['1', '2'], 'col2': [3, 'ABC']})
    >>> for col in df.columns:
    >>>     df[col] = toNumeric(df[col])
    >>> df
        col1    col2
    0   1       3.0
    1   2       NaN
    ```
    '''
    return pd.to_numeric(s, errors=errors)


def scoreDataset(X: pd.DataFrame, y: pd.Series, model: str = 'RandomForestRegressor', test_size: float | int | None = None, train_size: float | int | None = None, random_state: int | None = None, **kwargs):
    if type(model) != str:
        raiseTypeError(model, str)
    Xtrain, Xtest, ytrain, ytest = train_test_split(
        X, y, test_size=test_size, train_size=train_size, random_state=random_state)
    # building up the code
    argsComm = "("
    for k, v in kwargs.items():
        argsComm += f"{k}="
        argsComm += f"{v},"
    argsComm += "random_state=random_state)"
    m = eval(f"{model}{argsComm}")
    try:
        m.fit(Xtrain, ytrain)
    except NameError:
        raise NameError(f"Model name `{model}` not found.")
    p = m.predict(Xtest)
    score = mean_absolute_error(ytest, p)
    print(f'''
    Model: {model}
    Model Arguments: {kwargs}
    MAE Score: {score}
          ''')
    return score


def ohe(Xtrain: pd.DataFrame | pd.Series, Xtest: pd.DataFrame | pd.Series, catCols: list | str | None = None, handle_unknown: str = 'ignore', sparse: bool = False, **kwargs) -> tuple[pd.DataFrame, pd.Series]:
    '''
    Return encoded categorical features DataFrame with OneHotEncoder.

    - X: should be the whole matrix(`pd.DataFrame()`) of the features. INCLUDING NOT-CATEGORICAL FEATURES
    - y: should be the whole vector(`pd.Series()`) of the target feature.

    # Example:
    ```
    >>> df = pd.DataFrame({'taste': ['Sweet','Sweet', 'Sour','Sweet','Sour'], 'size': ['Big','Big','Small','Medium','Small'],'int_size': [7,8,2,4,2],'color':['Red','Green','Green','Red','Green']})
    >>> df
        taste	size	int_size	color
    0	Sweet	Big	    7	        Red
    1	Sweet	Big	    8	        Green
    2	Sour	Small	2	        Green
    3	Sweet	Medium	4	        Red
    4	Sour	Small	2	        Green
    >>> X = df[['taste','size','int_size']] # or: `X = df.select_dtypes(include=object)`
    >>> catCols = ['taste','size']
    >>> ohe(Xtrain,Xtest, catCols = catCols)
    (     0    1    2    3    4
    3   0.0  1.0  0.0  1.0  0.0
    0   0.0  1.0  1.0  0.0  0.0
    2   1.0  0.0  0.0  0.0  1.0, # totally same as 4 because their features are totally same, so they're included in 1 same type in One Hot Encoding
        0    1    2    3    4
    1   0.0  1.0  1.0  0.0  0.0
    4   1.0  0.0  0.0  0.0  1.0)
    ```

    # Explaination of the output:
    For One Hot Encoding, we observe all categorical features into single 1 monolithic, whole feature that is represented by a bunch of binary digits.

    There's 3 features and there should be 2*3*2=16<2**5 binary-formed encoded features
    eg: for Sample 3(`df[3]`): it is encoded to 01010, which represents Sweet, Medium, Red,
    and this 3 features are actually unique in this dataset,
    so there are no other 01010 encoded sample as sample 3.

    On the contrary of sample 2 and 4, their categorical features are TOTALLY same, so they were encoded totally same as well.

    NOTE: Those Features whose cardinality low is, are fit to preprocess with OneHotEncoder.
    Otherwise it's ok to use OrdinalEncoder
    '''
    if catCols == None:
        print(
            "Warning: You didn't input argument {catcols}, so we select all object-type columns to be one-hot encoded.")
        catCols = Xtrain.select_dtypes(include=object).columns
    otherCols = list(set(Xtrain.columns)-set(catCols))
    # Apply one-hot encoder to each column with categorical data
    print("Categorical Columns:", list(catCols))
    print("Other Columns:", otherCols)
    encoder = OneHotEncoder(
        handle_unknown=handle_unknown, sparse_output=sparse)
    catXtrainEncoded = pd.DataFrame(encoder.fit_transform(Xtrain[catCols]))
    catXtestEncoded = pd.DataFrame(encoder.transform(Xtest[catCols]))

    # One-hot encoding removed index; put it back
    catXtrainEncoded.index = Xtrain.index
    catXtestEncoded.index = Xtest.index

    # XtrainEncoded.columns = Xtrain.columns
    # XtestEncoded.columns= Xtest.columns

    # Add one-hot encoded columns to other features
    XtrainEncoded = pd.concat([Xtrain[otherCols], catXtrainEncoded], axis=1)
    XtestEncoded = pd.concat([Xtest[otherCols], catXtestEncoded], axis=1)

    # Ensure all columns have string type
    XtrainEncoded.columns = XtrainEncoded.columns.astype(str)
    XtestEncoded.columns = XtestEncoded.columns.astype(str)

    return XtrainEncoded, XtestEncoded


def oe(Xtrain: pd.DataFrame | pd.Series, Xtest: pd.DataFrame | pd.Series, catCols: list | str | None = None, handle_unknown: str = 'error'):
    '''
    Applying Ordinal Encoder to categorical feature columns.


    We assume you've got the good columns to be ordinal encoded.
    Bad columns mean the features in those, couldn't be found in validation/test dataframe.
    '''
    # Preprocess:
    if catCols == None:
        print(
            "Warning: You didn't input argument {catcols}, so we select all object-type columns to be one-hot encoded.")
        catCols = [col for col in Xtrain.columns if Xtrain[col].dtype == "object"]
        # Would this work instead? >>>
        # catCols = Xtrain.select_dtypes(include=object).columns

    # Columns that can be safely ordinal encoded
    goodCols = [col for col in catCols if set(
        Xtest[col]).issubset(set(Xtrain[col]))]  # because sample elements from Xtest is possible not appearing in Xtrain!
    badCols = list(set(catCols)-set(goodCols))
    print(f'''
    Columns that can fit ordinal encoder: {goodCols}
    Columns that cannot fit ordinal encoder: {badCols}
    ''')
    encoder = OrdinalEncoder(handle_unknown=handle_unknown)
    # !!! We cannot use encoder.transform for Xtest dataset, why?
    XtrainEncoded = encoder.fit_transform(Xtrain)
    XtestEncoded = encoder.fit_transform(Xtest)
    return XtrainEncoded, XtestEncoded


def nunique(df: pd.DataFrame, catCols: list | str | None = None, sort: bool = False) -> dict[tuple]:
    '''
    Return 
    '''
    if type(df) != type(pd.DataFrame()):
        raiseTypeError(df, 'DataFrame')
    if type(catCols) == type(None):
        print(
            "Warning: You didn't input argument {catcols}, so we select all object-type columns.")
        catCols = [col for col in df.columns if df[col].dtype == "object"]
        # Would this work instead? >>>
        # catCols = Xtrain.select_dtypes(include=object).columns
    return sorted(dict(
        zip(catCols, list(
            map(lambda col: df[col].nunique(), catCols)
        )
        )).items(), key=lambda t: t[1]) if sort else dict(
        zip(catCols, list(
            map(lambda col: df[col].nunique(), catCols)
        )
        ))


# class Data:
#     def __init__(self, df: pd.DataFrame | pd.Series, Xcols: list | str, ycol: list | str, test_size: float | int | None = None, train_size: float | int | None = None, random_state: int | None = None) -> None:
#         self._df = pd.DataFrame(df)
#         self._X, self._y = df[Xcols], df[ycol]
#         self._Xtrain, self._Xtest, self._ytrain, self._ytest = train_test_split(
#             df[Xcols], df[ycol], train_size=train_size, test_size=test_size, random_state=random_state)

#     def getData(self)->pd.DataFrame:
#         return self._df

#     def getXy(self)->tuple[pd.DataFrame,pd.Series]:
#         return self._X, self._y

#     def getTrainTestData(self)->tuple[pd.DataFrame,pd.DataFrame,pd.Series,pd.Series]:
#         return self._Xtrain, self._Xtest, self._ytrain, self._ytest

#     def update(self, **kwargs) -> None:
#         for argName,val in kwargs.items():

class Model():
    def __init__(self,df:pd.DataFrame, Xcols:list|str,ycol:str, model: str = 'RandomForestRegressor',test_size: float | int | None = None, train_size: float | int | None = None, random_state: int | None = None) -> None:
        self._df=df if isType(df,pd.DataFrame) else raiseTypeError(df,pd.DataFrame)
        self._X=df[Xcols]
        self._y=df[ycol]
        self._Xtrain, self._Xtest, self._ytrain, self._ytest = train_test_split(
            df[Xcols], df[ycol], train_size=train_size, test_size=test_size, random_state=random_state)

        self._model=model if isType(model,str) else raiseTypeError(model,str)
        self._mae = 0  # Mean Absolute Error score

    def __str__(self) -> str:
        return self._df.to_string()

    def __repr__(self) -> str:
        return self._df.to_string()

    def _validateParam(self) -> None:
        ...

    def getData(self):
        return self._df

    def update(self, **kwargs) -> None:
        '''
        You can literaly update any EXISTED attribute you want.

        If you want to update X:
        just input argument X=df[Xcols]
        '''
        for k,v in kwargs.items():
            try:
                print("attribute name: ",k,"; ORIGINAL value: ",v)
            except AttributeError:
                raise AttributeError(f"Attribute {k} not found.")
            exec(f"self._{k}={v}")
            print("attribute name: ",k,"; NEW value: ",v)

    def mae(self, inplace:bool=False,**kwargs):
        model=self._model
        Xtrain,Xtest,ytrain,ytest=self._Xtrain,self._Xtest,self._ytrain,self._ytest
        # building up the code
        argsComm = "("
        for k, v in kwargs.items():  # extra parameters for the model object itself
            argsComm += f"{k}="
            argsComm += f"{v},"
        argsComm += "random_state=random_state)"
        m = eval(f"{model}{argsComm}")
        try:
            m.fit(Xtrain, ytrain)
        except NameError:
            raise NameError(f"Model name `{model}` not found.")
        p = m.predict(Xtest)
        score = mean_absolute_error(ytest, p)
        print(f'''
        Model: {model}
        Model Arguments: {kwargs}
        MAE Score: {score}
            ''')
        if inplace:
            self._mae = score
        return score
    
    def ohe(self, catCols: list | str | None = None, handle_unknown: str = 'ignore', sparse: bool = False, inplace:bool=False,**kwargs) -> tuple[pd.DataFrame, pd.Series]:
        '''
        Return encoded categorical features DataFrame with OneHotEncoder.

        - X: should be the whole matrix(`pd.DataFrame()`) of the features. INCLUDING NOT-CATEGORICAL FEATURES
        - y: should be the whole vector(`pd.Series()`) of the target feature.

        # Example:
        ```
        >>> df = pd.DataFrame({'taste': ['Sweet','Sweet', 'Sour','Sweet','Sour'], 'size': ['Big','Big','Small','Medium','Small'],'int_size': [7,8,2,4,2],'color':['Red','Green','Green','Red','Green']})
        >>> df
            taste	size	int_size	color
        0	Sweet	Big	    7	        Red
        1	Sweet	Big	    8	        Green
        2	Sour	Small	2	        Green
        3	Sweet	Medium	4	        Red
        4	Sour	Small	2	        Green
        >>> X = df[['taste','size','int_size']] # or: `X = df.select_dtypes(include=object)`
        >>> catCols = ['taste','size']
        >>> ohe(Xtrain,Xtest, catCols = catCols)
        (     0    1    2    3    4
        3   0.0  1.0  0.0  1.0  0.0
        0   0.0  1.0  1.0  0.0  0.0
        2   1.0  0.0  0.0  0.0  1.0, # totally same as 4 because their features are totally same, so they're included in 1 same type in One Hot Encoding
            0    1    2    3    4
        1   0.0  1.0  1.0  0.0  0.0
        4   1.0  0.0  0.0  0.0  1.0)
        ```

        # Explaination of the output:
        For One Hot Encoding, we observe all categorical features into single 1 monolithic, whole feature that is represented by a bunch of binary digits.

        There's 3 features and there should be 2*3*2=16<2**5 binary-formed encoded features
        eg: for Sample 3(`df[3]`): it is encoded to 01010, which represents Sweet, Medium, Red,
        and this 3 features are actually unique in this dataset,
        so there are no other 01010 encoded sample as sample 3.

        On the contrary of sample 2 and 4, their categorical features are TOTALLY same, so they were encoded totally same as well.

        NOTE: Those Features whose cardinality low is, are fit to preprocess with OneHotEncoder.
        Otherwise it's ok to use OrdinalEncoder
        '''
        Xtrain,Xtest=self._Xtrain,self._Xtest
        if catCols == None:
            print(
                "Warning: You didn't input argument {catcols}, so we select all object-type columns to be one-hot encoded.")
            catCols = Xtrain.select_dtypes(include=object).columns
        otherCols = list(set(Xtrain.columns)-set(catCols))
        # Apply one-hot encoder to each column with categorical data
        print("Categorical Columns:", list(catCols))
        print("Other Columns:", otherCols)
        encoder = OneHotEncoder(
            handle_unknown=handle_unknown, sparse_output=sparse)
        catXtrainEncoded = pd.DataFrame(encoder.fit_transform(Xtrain[catCols]))
        catXtestEncoded = pd.DataFrame(encoder.transform(Xtest[catCols]))

        # One-hot encoding removed index; put it back
        catXtrainEncoded.index = Xtrain.index
        catXtestEncoded.index = Xtest.index

        # XtrainEncoded.columns = Xtrain.columns
        # XtestEncoded.columns= Xtest.columns

        # Add one-hot encoded columns to other features
        XtrainEncoded = pd.concat([Xtrain[otherCols], catXtrainEncoded], axis=1)
        XtestEncoded = pd.concat([Xtest[otherCols], catXtestEncoded], axis=1)

        # Ensure all columns have string type
        XtrainEncoded.columns = XtrainEncoded.columns.astype(str)
        XtestEncoded.columns = XtestEncoded.columns.astype(str)

        if inplace:
            self._Xtrain,self._Xtest=XtrainEncoded,XtestEncoded

        return XtrainEncoded, XtestEncoded
    def oe(self, catCols: list | str | None = None, handle_unknown: str = 'error',inplace:bool=False):
        '''
        Applying Ordinal Encoder to categorical feature columns.


        We assume you've got the good columns to be ordinal encoded.
        Bad columns mean the features in those, couldn't be found in validation/test dataframe.
        '''
        Xtrain,Xtest=self._Xtrain,self._Xtest
        # Preprocess:
        if catCols == None:
            print(
                "Warning: You didn't input argument {catcols}, so we select all object-type columns to be one-hot encoded.")
            catCols = [col for col in Xtrain.columns if Xtrain[col].dtype == "object"]
            # Would this work instead? >>>
            # catCols = Xtrain.select_dtypes(include=object).columns

        # Columns that can be safely ordinal encoded
        goodCols = [col for col in catCols if set(
            Xtest[col]).issubset(set(Xtrain[col]))]  # because sample elements from Xtest is possible not appearing in Xtrain!
        badCols = list(set(catCols)-set(goodCols))
        print(f'''
        Columns that can fit ordinal encoder: {goodCols}
        Columns that cannot fit ordinal encoder: {badCols}
        ''')
        encoder = OrdinalEncoder(handle_unknown=handle_unknown)
        # !!! We cannot use encoder.transform for Xtest dataset, why?
        XtrainEncoded = encoder.fit_transform(Xtrain)
        XtestEncoded = encoder.fit_transform(Xtest)
        if inplace:
            self._Xtrain,self._Xtest=XtrainEncoded,XtestEncoded

        return XtrainEncoded, XtestEncoded


if __name__ == "__main__":
    # def clear_last_line():
    #     sys.stdout.write('\x1b[1A')  # Move cursor up one line
    #     sys.stdout.write('\x1b[2K')  # Clear the line
    #     sys.stdout.flush()
    # while True:
    #     for i in range(1,6):
    #         clear_last_line()
    #         print(__file__," running"+'.'*i)
    #         time.sleep(1)
    pass

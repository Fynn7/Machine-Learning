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
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


def raiseTypeError(arg: object, shouldBe: type | object, origErrMsg: str | None = None) -> None:
    '''
    Should finally switch to InvalidParameterError,
    and written inside class as a methode
    But we don't wanna focus on this right now

    See: `InvalidErrorSample.txt` under this folder
    '''
    errmsg = f'【{arg}】 should be 【{shouldBe}】, not 【{type(arg)}.】'
    if origErrMsg:
        errmsg += f"Original Error Message: {str(origErrMsg)}"
    raise TypeError(errmsg)


def todf(l: list | pd.DataFrame | pd.Series) -> pd.DataFrame:
    '''
    ★ It saves A LOT OF to code!
    First try to convert l to DataFrame,
    it also automatically checked the types.
    '''
    try:
        return pd.DataFrame(l)
    except ValueError:
        raiseTypeError(l, '`DataFrame-Like`')


def toseries(l: list | pd.DataFrame | pd.Series) -> pd.DataFrame:
    '''
    ★ It saves A LOT OF to code!
    First try to convert l to Series,
    it also automatically checked the types.
    '''
    try:
        return pd.Series(l)
    except ValueError:
        raiseTypeError(l, '`Series-Like`')


# def isEmpty(df: pd.DataFrame) -> bool:
#     return todf(df).empty


# def compare(oldData: pd.DataFrame, newData: pd.DataFrame) -> pd.DataFrame:
#     return todf(oldData).compare(todf(newData))


# def exclude(df: pd.DataFrame, exclude: str | list | type) -> pd.DataFrame:
#     '''
#     ```
#     >>> exclude(df,object)

#     ```
#     '''
#     return todf(df).select_dtypes(exclude=exclude)


# def include(df: pd.DataFrame, include: str | list | type) -> pd.DataFrame:
#     '''
#     ```
#     >>> include(df,np.float64)
#     ```
#     '''
#     return todf(df).select_dtypes(include=include)


# def toNumeric(s: pd.Series, errors: str = 'coerce') -> pd.Series:
#     '''
#     if errors == coerce: the errors are transfered to NaN.
#     if errors == ignore: the errors stay their original values
#     # Example:
#     ```
#     >>> df = pd.DataFrame({'col1': ['1', '2'], 'col2': [3, 'ABC']})
#     >>> for col in df.columns:
#     >>>     df[col] = toNumeric(df[col])
#     >>> df
#         col1    col2
#     0   1       3.0
#     1   2       NaN
#     ```
#     '''
#     return pd.to_numeric(toseries(s), errors=errors)


# def scoreDataset(X: pd.DataFrame, y: pd.Series, model: str = 'RandomForestRegressor', test_size: float | int | None = None, train_size: float | int | None = None, random_state: int | None = None, **kwargs):
#     if type(model) != str:
#         raiseTypeError(model, str)
#     Xtrain, Xtest, ytrain, ytest = train_test_split(
#         todf(X), toseries(y), test_size=test_size, train_size=train_size, random_state=random_state)
#     # building up the code
#     argsComm = "("
#     for k, v in kwargs.items():
#         argsComm += f"{k}="
#         argsComm += f"{v},"
#     argsComm += "random_state=random_state)"
#     m = eval(f"{model}{argsComm}")
#     try:
#         m.fit(Xtrain, ytrain)
#     except NameError:
#         raise NameError(f"Model name `{model}` not found.")
#     p = m.predict(Xtest)
#     score = mean_absolute_error(ytest, p)
#     print(f'''
#     Model: {model}
#     Model Arguments: {kwargs}
#     MAE Score: {score}
#           ''')
#     return score


# def ohe(Xtrain: pd.DataFrame | pd.Series, Xtest: pd.DataFrame | pd.Series, catCols: list | str | None = None, handle_unknown: str = 'ignore', sparse: bool = False, **kwargs) -> tuple[pd.DataFrame, pd.Series]:
#     '''
#     Return encoded categorical features DataFrame with OneHotEncoder.

#     - X: should be the whole matrix(`pd.DataFrame()`) of the features. INCLUDING NOT-CATEGORICAL FEATURES
#     - y: should be the whole vector(`pd.Series()`) of the target feature.

#     # Example:
#     ```
#     >>> df = pd.DataFrame({'taste': ['Sweet','Sweet', 'Sour','Sweet','Sour'], 'size': ['Big','Big','Small','Medium','Small'],'int_size': [7,8,2,4,2],'color':['Red','Green','Green','Red','Green']})
#     >>> df
#         taste	size	int_size	color
#     0	Sweet	Big	    7	        Red
#     1	Sweet	Big	    8	        Green
#     2	Sour	Small	2	        Green
#     3	Sweet	Medium	4	        Red
#     4	Sour	Small	2	        Green
#     >>> X = df[['taste','size','int_size']] # or: `X = df.select_dtypes(include=object)`
#     >>> catCols = ['taste','size']
#     >>> ohe(Xtrain,Xtest, catCols = catCols)
#     (     0    1    2    3    4
#     3   0.0  1.0  0.0  1.0  0.0
#     0   0.0  1.0  1.0  0.0  0.0
#     2   1.0  0.0  0.0  0.0  1.0, # totally same as 4 because their features are totally same, so they're included in 1 same type in One Hot Encoding
#         0    1    2    3    4
#     1   0.0  1.0  1.0  0.0  0.0
#     4   1.0  0.0  0.0  0.0  1.0)
#     ```

#     # Explaination of the output:
#     For One Hot Encoding, we observe all categorical features into single 1 monolithic, whole feature that is represented by a bunch of binary digits.

#     There's 3 features and there should be 2*3*2=16<2**5 binary-formed encoded features
#     eg: for Sample 3(`df[3]`): it is encoded to 01010, which represents Sweet, Medium, Red,
#     and this 3 features are actually unique in this dataset,
#     so there are no other 01010 encoded sample as sample 3.

#     On the contrary of sample 2 and 4, their categorical features are TOTALLY same, so they were encoded totally same as well.

#     NOTE: Those Features whose cardinality low is, are fit to preprocess with OneHotEncoder.
#     Otherwise it's ok to use OrdinalEncoder
#     '''
#     Xtrain, Xtest = todf(Xtrain), todf(
#         Xtest)  # test if these are DataFrame-Like
#     if catCols == None:
#         print(
#             "Warning: You didn't input argument {catcols}, so we select all object-type columns to be one-hot encoded.")
#         catCols = Xtrain.select_dtypes(include=object).columns
#     otherCols = list(set(Xtrain.columns)-set(catCols))
#     # Apply one-hot encoder to each column with categorical data
#     print("Categorical Columns:", list(catCols))
#     print("Other Columns:", otherCols)
#     encoder = OneHotEncoder(
#         handle_unknown=handle_unknown, sparse_output=sparse)
#     catXtrainEncoded = pd.DataFrame(encoder.fit_transform(Xtrain[catCols]))
#     catXtestEncoded = pd.DataFrame(encoder.transform(Xtest[catCols]))

#     # One-hot encoding removed index; put it back
#     catXtrainEncoded.index = Xtrain.index
#     catXtestEncoded.index = Xtest.index

#     # XtrainEncoded.columns = Xtrain.columns
#     # XtestEncoded.columns= Xtest.columns

#     # Add one-hot encoded columns to other features
#     XtrainEncoded = pd.concat([Xtrain[otherCols], catXtrainEncoded], axis=1)
#     XtestEncoded = pd.concat([Xtest[otherCols], catXtestEncoded], axis=1)

#     # Ensure all columns have string type
#     XtrainEncoded.columns = XtrainEncoded.columns.astype(str)
#     XtestEncoded.columns = XtestEncoded.columns.astype(str)

#     return XtrainEncoded, XtestEncoded


# def oe(Xtrain: pd.DataFrame | pd.Series, Xtest: pd.DataFrame | pd.Series, catCols: list | str | None = None, handle_unknown: str = 'error'):
#     '''
#     Applying Ordinal Encoder to categorical feature columns.


#     We assume you've got the good columns to be ordinal encoded.
#     Bad columns mean the features in those, couldn't be found in validation/test dataframe.
#     '''
#     # Preprocess:
#     # test if these are DataFrame-Like
#     Xtrain, Xtest = todf(Xtrain), todf(Xtest)
#     if catCols == None:
#         print(
#             "Warning: You didn't input argument `catcols`, so we select all object-type columns to be one-hot encoded.")
#         catCols = [col for col in Xtrain.columns if Xtrain[col].dtype == "object"]
#         # Would this work instead? >>>
#         # catCols = Xtrain.select_dtypes(include=object).columns

#     # Columns that can be safely ordinal encoded
#     goodCols = [col for col in catCols if set(
#         Xtest[col]).issubset(set(Xtrain[col]))]  # because sample elements from Xtest is possible not appearing in Xtrain!
#     badCols = list(set(catCols)-set(goodCols))
#     print(f'''
#     Columns that can fit ordinal encoder: {goodCols}
#     Columns that cannot fit ordinal encoder: {badCols}
#     ''')
#     encoder = OrdinalEncoder(handle_unknown=handle_unknown)
#     # !!! We cannot use encoder.transform for Xtest dataset, why?
#     XtrainEncoded = encoder.fit_transform(Xtrain)
#     XtestEncoded = encoder.fit_transform(Xtest)
#     return XtrainEncoded, XtestEncoded


# def nunique(df: pd.DataFrame | pd.Series, catCols: list | str | None = None, sort: bool = False) -> dict[tuple]:
#     '''
#     Return a dict data structure, whose keys are the column name and whose values are the amount of the unique values.
#     '''
#     df = todf(df)
#     if type(catCols) == type(None):
#         print(
#             "Warning: You didn't input argument `catcols`, so we select all object-type columns.")
#         catCols = [col for col in df.columns if df[col].dtype == "object"]
#         # Would this work instead? >>>
#         # catCols = Xtrain.select_dtypes(include=object).columns
#     return {k: v for k, v in sorted(dict(zip(catCols, list(map(lambda col: df[col].nunique(), catCols)))).items(), key=lambda t: t[1])} if sort else dict(zip(catCols, list(map(lambda col: df[col].nunique(), catCols))))


# def getNanCols(df: pd.DataFrame | pd.Series, how: str = "any")->list:
#     df = todf(df)
#     print("STILL WORKING ON THE NAMEERROR BUG:df is not defined")
#     # return eval(f"[col for col in df.columns if df[col].isnull().{how}()]")
#     return [col for col in df.columns if getattr(df[col].isnull(), how)()]


class Model():
    def __init__(self, X: pd.DataFrame, y: pd.Series, model: str | None = None, train_size: float | int | None = None, test_size: float | int | None = None, random_state: int | None = None, **modelArgs) -> None:
        '''
        ```
        >>> from fynns_tool_model import *
        >>> df = pd.DataFrame({'taste': ['Sweet', 'Sweet', 'Sour', 'Sweet', 'Sour'], 'size': [
                        'Big', 'Big', 'Small', 'Medium', 'Small'], 'int_size': [7, 8, 2, 4, 2], 'color': ['Red', 'Green', 'Green', 'Red', 'Green']})
        >>> Xcols = list(set(df.columns)-set(['color']))
        >>> m = Model(df, Xcols=Xcols, ycol='color')
        >>> m.getXy()

        (   int_size  taste    size
        0         7  Sweet     Big
        1         8  Sweet     Big
        2         2   Sour   Small
        3         4  Sweet  Medium
        4         2   Sour   Small,
        0      Red
        1    Green
        2    Green
        3      Red
        4    Green
        Name: color, dtype: object)
        ```
        '''
        self.init(X, y, model=model, train_size=train_size,
                  test_size=test_size, random_state=random_state, **modelArgs)

    def __str__(self) -> str:
        return self._df.to_string()

    def __repr__(self) -> str:
        return self._df.to_string()

    def init(self, X: pd.DataFrame, y: pd.Series, model: str | None = None, train_size: float | int | None = None, test_size: float | int | None = None, random_state: int | None = None, **modelArgs):
        '''
        ★ Initalize all data. Both suitable for constructing and updating values.

        Args:
        **modelArgs: if inputted wrong args for the model, an custom exception is raised
        ```
        >>> from fynns_tool_model import *
        >>> df = pd.DataFrame({'taste': ['Sweet', 'Sweet', 'Sour', 'Sweet', 'Sour'], 'size': [
                        'Big', 'Big', 'Small', 'Medium', 'Small'], 'int_size': [7, 8, 2, 4, 2], 'color': ['Red', 'Green', 'Green', 'Red', 'Green']})
        >>> Xcols = list(set(df.columns)-set(['color']))
        >>> m = Model(df[Xcols],df['color'])
        >>> m.getXy()

        (   int_size  taste    size
        0         7  Sweet     Big
        1         8  Sweet     Big
        2         2   Sour   Small
        3         4  Sweet  Medium
        4         2   Sour   Small,
        0      Red
        1    Green
        2    Green
        3      Red
        4    Green
        Name: color, dtype: object)

        >>> from sklearn.datasets import load_breast_cancer
        >>> data=load_breast_cancer()
        >>> X=pd.DataFrame(data.data)
        >>> y=data.target
        >>> m.init(X=X,y=y)
        >>> m.getXy()

        (        0      1       2       3        4        5        6        7       8    
        0    17.99  10.38  122.80  1001.0  0.11840  0.27760  0.30010  0.14710  0.2419  \
        1    20.57  17.77  132.90  1326.0  0.08474  0.07864  0.08690  0.07017  0.1812   
        2    19.69  21.25  130.00  1203.0  0.10960  0.15990  0.19740  0.12790  0.2069   
        3    11.42  20.38   77.58   386.1  0.14250  0.28390  0.24140  0.10520  0.2597   
        4    20.29  14.34  135.10  1297.0  0.10030  0.13280  0.19800  0.10430  0.1809   
        ..     ...    ...     ...     ...      ...      ...      ...      ...     ...   
        564  21.56  22.39  142.00  1479.0  0.11100  0.11590  0.24390  0.13890  0.1726   
        565  20.13  28.25  131.20  1261.0  0.09780  0.10340  0.14400  0.09791  0.1752   
        566  16.60  28.08  108.30   858.1  0.08455  0.10230  0.09251  0.05302  0.1590   
        567  20.60  29.33  140.10  1265.0  0.11780  0.27700  0.35140  0.15200  0.2397   
        568   7.76  24.54   47.92   181.0  0.05263  0.04362  0.00000  0.00000  0.1587   

                9   ...      20     21      22      23       24       25      26   
        0    0.07871  ...  25.380  17.33  184.60  2019.0  0.16220  0.66560  0.7119  \
        1    0.05667  ...  24.990  23.41  158.80  1956.0  0.12380  0.18660  0.2416   
        2    0.05999  ...  23.570  25.53  152.50  1709.0  0.14440  0.42450  0.4504   
        3    0.09744  ...  14.910  26.50   98.87   567.7  0.20980  0.86630  0.6869   
        4    0.05883  ...  22.540  16.67  152.20  1575.0  0.13740  0.20500  0.4000   
        ..       ...  ...     ...    ...     ...     ...      ...      ...     ...   
        564  0.05623  ...  25.450  26.40  166.10  2027.0  0.14100  0.21130  0.4107   
        565  0.05533  ...  23.690  38.25  155.00  1731.0  0.11660  0.19220  0.3215   
        566  0.05648  ...  18.980  34.12  126.70  1124.0  0.11390  0.30940  0.3403   
        567  0.07016  ...  25.740  39.42  184.60  1821.0  0.16500  0.86810  0.9387   
        568  0.05884  ...   9.456  30.37   59.16   268.6  0.08996  0.06444  0.0000   
        ...
        565    0
        566    0
        567    0
        568    1
        Length: 569, dtype: int32)
        ```
        '''
        self._X = todf(X)
        self._y = toseries(y)
        # !!! MUST SET .copy() !! Otherwise self._X will just set to df (whole dataframe) and idk why
        dfcopy = self._X.copy()
        dfcopy[self._y.name] = self._y
        self._df = dfcopy

        self._Xtrain, self._Xtest, self._ytrain, self._ytest = train_test_split(
            X, y, train_size=train_size, test_size=test_size, random_state=random_state)
        self._Xtrain, self._Xtest, self._ytrain, self._ytest = todf(self._Xtrain), todf(
            self._Xtest), toseries(self._ytrain), toseries(self._ytest)
        if model == None:
            print(
                "Model has automatically set to RandomForestRegressor since you didn't input model name.")
            self._model = 'RandomForestRegressor'
        else:  # user has given a model argument
            self._model = model if type(
                model) == str else raiseTypeError(model, str)
        self._modelArgs = modelArgs  # !!!Attribute self._modelArgs USE ONLY FOR .info(), not for **modelArgs argument(implementing on other methodes). **modelArgs, aka new input, has higher priority. It's just like other methodes, we make a copy(local var) for self._XXX (private attribute) in almost every methodes. And we would like to delete it before the methode ends, if necessary.
        self._categoricalCols = list(self._df.select_dtypes(
            include=["object"]).columns)  # or [col for col in df.columns if df[col].dtype=='object']
        self._numericCols = list(
            self._df.select_dtypes(exclude=["object"]).columns)
        self._colsContainNaN = [
            col for col in dfcopy.columns if dfcopy[col].isnull().any()]
        # delete dfcopy, although local. USE FOR OUTPUTTING LOG (attrNames)
        del dfcopy
        # Mean Absolute Error score
        try:# in case the data is not clean.
            self._mae = self.mae(random_state=random_state, **modelArgs)
        except Exception:
            print("The Data Is Not Clean. Set mae score to 0.")
            self._mae=0
        # pop out these names we don't expect: 'self','attrNames','attr'
        attrNames = list(self.init.__code__.co_varnames)[1:-2]
        # print(attrNames)
        for attr in attrNames:
            # if user DID given these values to the corresponding arguments in .init()/__init__() constructing/updating methode
            if type(eval(attr)) != type(None):
                print(attr, "initiallized/updated.")

    def _validateParam(self) -> None:
        ...

    def info(self) -> None:
        for attr, v in self.__dict__.items():
            print(
                f"==========================================\n{attr.strip('_')}:\n---------------------\n{v}\n")
        print("==========================================")

    def getdf(self) -> pd.DataFrame:
        return self._df

    def getXy(self) -> tuple[pd.DataFrame, pd.Series]:
        return self._X, self._y

    def getTrainTest(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        return self._Xtrain, self._Xtest, self._ytrain, self._ytest

    def getModel(self) -> str:
        return self._model

    def getModelArgs(self) -> dict:
        return self._modelArgs

    def getMaeScore(self) -> float:
        return self._mae

    def getCategoricalCols(self) -> list:
        return self._categoricalCols

    def getNumericalCols(self) -> list:
        return self._numericCols

    def getColsContainNaN(self) -> list:
        return self._colsContainNaN

    def _update(self, **kwargs) -> None:
        '''
        You can literaly update any EXISTED attribute you want.

        If you want to update X,y:
        just input argument.
        ```
        >>> from fynns_tool_model import *
        >>> df = pd.DataFrame({'taste': ['Sweet', 'Sweet', 'Sour', 'Sweet', 'Sour'], 'size': [
                        'Big', 'Big', 'Small', 'Medium', 'Small'], 'int_size': [7, 8, 2, 4, 2], 'color': ['Red', 'Green', 'Green', 'Red', 'Green']})
        >>> Xcols = list(set(df.columns)-set(['color']))
        >>> m = Model(df, Xcols=Xcols, ycol='color')
        >>> m.getXy()

        (   int_size    size  taste
        0         7     Big  Sweet
        1         8     Big  Sweet
        2         2   Small   Sour
        3         4  Medium  Sweet
        4         2   Small   Sour,
        0      Red
        1    Green
        2    Green
        3      Red
        4    Green
        Name: color, dtype: object)

        >>> from sklearn.datasets import load_breast_cancer
        >>> data=load_breast_cancer()
        >>> X=pd.DataFrame(data.data)
        >>> y=data.target
        >>> m.update(X=X,y=y)
        >>> m.getXy()

        (        0      1       2       3        4        5        6        7       8    
        0    17.99  10.38  122.80  1001.0  0.11840  0.27760  0.30010  0.14710  0.2419  \
        1    20.57  17.77  132.90  1326.0  0.08474  0.07864  0.08690  0.07017  0.1812   
        2    19.69  21.25  130.00  1203.0  0.10960  0.15990  0.19740  0.12790  0.2069   
        3    11.42  20.38   77.58   386.1  0.14250  0.28390  0.24140  0.10520  0.2597   
        4    20.29  14.34  135.10  1297.0  0.10030  0.13280  0.19800  0.10430  0.1809   
        ..     ...    ...     ...     ...      ...      ...      ...      ...     ...   
        564  21.56  22.39  142.00  1479.0  0.11100  0.11590  0.24390  0.13890  0.1726   
        565  20.13  28.25  131.20  1261.0  0.09780  0.10340  0.14400  0.09791  0.1752   
        566  16.60  28.08  108.30   858.1  0.08455  0.10230  0.09251  0.05302  0.1590   
        567  20.60  29.33  140.10  1265.0  0.11780  0.27700  0.35140  0.15200  0.2397   
        568   7.76  24.54   47.92   181.0  0.05263  0.04362  0.00000  0.00000  0.1587   

                9   ...      20     21      22      23       24       25      26   
        0    0.07871  ...  25.380  17.33  184.60  2019.0  0.16220  0.66560  0.7119  \
        1    0.05667  ...  24.990  23.41  158.80  1956.0  0.12380  0.18660  0.2416   
        2    0.05999  ...  23.570  25.53  152.50  1709.0  0.14440  0.42450  0.4504   
        3    0.09744  ...  14.910  26.50   98.87   567.7  0.20980  0.86630  0.6869   
        4    0.05883  ...  22.540  16.67  152.20  1575.0  0.13740  0.20500  0.4000   
        ..       ...  ...     ...    ...     ...     ...      ...      ...     ...   
        564  0.05623  ...  25.450  26.40  166.10  2027.0  0.14100  0.21130  0.4107   
        565  0.05533  ...  23.690  38.25  155.00  1731.0  0.11660  0.19220  0.3215   
        566  0.05648  ...  18.980  34.12  126.70  1124.0  0.11390  0.30940  0.3403   
        567  0.07016  ...  25.740  39.42  184.60  1821.0  0.16500  0.86810  0.9387   
        568  0.05884  ...   9.456  30.37   59.16   268.6  0.08996  0.06444  0.0000   
        ...
                1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1,
                1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1,
                1, 1, 1, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1,
                1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1]))
        ```
        '''
        for k, v in kwargs.items():
            try:
                exec(f"self._{k}")  # check if the attribute exists
            except AttributeError:
                raise AttributeError(f"Attribute {k} not found.")
            exec(f"self._{k}=v")
            print("Attribute", k, "updated.")

    def mae(self, random_state: int | None = None, inplace: bool = False, **modelArgs):
        model = self._model
        Xtrain, Xtest, ytrain, ytest = self._Xtrain, self._Xtest, self._ytrain, self._ytest
        # building up the code(MODEL_NAME(ARG1=VAL1,ARG2=VAL2,...))
        argsComm = "("
        for k, v in modelArgs.items():  # extra parameters for the model object itself
            argsComm += f"{k}="
            argsComm += f"{v},"
        argsComm += "random_state=random_state)"
        try:
            m = eval(f"{model}{argsComm}")
        except TypeError as e:  # if the correct arguments are given to the corresponding model
            raise Exception(
                f'Some arguments not found. \nOriginal Error Message:\n【{e}】')
        # !!! Avoid using eval and getattr for dynamic code execution: Instead of dynamically constructing and executing code strings, it's generally safer and more readable to directly call the methods and classes.
        try:
            m.fit(Xtrain, ytrain)  # if the correct model name is given
        except NameError:
            raise NameError(f"Model name `{model}` not found.")
        p = m.predict(Xtest)
        score = mean_absolute_error(ytest, p)
        print(f'''
        Model: {model}
        Model Arguments: {modelArgs}
        MAE Score: {score}
            ''')
        if inplace:
            self._mae = score
        return score

    def ohe(self, catCols: list | str | None = None, handle_unknown: str = 'ignore', sparse: bool = False, inplace: bool = False, **kwargs) -> None | tuple[pd.DataFrame, pd.Series]:
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
        if not self._categoricalCols:  # If there is no categorical columns in this dataset.
            print(
                "There is no categorical columns in this dataset.\n So you don't need Encoder.")
            return
        df, Xtrain, Xtest = self._df, self._Xtrain, self._Xtest
        if catCols == None:
            print(
                "Warning: You didn't input argument `catcols`, so we select all object-type columns to be one-hot encoded.")
            catCols = self._categoricalCols
        # uncategorical columns
        otherCols = list(set(df.columns)-set(catCols))
        # Apply one-hot encoder to each column with categorical data
        print("All Columns:", list(df.columns))
        print("Categorical Columns:", list(catCols))
        print("Other Columns (non-categorical):", otherCols)
        encoder = OneHotEncoder(
            handle_unknown=handle_unknown, sparse_output=sparse)
        catXtrainEncoded = todf(encoder.fit_transform(Xtrain[catCols]))
        catXtestEncoded = todf(encoder.transform(Xtest[catCols]))

        # One-hot encoding removed index; put it back
        catXtrainEncoded.index = Xtrain.index
        catXtestEncoded.index = Xtest.index

        # XtrainEncoded.columns = Xtrain.columns
        # XtestEncoded.columns= Xtest.columns

        # Add one-hot encoded columns to other features
        XtrainEncoded = pd.concat(
            [Xtrain[otherCols], catXtrainEncoded], axis=1)
        XtestEncoded = pd.concat([Xtest[otherCols], catXtestEncoded], axis=1)

        # Ensure all columns have string type
        XtrainEncoded.columns = XtrainEncoded.columns.astype(str)
        XtestEncoded.columns = XtestEncoded.columns.astype(str)

        if inplace:
            self._Xtrain, self._Xtest = XtrainEncoded, XtestEncoded

        return XtrainEncoded, XtestEncoded

    def oe(self, catCols: list | str | None = None, handle_unknown: str = 'error', inplace: bool = False) -> tuple[pd.DataFrame, pd.DataFrame] | None:
        '''
        Applying Ordinal Encoder to categorical feature columns.


        We assume you've got the good columns to be ordinal encoded.
        Bad columns mean the features in those, couldn't be found in validation/test dataframe.
        '''
        if not self._categoricalCols:  # There is no categorical columns in this dataset.
            print(
                "There is no categorical columns in this dataset.\n So you don't need Encoder.")
            return
        df, Xtrain, Xtest = self._df, self._Xtrain, self._Xtest
        # Preprocess:
        if catCols == None:
            print(
                "Warning: You didn't input argument `catcols`, so we select all object-type columns to be ordinal encoded.")
            catCols = self._categoricalCols

            # Would this work instead? >>>
            # catCols = df.select_dtypes(include=object).columns

        # uncategorical columns
        otherCols = list(set(df.columns)-set(catCols))
        # Columns that can be safely ordinal encoded
        goodCols = [col for col in catCols if set(
            Xtest[col]).issubset(set(Xtrain[col]))]  # because sample elements from Xtest is possible not appearing in Xtrain!
        badCols = list(set(catCols)-set(goodCols))
        print(f'''
        All columns: {list(df.columns)}
        Categorical Columns that can fit ordinal encoder: {goodCols}
        Categorical Columns that cannot fit ordinal encoder: {badCols}
        Other columns (non categorical): {otherCols}
        ''')
        encoder = OrdinalEncoder(handle_unknown=handle_unknown)
        # !!! We cannot use encoder.transform for Xtest dataset, why?
        XtrainEncoded = encoder.fit_transform(Xtrain)
        XtestEncoded = encoder.fit_transform(Xtest)
        if inplace:
            print("Note: inplace=True will only affect _Xtrain and _Xtest.")
            self._Xtrain, self._Xtest = XtrainEncoded, XtestEncoded

        return XtrainEncoded, XtestEncoded

    def impute(self, inplace: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
        '''
        Impute Xtrain and Xcols, as in handeling missing values. (np.nan,Null)
        '''
        Xtrain, Xtest, colsContainNaN, numCols = self._Xtrain, self._Xtest, self._colsContainNaN, self._numericCols
        # columns who both contain NaN and also numeric
        numNanCols = list(set(colsContainNaN).intersection(numCols))
        if not numNanCols:
            print(
                "There is no columns who both contain NaN and also numeric.\n Don't need to be imputed.")
            return
        print("Columns who both contain NaN and also numeric:", numNanCols)
        XtrainNumNan, XtestNumNan = Xtrain[numNanCols], Xtest[numNanCols]
        si = SimpleImputer()
        imputedXtrain = pd.DataFrame(si.fit_transform(XtrainNumNan))
        imputedXtest = pd.DataFrame(si.transform(XtestNumNan))
        if inplace:
            print("Note: inplace=True will only affect _Xtrain and _Xtest.")
            self._Xtrain, self._Xtest = imputedXtrain, imputedXtest
        return imputedXtrain, imputedXtest
    
    # -----------------------------------------------------------------------------------------------------------------------------------
    # Methods without interacting(not changing values, only passing values of attributes to the corresp. methods ) with class attributes

    def transformNumCols(self, strategy: str = 'constant') -> SimpleImputer:
        '''
        We just need to impute the missing values from num cols.

        ```
        >>> 
        ```
        '''
        return SimpleImputer(strategy=strategy)

    def transformCatCols(self, imputeStrategy: str = 'most_frequent', encoder: str = 'OneHotEncoder', **encoderArgs) -> Pipeline:
        argsComm = "("
        for k, v in encoderArgs.items():  # extra parameters for the model object itself
            argsComm += f"{k}="
            argsComm += f"{v},"
        argsComm += ")"
        try:
            return Pipeline(steps=[
                ('imputer', SimpleImputer(strategy=imputeStrategy)),
                (encoder, eval(f"{encoder}{argsComm}"))])
        except TypeError as e:  # if the correct arguments are given to the corresponding model
            raise Exception(
                f'Some arguments not found. \nOriginal Error Message:\n【{e}】')
        # !!! Avoid using eval and getattr for dynamic code execution: Instead of dynamically constructing and executing code strings, it's generally safer and more readable to directly call the methods and classes.


    def preprocessor(self) -> ColumnTransformer:
        '''
        Return a ColumnTransformer for preprocessing (preprocess to handle-able data, aka. numbers).
        '''
        numCols,catCols = self._numericCols,self._categoricalCols
        numColTransformer,catColTransformer=self.transformNumCols(),self.transformCatCols()
        return ColumnTransformer(
            transformers=[
                ('numerical Transformer', numColTransformer, numCols),
                ('categorical Transformer', catColTransformer, catCols)
            ])

    def pipeline(self)->Pipeline:
        '''
        Bundle preprocessing and modeling code in a pipeline
        Return model with full steps of preprocessing+model

        ```
        ```
        '''
        model,modelArgs=self._model,self._modelArgs
        argsComm = "("
        preprocessor=self.preprocessor()
        for k, v in modelArgs.items():  # extra parameters for the model object itself
            argsComm += f"{k}="
            argsComm += f"{v},"
        argsComm += ")"
        try:

            return Pipeline(steps=[('preprocessor', preprocessor),
                              ('model', eval(f"{model}{argsComm}"))
                             ])

        except TypeError as e:  # if the correct arguments are given to the corresponding model
            raise Exception(
                f'Some arguments not found. \nOriginal Error Message:\n【{e}】')
        # !!! Avoid using eval and getattr for dynamic code execution: Instead of dynamically constructing and executing code strings, it's generally safer and more readable to directly call the methods and classes.

    def crossValidation(self) -> ...:
        pass


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

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer


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
        comm += f"{v},"
    comm = comm.strip(',') + suffix
    return comm


class Model:
    def __init__(self, X: pd.DataFrame, y: pd.Series, model: str | None = None, encoder: str | None = None, train_size: float | int | None = None, test_size: float | int | None = None, random_state: int | None = None, scoreIt: bool | None = None, modelArgs: dict | None = None, encoderArgs: dict | None = None) -> None:
        self.init(X=X, y=y, model=model, encoder=encoder, train_size=train_size,
                  test_size=test_size, random_state=random_state, scoreIt=scoreIt, modelArgs=modelArgs, encoderArgs=encoderArgs)

    def __str__(self) -> str:
        return self._df.to_string()

    def __repr__(self) -> str:
        return self._df.to_string()

    def _update(self, **attrs) -> None:
        for a, v in attrs.items():
            try:
                exec(f"self._{a}")  # check if the attribute exists
            except AttributeError:
                raise AttributeError(f"Attribute {a} not found.")
            exec(f"self._{a} = v")
            print("Attribute", a, "updated.")

    def init(self, X: pd.DataFrame, y: pd.Series, model: str | None = None, encoder: str | None = None, train_size: float | int | None = None, test_size: float | int | None = None, random_state: int | None = None,  scoreIt: bool | None = None, modelArgs: dict | None = None, encoderArgs: dict | None = None) -> None:
        self._X, self._y = todf(X), toseries(y)
        # create a copy for the whole dataframe
        dfcopy = self._X.copy()
        dfcopy[self._y.name] = self._y
        self._df = dfcopy
        # set train test split data
        self._Xtrain, self._Xtest, self._ytrain, self._ytest = train_test_split(
            X, y, train_size=train_size, test_size=test_size, random_state=random_state)

        if model == None:  # User hasn't given any values to `model`
            print(
                "Model set to `RandomForestRegressor` since no input for argument `model`.")
            self._model = 'RandomForestRegressor'
        else:  # user has set value to argument `model`
            self._model = model if type(
                model) == str else raiseTypeError(model, str)
        # set the model arguments from modelArgs
        if modelArgs == None:
            self._modelArgs = {}
        else:
            self._modelArgs = modelArgs if type(
                modelArgs) == dict else raiseTypeError(modelArgs, dict)

        if encoder == None:  # User hasn't given any values to `model`
            print(
                "Model set to `OrdinalEncoder` since no input for argument `model`.")
            self._encoder = 'OrdinalEncoder'
        else:  # user has set value to argument `encoder`
            self._encoder = encoder if type(
                encoder) == str else raiseTypeError(encoder, str)
        # set the model arguments from encoderArgs
        if encoderArgs == None:
            self._encoderArgs = {}
        else:
            self._encoderArgs = encoderArgs if type(
                encoderArgs) == dict else raiseTypeError(encoderArgs, dict)

        # set columns that only contains numerical, categorical, NaN data.
        self._numCols, self._catCols, self._nanCols = list(
            dfcopy.select_dtypes(exclude=["object"]).columns), list(dfcopy.select_dtypes(
                include=["object"]).columns), [col for col in dfcopy.columns if dfcopy[col].isnull().any()]

        self._numColsX, self._catColsX, self._nanColsX = list(
            X.select_dtypes(exclude=["object"]).columns), list(X.select_dtypes(
                include=["object"]).columns), [col for col in X.columns if X[col].isnull().any()]
        # delete dfcopy to avoid this goes into local variable `attrNames` for logging updates/initializations for the corresp. arguments
        del dfcopy
        # Determine the score of the dataset using given model. !!! The dataset should be clean!
        # self._score = self.score_dataset(
        #     random_state=random_state, modelArgs=modelArgs) if scoreIt else None
        # For update log: pop out these names we don't expect: 'self','attrNames','attr', so we slice it to [1:-2]
        attrNames = list(self.init.__code__.co_varnames)[1:-2]
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

    def getTrainTest(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        return self._Xtrain, self._Xtest, self._ytrain, self._ytest

    def getModel(self) -> str:
        return self._model

    def getModelArgs(self) -> dict:
        return self._modelArgs

    def getCatCols(self) -> list:
        return self._catCols

    def getNumCols(self) -> list:
        return self._numCols

    def getNanCols(self) -> list:
        return self._nanCols

    def getCatColsX(self) -> list:
        return self._catColsX

    def getNumColsX(self) -> list:
        return self._numColsX

    def getNanColsX(self) -> list:
        return self._nanColsX

    def getScore(self) -> float | None:
        return self._score

    # -------------------------------------------
    # Data Preprocessing

    def transformCatCols(self, imputeStrategy: str = 'most_frequent') -> Pipeline:
        '''
        Fill the missing values and encode the categorical values for ONLY FEATURE (Xtrain & Xtest) categorical columns.

        ```
        >>> from fynns_tool_model_v1 import *
        >>> df = pd.DataFrame({ 'color': ['Red', 'Green', 'Green','Green', 'Red', 'Green'],'taste': ['Sweet', np.nan,'Sweet', 'Sour', 'Sweet','Sour'], 'size': [
        >>>                   'Big', 'Big', 'Small', 'Medium',np.nan, 'Small'], 'int_size': [7, 8, 2, np.nan,4, 2]})
        >>> Xcols = list(set(df.columns)-set(['color']))
        >>> m = Model(df[Xcols],df['color'],modelArgs={'n_estimators':100})
        >>> m
        Model set to `RandomForestRegressor` since no input for argument `model`.
        Model set to `OrdinalEncoder` since no input for argument `model`.
        X initiallized/updated.
        y initiallized/updated.
        modelArgs initiallized/updated.
        taste  int_size    size  color
        0  Sweet       7.0     Big    Red
        1    NaN       8.0     Big  Green
        2  Sweet       2.0   Small  Green
        3   Sour       NaN  Medium  Green
        4  Sweet       4.0     NaN    Red
        5   Sour       2.0   Small  Green

        >>> m.getCatColsX()
        ['size', 'taste']

        >>> df=m.getdf()
        >>> todf(m.transformCatCols().fit_transform(df[m.getCatColsX()]))

        Transform Imputer Strategy: most_frequent
        Transform Encoder: OrdinalEncoder
        Encoder Arguments: {}   

            0	    1
        0	0.0	    1.0
        1	0.0	    1.0
        2	2.0	    1.0
        3	1.0	    0.0
        4	0.0	    1.0
        5	2.0	    0.0
        ```
        '''
        # df, X, y, Xtrain, Xtest, ytrain, ytest, allCatCols, nanCols = self._df, self._X, self._y, self._Xtrain, self._Xtest, self._ytrain, self._ytest, self._catCols, self._nanCols
        encoder, encoderArgs = self._encoder, self._encoderArgs
        # Create pipeline for imputing & encoding
        argsComm = buildModelComm(
            encoderArgs, prefix=f"{encoder}(", suffix=')')
        try:
            ppl = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy=imputeStrategy)),
                ('encoder', eval(f"{argsComm}"))])
        except TypeError as e:  # if the correct arguments are given to the corresponding model
            raise Exception(
                f'Some arguments not found. \nOriginal Error Message:\n【{e}】')
        print(f'''
        Transform Imputer Strategy: {imputeStrategy}
        Transform Encoder: {encoder}
        Encoder Arguments: {encoderArgs}   
        ''')
        return ppl

    def transformNumCols(self, strategy: str = 'mean') -> SimpleImputer:
        '''
        Fill the missing values for ONLY FEATURE (Xtrain & Xtest) numerical columns.

        ```
        >>> from fynns_tool_model_v1 import *
        >>> df = pd.DataFrame({ 'color': ['Red', 'Green', 'Green','Green', 'Red', 'Green'],'taste': ['Sweet', np.nan,'Sweet', 'Sour', 'Sweet','Sour'], 'size': [
        >>>                   'Big', 'Big', 'Small', 'Medium',np.nan, 'Small'], 'int_size': [7, 8, 2, np.nan,4, 2]})
        >>> Xcols = list(set(df.columns)-set(['color']))
        >>> m = Model(df[Xcols],df['color'],modelArgs={'n_estimators':100})
        >>> m
        Model set to `RandomForestRegressor` since no input for argument `model`.
        Model set to `OrdinalEncoder` since no input for argument `model`.
        X initiallized/updated.
        y initiallized/updated.
        modelArgs initiallized/updated.

        taste  int_size    size  color
        0  Sweet       7.0     Big    Red
        1    NaN       8.0     Big  Green
        2  Sweet       2.0   Small  Green
        3   Sour       NaN  Medium  Green
        4  Sweet       4.0     NaN    Red
        5   Sour       2.0   Small  Green

        >>> m.getNumColsX()
        ['int_size']

        >>> df=m.getdf()
        >>> todf(m.transformNumCols().fit_transform(df[m.getNumColsX()]))
        Transform Imputer Strategy: mean

            0
        0	7.0
        1	8.0
        2	2.0
        3	4.6                         <------------- mean value
        4	4.0
        5	2.0
        ```
        '''
        print(f'''
        Transform Imputer Strategy: {strategy}
        ''')
        return SimpleImputer(strategy=strategy)

    def preprocess(self, numColTransformer: SimpleImputer, catColTransformer: Pipeline) -> ColumnTransformer:
        '''
        >>> from fynns_tool_model_v1 import *
        >>> df = pd.DataFrame({ 'color': ['Red', 'Green', 'Green','Green', 'Red', 'Green'],'taste': ['Sweet', np.nan,'Sweet', 'Sour', 'Sweet','Sour'], 'size': [
        >>>                   'Big', 'Big', 'Small', 'Medium',np.nan, 'Small'], 'int_size': [7, 8, 2, np.nan,4, 2]})
        >>> Xcols = list(set(df.columns)-set(['color']))
        >>> m = Model(df[Xcols],df['color'],modelArgs={'n_estimators':100})
        >>> m
        Model set to `RandomForestRegressor` since no input for argument `model`.
        Model set to `OrdinalEncoder` since no input for argument `model`.
        X initiallized/updated.
        y initiallized/updated.
        modelArgs initiallized/updated.

        taste  int_size    size  color
        0  Sweet       7.0     Big    Red
        1    NaN       8.0     Big  Green
        2  Sweet       2.0   Small  Green
        3   Sour       NaN  Medium  Green
        4  Sweet       4.0     NaN    Red
        5   Sour       2.0   Small  Green


        >>> X=m.getXy()[0]
        >>> nct,cct=m.transformNumCols(),m.transformCatCols()
        >>> m.preprocess(nct,cct).fit_transform(X)

        Transform Imputer Strategy: mean

        Transform Imputer Strategy: most_frequent
        Transform Encoder: OrdinalEncoder
        Encoder Arguments: {}   

        array([[7. , 1. , 0. ],
            [8. , 1. , 0. ],
            [2. , 1. , 2. ],
            [4.6, 0. , 1. ],
            [4. , 1. , 0. ],
            [2. , 0. , 2. ]])
        '''
        numColsX, catColsX = self._numColsX, self._catColsX
        return ColumnTransformer(
            transformers=[
                ('numerical Transformer', numColTransformer, numColsX),
                ('categorical Transformer', catColTransformer, catColsX)
            ])

    def pipeline(self, preprocessor: ColumnTransformer) -> Pipeline:
        '''
        Bundle preprocessing and modeling for in a pipeline for DIRECTLY TRAINING DATA.

        Only fit for regression problems. (aka. target series should be continuous and numerical.)
        ```
        >>> from fynns_tool_model_v1 import *
        >>> df = pd.DataFrame({ 'color': ['Red', 'Green', 'Green','Green', 'Red', 'Green'],'taste': ['Sweet', np.nan,'Sweet', 'Sour', 'Sweet','Sour'], 'size': [
        >>>                 'Big', 'Big', 'Small', 'Medium',np.nan, 'Small'], 'int_size': [7, 8, 2, 5,4, 2]})
        >>> Xcols = list(set(df.columns)-set(['int_size']))
        >>> m = Model(df[Xcols],df['int_size'],modelArgs={'n_estimators':100})
        >>> m

        Model set to `RandomForestRegressor` since no input for argument `model`.
        Model set to `OrdinalEncoder` since no input for argument `model`.
        X initiallized/updated.
        y initiallized/updated.
        modelArgs initiallized/updated.
            size  color  taste  int_size
        0     Big    Red  Sweet         7
        1     Big  Green    NaN         8
        2   Small  Green  Sweet         2
        3  Medium  Green   Sour         5
        4     NaN    Red  Sweet         4
        5   Small  Green   Sour         2


        >>> X,y=m.getXy()
        >>> Xtrain,Xtest,ytrain,ytest=m.getTrainTest()
        >>> nct,cct=m.transformNumCols(),m.transformCatCols()
        >>> preprocessor=m.preprocess(nct,cct)
        >>> trained=m.pipeline(preprocessor).fit(Xtrain,ytrain)
        >>> preds=trained.predict(Xtest)
        >>> preds

        Transform Imputer Strategy: mean


        Transform Imputer Strategy: most_frequent
        Transform Encoder: OrdinalEncoder
        Encoder Arguments: {}   

        array([5.12, 3.03])

        >>> score = mean_absolute_error(ytest, preds)
        >>> score
        ```
        '''
        model, modelArgs = self._model, self._modelArgs
        modelArgsComm = buildModelComm(
            modelArgs, prefix=f"{model}(", suffix=')')
        try:
            return Pipeline(steps=[('preprocessor', preprocessor),
                                   ('model', eval(f"{modelArgsComm}"))
                                   ])

        except TypeError as e:  # if the correct arguments are given to the corresponding model
            raise Exception(
                f'Some arguments not found. \nOriginal Error Message:\n【{e}】')

    def score_dataset(self) -> float:
        ...

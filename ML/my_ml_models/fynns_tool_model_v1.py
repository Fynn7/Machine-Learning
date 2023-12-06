import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
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
        if type(v)==str: # if the value of the argument is str type, like handle_unknown='ignore'
            comm += f"\'{v}\',"
        else:
            comm += f"{v},"
    comm = comm.strip(',') + suffix
    return comm


class RegressionModel:
    def __init__(self, X: pd.DataFrame, y: pd.Series, model: str | None = None, encoder: str | None = None, train_size: float | int | None = None, test_size: float | int | None = None, train_test_split_random_state: int | None = None, scoreIt: bool | None = None, modelArgs: dict | None = None, encoderArgs: dict | None = None) -> None:
        self.init(X=X, y=y, model=model, encoder=encoder, train_size=train_size,
                  test_size=test_size, train_test_split_random_state=train_test_split_random_state, scoreIt=scoreIt, modelArgs=modelArgs, encoderArgs=encoderArgs)

    def __str__(self) -> str:
        return self._df.to_string()

    def __repr__(self) -> str:
        return self._df.to_string()

    def _update(self, **attrs) -> None:
        '''
        Only affect attribute that is brought as _update methode's argument.
        Any other attribute WON'T BE CHANGED chainly.
        '''
        for a, v in attrs.items():
            try:
                exec(f"self._{a}")  # check if the attribute exists
            except AttributeError:
                raise AttributeError(f"Attribute {a} not found.")
            exec(f"self._{a} = v")
            print("Attribute", a, "updated.")

    def init(self, X: pd.DataFrame | None = None, y: pd.Series | None = None, model: str | None = None, encoder: str | None = None, train_size: float | int | None = None, test_size: float | int | None = None, train_test_split_random_state: int | None = None,  scoreIt: bool | None = None, modelArgs: dict | None = None, encoderArgs: dict | None = None) -> None:
        '''
        Initial all relevant data if you call this init method.
        Aka. if you init only y, the train-test-split and other data will be changed as well.

        # P.S. NOTE: Not forget to put `random_state` argument inside `modelArgs` & `encoderArgs` if you tent to put this argument in training data ( argument: `train_test_split_random_state`)
        '''
        if type(X) != type(None):
            self._X = todf(X)
        if type(y) != type(None):
            self._y = toseries(y)
        # create a copy for the whole dataframe
        dfcopy = self._X.copy()
        dfcopy[self._y.name] = self._y
        self._df = dfcopy
        # set train test split data
        self._Xtrain, self._Xtest, self._ytrain, self._ytest = train_test_split(
            X, y, train_size=train_size, test_size=test_size, random_state=train_test_split_random_state)
        self._trainTestSplitArgs = {
            'train_size': train_size,
            'test_size': test_size,
            'random_state': train_test_split_random_state,
        }
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
            print(
                f"`modelArgs` set to {self._modelArgs} since no input for argument `modelArgs`.")
        else:
            self._modelArgs = modelArgs if type(
                modelArgs) == dict else raiseTypeError(modelArgs, dict)

        if encoder == None:  # User hasn't given any values to `model`
            print(
                "Encoder set to `OrdinalEncoder` since no input for argument `encoder`.")
            self._encoder = 'OrdinalEncoder'
        else:  # user has set value to argument `encoder`
            self._encoder = encoder if type(
                encoder) == str else raiseTypeError(encoder, str)
        # set the model arguments from encoderArgs
        if encoderArgs == None:
            self._encoderArgs = {}
            print(
                f"`encoderArgs` set to {self._encoderArgs} since no input for argument `encoderArgs`.")
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
    # Handling missing values (impute) & Encoding categorical columns (encoder)
    # => preprocessing (class <ColumnTransformer>)
    # => autoPipeline

    def transformCatCols(self, imputeStrategy: str = 'most_frequent') -> Pipeline:
        '''
        Fill the missing values and encode the categorical values for ONLY FEATURE (Xtrain & Xtest) categorical columns.

        ### NOTE: If Xtrain contains numerical columns, and it's used in fit_transform(), the numerical columns will be observed AS CATEGORICAL COLUMNS!
        ### JUST LIKE IN THIS EXAMPLE: (it won't raise Error if it counters numerical columns while transforming categorical columns!)
        ```
        >>> from fynns_tool_model_v1 import *
        >>> df = pd.DataFrame({ 'color': ['Red', 'Green', 'Green','Green', 'Red', 'Green'],'int_gone_bad':[1,0,np.nan,0,0,0],'taste': ['Sweet', np.nan,'Sweet', 'Sour', 'Sweet','Sour'], 'size': [
        >>>                   'Big', 'Big', 'Small', 'Medium',np.nan, 'Small'], 'int_size': [7, 8, 2, 5,4, np.nan]})
        >>> Xcols = list(set(df.columns)-set(['taste']))
        >>> m = RegressionModel(df[Xcols],df['taste'],modelArgs={'n_estimators':100})
        >>> m

        Model set to `RandomForestRegressor` since no input for argument `model`.
        Model set to `OrdinalEncoder` since no input for argument `model`.
        X initiallized/updated.
        y initiallized/updated.
        modelArgs initiallized/updated.
        color  int_size    size  int_gone_bad  taste
        0    Red       7.0     Big           1.0  Sweet
        1  Green       8.0     Big           0.0    NaN
        2  Green       2.0   Small           NaN  Sweet
        3  Green       5.0  Medium           0.0   Sour
        4    Red       4.0     NaN           0.0  Sweet
        5  Green       NaN   Small           0.0   Sour

        >>> X,y=m.getXy()
        >>> X
        color	int_size	int_gone_bad	size
0	Red	7.0	1.0	Big
1	Green	8.0	0.0	Big
2	Green	2.0	NaN	Small
3	Green	5.0	0.0	Medium
4	Red	4.0	0.0	NaN
5	Green	NaN	0.0	Small

        >>> nct,cct=m.transformNumCols(),m.transformCatCols()
        >>> transformed=todf(cct.fit_transform(X,y))
        >>> transformed.index,transformed.columns=X.index,X.columns
        >>> transformed

        Transform Categorical Features Imputer Strategy: mean


        Transform Numerical Features Imputer Strategy: most_frequent
        Transform Encoder: OrdinalEncoder
        Encoder Arguments: {}   

color	int_size	int_gone_bad	size
0	1.0	3.0	1.0	0.0
1	0.0	4.0	0.0	0.0
2	0.0	0.0	0.0	2.0
3	0.0	2.0	0.0	1.0
4	1.0	1.0	0.0	0.0
5	0.0	0.0	0.0	2.0
        ```
        '''
        # df, X, y, Xtrain, Xtest, ytrain, ytest, allCatCols, nanCols = self._df, self._X, self._y, self._Xtrain, self._Xtest, self._ytrain, self._ytest, self._catCols, self._nanCols
        encoder, encoderArgs = self._encoder, self._encoderArgs
        # Create pipeline for imputing & encoding
        argsComm = buildModelComm(
            encoderArgs, prefix=f"{encoder}(", suffix=')')
        print("Model Code Built As:",argsComm)
        try:
            cct = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy=imputeStrategy)),
                ('encoder', eval(f"{argsComm}"))])
        except TypeError as e:  # if the correct arguments are given to the corresponding model
            raise Exception(
                f'Some arguments not found. \nOriginal Error Message:\n【{e}】')
        print(f'''
        Transform Numerical Features Imputer Strategy: {imputeStrategy}
        Transform Encoder: {encoder}
        Encoder Arguments: {encoderArgs}   
        ''')

        X, y = self._X, self._y
        transformed = todf(cct.fit_transform(X, y))
        transformed.index, transformed.columns = X.index, X.columns
        print(f'''
              transformed:
              {transformed}
              ''')
        return cct

    def transformNumCols(self, strategy: str = 'mean') -> SimpleImputer:
        '''
        Only filling up the missing values for ONLY FEATURE (Xtrain & Xtest) numerical columns.


        ## Only fit for regression problems. (aka. target series should be continuous and numerical.)
        ## NOT FIT IF THE TARGET COLUMN IS CATEGORICAL. MUST PREPROCESS y Column first. (e.g. encode it first)

        ```
        >>> from fynns_tool_model_v1 import *
        >>> df = pd.DataFrame({ 'color': ['Red', 'Green', 'Green','Green', 'Red', 'Green'],'int_gone_bad':[1,0,np.nan,0,0,0],'taste': ['Sweet', np.nan,'Sweet', 'Sour', 'Sweet','Sour'], 'size': [
        >>>                   'Big', 'Big', 'Small', 'Medium',np.nan, 'Small'], 'int_size': [7, 8, 2, 5,4, np.nan]})
        >>> Xcols = list(set(df.columns)-set(['taste']))
        >>> m = RegressionModel(df[Xcols],df['taste'],modelArgs={'n_estimators':100})
        >>> m

        Model set to `RandomForestRegressor` since no input for argument `model`.
        Model set to `OrdinalEncoder` since no input for argument `model`.
        X initiallized/updated.
        y initiallized/updated.
        modelArgs initiallized/updated.
        color  int_size    size  int_gone_bad  taste
        0    Red       7.0     Big           1.0  Sweet
        1  Green       8.0     Big           0.0    NaN
        2  Green       2.0   Small           NaN  Sweet
        3  Green       5.0  Medium           0.0   Sour
        4    Red       4.0     NaN           0.0  Sweet
        5  Green       NaN   Small           0.0   Sour


        >>> X,y=m.getXy()
        >>> Xtrain,Xtest,ytrain,ytest=m.getTrainTest()
        >>> Xtrain
            color	int_size	size	int_gone_bad
        4	Red	    4.0	        NaN	            0.0
        5	Green	NaN	        Small	        0.0
        2	Green	2.0	        Small	        NaN
        1	Green	8.0	        Big	            0.0

        >>> numColsX=m.getNumColsX()
        >>> numColsX

        ['int_gone_bad', 'int_size']


        >>> nct,cct=m.transformNumCols(),m.transformCatCols()
        >>> transformed=todf(nct.fit_transform(X[numColsX],y))
        >>> transformed.index,transformed.columns=X[numColsX].index,X[numColsX].columns
        >>> transformed


        Transform Categorical Features Imputer Strategy: mean


        Transform Numerical Features Imputer Strategy: most_frequent
        Transform Encoder: OrdinalEncoder
        Encoder Arguments: {}   

int_gone_bad	int_size
0	1.0	7.0
1	0.0	8.0
2	0.2	2.0
3	0.0	5.0
4	0.0	4.0
5	0.0	5.2
        ```
        '''
        print(f'''
        Transform Categorical Features Imputer Strategy: {strategy}
        ''')
        return SimpleImputer(strategy=strategy)

    def preprocessX(self, numColTransformer: SimpleImputer, catColTransformer: Pipeline) -> ColumnTransformer:
        '''

        ```
        >>> from fynns_tool_model_v1 import *
        >>> df = pd.DataFrame({ 'color': ['Red', 'Green', 'Green','Green', 'Red', 'Green'],'int_gone_bad':[1,0,np.nan,0,0,0],'taste': ['Sweet', np.nan,'Sweet', 'Sour', 'Sweet','Sour'], 'size': [
        >>>                   'Big', 'Big', 'Small', 'Medium',np.nan, 'Small'], 'int_size': [7, 8, 2, 5,4, np.nan]})
        >>> Xcols = list(set(df.columns)-set(['taste']))
        >>> m = RegressionModel(df[Xcols],df['taste'],modelArgs={'n_estimators':100},random_state=42)
        >>> m

Model set to `RandomForestRegressor` since no input for argument `model`.
Model set to `OrdinalEncoder` since no input for argument `model`.
X initiallized/updated.
y initiallized/updated.
random_state initiallized/updated.
modelArgs initiallized/updated.
   color  int_size    size  int_gone_bad  taste
0    Red       7.0     Big           1.0  Sweet
1  Green       8.0     Big           0.0    NaN
2  Green       2.0   Small           NaN  Sweet
3  Green       5.0  Medium           0.0   Sour
4    Red       4.0     NaN           0.0  Sweet
5  Green       NaN   Small           0.0   Sour


        >>> X,y=m.getXy()
        >>> Xtrain,Xtest,ytrain,ytest=m.getTrainTest()
        >>> Xtrain

        color	int_size	size	int_gone_bad
5	Green	NaN	Small	0.0
2	Green	2.0	Small	NaN
4	Red	4.0	NaN	0.0
3	Green	5.0	Medium	0.0

        >>> numColsX,catColsX=m.getNumColsX(),m.getCatColsX()
        >>> numColsX,catColsX

(['int_size', 'int_gone_bad'], ['color', 'size'])

        >>> nct,cct=m.transformNumCols(),m.transformCatCols()
        >>> preprocessor=m.preprocess(numColTransformer=nct,catColTransformer=cct)
        >>> preprocessed=todf(preprocessor.fit_transform(X,y))
        >>> preprocessed.index,preprocessed.columns=X.index,list(X[numColsX].columns)+list(X[catColsX].columns) # according the order of the pipeline inside methode `.preprocess()`, we first append num columns' names and then cat cols' names
        >>> preprocessed

        Transform Categorical Features Imputer Strategy: mean


        Transform Numerical Features Imputer Strategy: most_frequent
        Transform Encoder: OrdinalEncoder
        Encoder Arguments: {}   

int_gone_bad	int_size	color	size
0	1.0	7.0	1.0	0.0
1	0.0	8.0	0.0	0.0
2	0.2	2.0	0.0	2.0
3	0.0	5.0	0.0	1.0
4	0.0	4.0	1.0	0.0
5	0.0	5.2	0.0	2.0
        ```
        '''
        numColsX, catColsX = self._numColsX, self._catColsX
        transformers = [
            ('numerical Transformer X', numColTransformer, numColsX),
            ('categorical Transformer X', catColTransformer, catColsX)
        ]
        return ColumnTransformer(
            transformers=transformers
        )

    # !!!not valid y feature, should transform it first!!!
    def cleanCatY(self) -> pd.Series:
        '''
        ```
        X=m.getXy()[0]
        transformedY=m.cleanCatY()
        m.init(X=X,y=transformedY)
        ```
        '''
        y = self._y
        catColsTransformer = self.transformCatCols()
        transformedY = todf(catColsTransformer.fit_transform(todf(y)))
        # transformedY.index,transformedY.columns=y.index,[y.name]
        transformedY = transformedY[0]  # toseries
        transformedY.index, transformedY.name = y.index, y.name
        return transformedY

    def pipeline(self, preprocessorX: ColumnTransformer) -> Pipeline:
        '''
        Bundle preprocessing and modeling for in a pipeline for DIRECTLY TRAINING DATA.

        - preprocess_y
            True: We first preprocess X, then preorcess y in the pipeline (WE EXPECT y column is CATEGORICAL and CONTAINS MISSING VALUES)
            False: Only X will be preprocessed. (avoid y being preprocessed, WE EXPECT y column is clean, AKA without missing values and numerical)

        ```
        from fynns_tool_model_v1 import *
df = pd.DataFrame({ 'color': ['Red', 'Green', 'Green','Green', 'Red', 'Green'],'int_gone_bad':[1,0,np.nan,0,0,0],'taste': ['Sweet', np.nan,'Sweet', 'Sour', 'Sweet','Sour'], 'size': [
                  'Big', 'Big', 'Small', 'Medium',np.nan, 'Small'], 'int_size': [7, 8, 2, 5,4, np.nan]})
Xcols = list(set(df.columns)-set(['taste']))
m = RegressionModel(df[Xcols],df['taste'],modelArgs={'n_estimators':100,'random_state':42})
m


Model set to `RandomForestRegressor` since no input for argument `model`.
Model set to `OrdinalEncoder` since no input for argument `model`.
X initiallized/updated.
y initiallized/updated.
modelArgs initiallized/updated.
   color  int_size  int_gone_bad    size  taste
0    Red       7.0           1.0     Big  Sweet
1  Green       8.0           0.0     Big    NaN
2  Green       2.0           NaN   Small  Sweet
3  Green       5.0           0.0  Medium   Sour
4    Red       4.0           0.0     NaN  Sweet
5  Green       NaN           0.0   Small   Sour


X,y=m.getXy()
X


color	int_size	int_gone_bad	size
0	Red	7.0	1.0	Big
1	Green	8.0	0.0	Big
2	Green	2.0	NaN	Small
3	Green	5.0	0.0	Medium
4	Red	4.0	0.0	NaN
5	Green	NaN	0.0	Small



numColsX,catColsX=m.getNumColsX(),m.getCatColsX()
numColsX,catColsX

(['int_size', 'int_gone_bad'], ['color', 'size'])


Xtrain,Xtest,ytrain,ytest=m.getTrainTest()
Xtrain,ytrain

(   color  int_size  int_gone_bad   size
 2  Green       2.0           NaN  Small
 5  Green       NaN           0.0  Small
 1  Green       8.0           0.0    Big
 4    Red       4.0           0.0    NaN,
 2    Sweet
 5     Sour
 1      NaN
 4    Sweet
 Name: taste, dtype: object)


 nct,cct=m.transformNumCols(),m.transformCatCols()
preprocessorX=m.preprocessX(numColTransformer=nct,catColTransformer=cct)
transformedy=todf(cct.fit_transform(todf(y)))
transformedy.index,transformedy.columns=y.index,[y.name]
model=m.pipeline(preprocessorX)
trainedModel=model.fit(X,transformedy)
trainedModel.predict(Xtest)



        Transform Categorical Features Imputer Strategy: mean


        Transform Numerical Features Imputer Strategy: most_frequent
        Transform Encoder: OrdinalEncoder
        Encoder Arguments: {}   

        array([0.29, 0.93])



        Xtest,ytest


        (   color  int_size  int_gone_bad    size
 3  Green       5.0           0.0  Medium
 0    Red       7.0           1.0     Big,
 3     Sour
 0    Sweet
 Name: taste, dtype: object)
        ```
        '''
        model, modelArgs = self._model, self._modelArgs
        modelArgsComm = buildModelComm(
            modelArgs, prefix=f"{model}(", suffix=')')
        pplSteps = [('preprocessorX', preprocessorX),
                    ('model', eval(f"{modelArgsComm}"))
                    ]

        try:
            return Pipeline(steps=pplSteps)

        except TypeError as e:  # if the correct arguments are given to the corresponding model
            raise Exception(
                f'Some arguments not found. \nOriginal Error Message:\n【{e}】')

    def autoPipeline(self, transformNumStrategy: str = 'mean', transformCatColImputeStrategy: str = 'most_frequent') -> tuple[Pipeline, float]:
        '''
        Bundle methodes: 
        - transformNumCols
        - transformCatCols
        - preprocess
        - pipeline
        # Must set `m.encoderArgs={'handle_unknown':'ignore'}` while you have a small dataset,
        to prevent test samples that HAVEN'T SHOWN UP IN THE train samples:
        
        Original ChatGPT Solution: 

        增加样本量： 如果可能的话，尝试增加数据集的样本量，这样模型就能更好地捕捉到所有类别，并且不同的随机种子在数据集的不同划分下会更加稳定。

        调整训练集和测试集的划分方式： 可以尝试使用交叉验证，或者手动指定训练集和测试集的样本，而不是依赖于 train_test_split 的随机划分。

        检查数据质量： 仔细检查数据，确保没有不一致的地方，例如空格、数据类型不匹配等。

        处理未知类别： 如前所述，可以使用编码器的 handle_unknown 参数来处理未知类别，以防止在测试集中出现训练集中没有的类别。
            
        **确保一致的数据类型：**确保列的数据类型适用于分类变量。如果该列应包含分类数据，请确保其为 category 类型。
        `df['column_name'] = df['column_name'].astype('category')`

        
        ```
from fynns_tool_model_v1 import *
df = pd.DataFrame({ 'color': ['Red', 'Green', 'Green','Green', 'Red', 'Green'],'int_gone_bad':[1,0,np.nan,0,0,0],'taste': ['Sweet', np.nan,'Sweet', 'Sour', 'Sweet','Sour'], 'size': [
                  'Big', 'Big', 'Small', 'Medium',np.nan, 'Small'], 'int_size': [7, 8, 2, 5,4, np.nan]})
Xcols = list(set(df.columns)-set(['taste']))
m = RegressionModel(df[Xcols],df['taste'],modelArgs={'n_estimators':100,'random_state':42})
m

Model set to `RandomForestRegressor` since no input for argument `model`.
Model set to `OrdinalEncoder` since no input for argument `model`.
X initiallized/updated.
y initiallized/updated.
modelArgs initiallized/updated.
   int_size  color    size  int_gone_bad  taste
0       7.0    Red     Big           1.0  Sweet
1       8.0  Green     Big           0.0    NaN
2       2.0  Green   Small           NaN  Sweet
3       5.0  Green  Medium           0.0   Sour
4       4.0    Red     NaN           0.0  Sweet
5       NaN  Green   Small           0.0   Sour

numColsX,catColsX=m.getNumColsX(),m.getCatColsX()
numColsX,catColsX

(['int_size', 'int_gone_bad'], ['color', 'size'])


X,y=m.getXy()
y

0    1.0
1    1.0
2    1.0
3    0.0
4    1.0
5    0.0
Name: taste, dtype: float64


cleanedy=m.cleanCatY()

m.init(X=X,y=cleanedy)
ap=m.autoPipeline()
Xtrain,Xtest,ytrain,ytest=m.getTrainTest()
ap



        Transform Numerical Features Imputer Strategy: most_frequent
        Transform Encoder: OrdinalEncoder
        Encoder Arguments: {}   


              transformed:
                 int_size  color  size  int_gone_bad
0       3.0    1.0   0.0           1.0
1       4.0    0.0   0.0           0.0
2       0.0    0.0   2.0           0.0
3       2.0    0.0   1.0           0.0
4       1.0    1.0   0.0           0.0
5       0.0    0.0   2.0           0.0

Model set to `RandomForestRegressor` since no input for argument `model`.
Model set to `OrdinalEncoder` since no input for argument `model`.
X initiallized/updated.
y initiallized/updated.

        Transform Categorical Features Imputer Strategy: mean


        Transform Numerical Features Imputer Strategy: most_frequent
        Transform Encoder: OrdinalEncoder
        Encoder Arguments: {}   


              transformed:
                 int_size  color  size  int_gone_bad
0       3.0    1.0   0.0           1.0
1       4.0    0.0   0.0           0.0
2       0.0    0.0   2.0           0.0
3       2.0    0.0   1.0           0.0
4       1.0    1.0   0.0           0.0
5       0.0    0.0   2.0           0.0

preprocessed_X:
    int_size  int_gone_bad  color  size
0       7.0           1.0    1.0   0.0
1       8.0           0.0    0.0   0.0
2       2.0           0.2    0.0   2.0
3       5.0           0.0    0.0   1.0
4       4.0           0.0    1.0   0.0
5       5.2           0.0    0.0   2.0
ypred:
       0
0  0.62
1  0.40


(Pipeline(steps=[('preprocessorX',
                  ColumnTransformer(transformers=[('numerical Transformer X',
                                                   SimpleImputer(),
                                                   ['int_size', 'int_gone_bad']),
                                                  ('categorical Transformer X',
                                                   Pipeline(steps=[('imputer',
                                                                    SimpleImputer(strategy='most_frequent')),
                                                                   ('encoder',
                                                                    OrdinalEncoder())]),
                                                   ['color', 'size'])])),
                 ('model', RandomForestRegressor())]),
 0.39)

        ```
        '''

        # First we get X,y and split them as a complete dataframe & pd.Series
        X, y = self._X, self._y
        numColsX, catColsX = self._numColsX, self._catColsX
        # get train_test_split arguments from the model we saved from before
        train_size, test_size, random_state = self._trainTestSplitArgs[
            'train_size'], self._trainTestSplitArgs['test_size'], self._trainTestSplitArgs['random_state']
        # get 2 transformers
        nct, cct = self.transformNumCols(strategy=transformNumStrategy), self.transformCatCols(
            imputeStrategy=transformCatColImputeStrategy)
        # get the preprocessor with 2 transformers and automatically their arguments as well
        preprocessorX = self.preprocessX(nct, cct)
        # get transformed/preprocessed X as dataframe
        preprocessed_X = todf(preprocessorX.fit_transform(X, y))
        # put the columns' name back to transformed X dataframe, as we know the preprocessor has the order: first impute numerical (so we add numcols), then handle categorical (then add catcols)
        preprocessed_X.columns = numColsX+catColsX
        # Ensure all columns have string type
        preprocessed_X.columns = preprocessed_X.columns.astype(str)
        preprocessed_X.columns = preprocessed_X.columns.astype(str)
        print("preprocessed_X:\n", preprocessed_X)
        # bundle preprocessor as a pipeline, bring in the model(likely RandomForestRegressor) that was initiallized before in the model attribute
        model = self.pipeline(preprocessorX)
        # ! split a new part of train test, using arguments as given before, as when the model initiallized,  BUT WE DON'T SAVE THESE
        Xtrain, Xtest, ytrain, ytest = train_test_split(
            preprocessed_X, y, train_size=train_size, test_size=test_size, random_state=random_state)
        print("Xtrain =\n",Xtrain)
        print("Xtest =\n",Xtest)
        # fit/train the empty model with the new Xtrain and ytrain dataset
        try:
            trained = model.fit(Xtrain, ytrain)
        except ValueError as e:
            raise Exception(f"The y (aka. target) feature column is maybe not clean. It either contains missing values, or it is categorical, non numerical.\n Try use these code before creating autoPipeline:\n\ncleanedy=m.cleanCatY()\nm.autoPipeline()\n\nOriginal Error Message\n>>> 【{e}】")
        try:
            ypred = trained.predict(Xtest)  # should be a y prediction
            # print the prediction as dataframe
            print("ypred:\n", todf(ypred))
        except ValueError as e:
            raise Exception(f"Found categorical value in test samples that didn't show up in the training dataset value.\nTry Cross Validation, instead of randomly train_test_split, or try a bigger dataset.\nOr try set `handle_unknown` parameter for encoder like OneHotEncoder as 'ignore'\n\nOriginal Error Message\n>>> 【{e}】\n\nThat `Found unknown categories [XXX] in column X during transform`, the `column X` is the categorical column index. Check all categorical columns names' list and find the column name under corresponding index. (e.g. column 1 is the 2rd categorical columns in m._catColsX)")
        # score the prediction
        mae = mean_absolute_error(ytest, ypred)
        return model, mae

        Xtrain, Xtest, ytrain, ytest, numColsX, catColsX, nct, cct = self.getTrainTest(), self._numColsX, self._catColsX, self.transformNumCols(strategy=transformNumStrategy), self.transformCatCols(
            imputeStrategy=transformCatColImputeStrategy)
        nct, cct = self.transformNumCols(strategy=transformNumStrategy), self.transformCatCols(
            imputeStrategy=transformCatColImputeStrategy)
        preprocessorX = self.preprocessX(nct, cct)
        ppl = self.pipeline(preprocessorX)
        try:
            model = ppl.fit(Xtrain, ytrain)
        except ValueError as e:
            raise Exception(
                f"Maybe your y column (aka. target feature) is not valid. Either it contains NaN, or it's not clean.\nTry to use these code first to clean y feature:\ncleanedy=m.cleanCatY()\nm.init(X=X,y=cleanedy)\nOriginal Error Message:\n{e}")

        # !!!We should also preprocess Xtest as well:
        Xtest = todf(preprocessorX.fit_transform(Xtest, ytest))
        Xtest.columns = numColsX+catColsX
        print("Xtest:\n", Xtest)

        pred = model.predict(Xtest)
        maeScore=mean_absolute_error(ytest,pred)
        return model,maeScore

    # -------------------------------------------
    # Cross Validation

    def crossValScore(self, pipeline: Pipeline, X: pd.DataFrame, y: pd.Series,
                      cv=5,
                      scoring='neg_mean_absolute_error') -> ...:
        return

    def score_dataset(self) -> float:
        ...

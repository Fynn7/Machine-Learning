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

    def init(self, X: pd.DataFrame, y: pd.Series, model: str | None = None, encoder: str | None = None, train_size: float | int | None = None, test_size: float | int | None = None, train_test_split_random_state: int | None = None,  scoreIt: bool | None = None, modelArgs: dict | None = None, encoderArgs: dict | None = None) -> None:
        '''
        Initial all relevant data if you call this init method.
        Aka. if you init only y, the train-test-split and other data will be changed as well.

        # P.S. NOTE: Not forget to put `random_state` argument inside `modelArgs` & `encoderArgs` if you tent to put this argument in training data ( argument: `train_test_split_random_state`)
        '''
        self._X, self._y = todf(X), toseries(y)
        # create a copy for the whole dataframe
        dfcopy = self._X.copy()
        dfcopy[self._y.name] = self._y
        self._df = dfcopy
        # set train test split data
        self._Xtrain, self._Xtest, self._ytrain, self._ytest = train_test_split(
            X, y, train_size=train_size, test_size=test_size, random_state=train_test_split_random_state)

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
        try:
            ppl = Pipeline(steps=[
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
        return ppl

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
        transformers=[
                ('numerical Transformer X', numColTransformer, numColsX),
                ('categorical Transformer X', catColTransformer, catColsX)
            ]
        return ColumnTransformer(
            transformers=transformers
            )


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
ppl=m.pipeline(preprocessorX)
model=ppl.fit(X,transformedy)
model.predict(Xtest)



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
        pplSteps=[('preprocessorX', preprocessorX),
                                   ('model', eval(f"{modelArgsComm}"))
                                   ]

        try:
            return Pipeline(steps=pplSteps)

        except TypeError as e:  # if the correct arguments are given to the corresponding model
            raise Exception(
                f'Some arguments not found. \nOriginal Error Message:\n【{e}】')

    # def autoPipeline(self, transformNumStrategy: str = 'mean', transformCatColImputeStrategy: str = 'most_frequent',preprocess_y:bool=False) -> Pipeline:
    #     '''
    #     Bundle methodes: 
    #     - transformNumCols
    #     - transformCatCols
    #     - preprocess
    #     - pipeline
    #     ```
    #     >>> from fynns_tool_model_v1 import *
    #     >>> df = pd.DataFrame({ 'color': ['Red', 'Green', 'Green','Green', 'Red', 'Green'],'taste': ['Sweet', np.nan,'Sweet', 'Sour', 'Sweet','Sour'], 'size': [
    #     >>>                 'Big', 'Big', 'Small', 'Medium',np.nan, 'Small'], 'int_size': [7, 8, 2, 5,4, 2]})
    #     >>> Xcols = list(set(df.columns)-set(['int_size']))
    #     >>> m = Model(df[Xcols],df['int_size'],modelArgs={'n_estimators':100})
    #     >>> m

    #     Model set to `RandomForestRegressor` since no input for argument `model`.
    #     Model set to `OrdinalEncoder` since no input for argument `model`.
    #     X initiallized/updated.
    #     y initiallized/updated.
    #     modelArgs initiallized/updated.
    #         size  color  taste  int_size
    #     0     Big    Red  Sweet         7
    #     1     Big  Green    NaN         8
    #     2   Small  Green  Sweet         2
    #     3  Medium  Green   Sour         5
    #     4     NaN    Red  Sweet         4
    #     5   Small  Green   Sour         2

    #     >>> Xtest = m.getTrainTest()[1]
    #     >>> ap = m.autoPipeline()
    #     >>> pred = ap.predict(Xtest)
    #     >>> pred

    #     Transform Categorical Features Imputer Strategy: mean


    #     Transform Numerical Features Imputer Strategy: most_frequent
    #     Transform Encoder: OrdinalEncoder
    #     Encoder Arguments: {}   

    #     array([4.77, 2.96])
    #     ```
    #     '''

    #     X,y=self._X,self._y
    #     nct, cct = self.transformNumCols(strategy=transformNumStrategy), self.transformCatCols(
    #         imputeStrategy=transformCatColImputeStrategy)
        
    #     if preprocess_y:
    #         # actually we just preprocess y and overwrite it into the model attribute

    #         print("!!!WARNING: ALL train_test_split data WILL BE OVERWRITTEN BY NEW VALUE BECAUSE y is encoded!!!")
    #         # preprocessing y directly and separately here. not through single pipelines before
    #         transformedy=todf(cct.fit_transform(todf(y)))
    #         print("Transformed y:\n",y)
    #         transformedy.index,transformedy.columns=y.index,[y.name]
    #         # transformedy # dataframe
    #         self.init(X=X,y=transformedy[0]) # here we gets y back to series!!!!!!!!!!!!
    #         print("!!!WARNING: THE init arguments were default used, didn't bring per argument by in this function")
    #         print("Could bring in initArgs dict argument, to bring args into These init")

    #     # after initialing we call the train test data
    #     Xtrain,ytrain,Xtest,ytest=self.getTrainTest()

    #     preprocessorX = self.preprocessX(nct, cct)

    #     # Here we have to bring in transformed y directly and but leave Xtrain untransformed (we did it before just using preprocessor)
    #     model = self.pipeline(preprocessorX).fit(Xtrain, ytrain)
    #     return model

    # -------------------------------------------
    # Cross Validation

    def crossValScore(self, pipeline: Pipeline, X: pd.DataFrame, y: pd.Series,
                      cv=5,
                      scoring='neg_mean_absolute_error') -> ...:
        return

    def score_dataset(self) -> float:
        ...

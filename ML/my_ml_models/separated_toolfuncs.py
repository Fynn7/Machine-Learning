from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor

# -----------------------------------------------------------------------------------------------------------------------------------
# Funcs without interacting(not changing values, only passing values of attributes to the corresp. methods ) with class attributes


def transformNumCols(strategy: str='mean') -> SimpleImputer:
    '''
    We just need to impute the missing values from num cols.

    ```
    >>> from fynns_tool_model import *
    >>> from separated_toolfuncs import *
    >>> df = pd.DataFrame({'taste': ['Sweet', np.nan,'Sweet', 'Sour', 'Sweet','Sour'], 'size': [
    >>>                   'Big', 'Big', 'Small', 'Medium',np.nan, 'Small'], 'int_size': [7, 8, 2, np.nan,4, 2], 'color': ['Red', 'Green', 'Green','Green', 'Red', 'Green']})
    >>> Xcols = list(set(df.columns)-set(['color']))
    >>> m = Model(df[Xcols],df['color'])
    >>> X,y=m.getXy()
    >>> X,y

    Model has automatically set to RandomForestRegressor since you didn't input model name.
    The Data Is Not Clean. Set mae score to 0.
    X initiallized/updated.
    y initiallized/updated.
    modelArgs initiallized/updated.
    (   int_size    size  taste
    0       7.0     Big  Sweet
    1       8.0     Big    NaN
    2       2.0   Small  Sweet
    3       NaN  Medium   Sour
    4       4.0     NaN  Sweet
    5       2.0   Small   Sour,
    0      Red
    1    Green
    2    Green
    3    Green
    4      Red
    5    Green
    Name: color, dtype: object)


    ```
    '''
    return SimpleImputer(strategy=strategy)


def transformCatCols(imputeStrategy: str = 'most_frequent', encoder: str = 'OneHotEncoder', **encoderArgs) -> Pipeline:
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


def preprocessor(numCols: list, catCols: list, numColTransformer:SimpleImputer,catColTransformer:Pipeline) -> ColumnTransformer:
    '''
    Return a ColumnTransformer for preprocessing (preprocess to handle-able data, aka. numbers).
    ```

    ```
    '''
    return ColumnTransformer(
        transformers=[
            ('numerical Transformer', numColTransformer, numCols),
            ('categorical Transformer', catColTransformer, catCols)
        ])


def pipeline(preprocessor: ColumnTransformer, model: str = "RandomForestRegressor", **modelArgs) -> Pipeline:
    '''
    Bundle preprocessing and modeling code in a pipeline
    Return model with full steps of preprocessing+model

    ```
    ```
    '''
    argsComm = "("
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

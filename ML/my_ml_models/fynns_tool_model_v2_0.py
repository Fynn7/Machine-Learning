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


def fitModel(tts: tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series], modelName: str = 'RandomForestRegressor', cv: int = 5, **modelArgs) -> tuple[RandomForestRegressor, dict[float | np.ndarray]]:
    '''
    Args:
        tts: train_test_split(X,y)
        modelName: model name, default as 'RandomForestRegressor'. Support values: 'ARDRegression','AdaBoostClassifier','AdaBoostRegressor','BaggingClassifier','BaggingRegressor','BayesianRidge','BernoulliNB','CatBoostClassifier','CatBoostRegressor','ComplementNB','DecisionTreeClassifier','DecisionTreeRegressor','ElasticNet','ElasticNetCV','EllipticEnvelope','ExtraTreeRegressor','ExtraTreesClassifier','ExtraTreesRegressor','GammaRegressor','GaussianNB','GaussianProcessClassifier','GaussianProcessRegressor','GeneralizedLinearRegressor','GradientBoostingClassifier','GradientBoostingRegressor','HistGradientBoostingClassifier','HistGradientBoostingRegressor','HuberRegressor','IsolationForest','KNeighborsClassifier','KNeighborsRegressor','KernelRidge','LGBMClassifier','LGBMRegressor','LabelPropagation','LabelSpreading','Lasso','LassoCV','LinearDiscriminantAnalysis','LinearRegression','LinearSVR','LocalOutlierFactor','LogisticRegression','LogisticRegressionCV','MLPClassifier','MLPRegressor','MultinomialNB','NearestCentroid','NuSVR','OneClassSVM','OrthogonalMatchingPursuit','OrthogonalMatchingPursuitCV','PassiveAggressiveRegressor','Perceptron','PoissonRegressor','QuadraticDiscriminantAnalysis','RANSACRegressor','RadiusNeighborsClassifier','RadiusNeighborsRegressor','RandomForestRegressor','Ridge','RidgeCV','RidgeClassifier','RidgeClassifierCV','SGDClassifier','SGDRegressor','SVC','SVR','StackingClassifier','StackingRegressor','TheilSenRegressor','TweedieRegressor','VotingClassifier','VotingRegressor','XGBRegressor'
        cv: cross validation number, default as 5
        **modelArgs: model arguments, default as empty dict
    ```
        >>> from fynns_tool_model_v2_0 import *
        >>> df = pd.DataFrame({'color': ['Red', 'Green', 'Green','Green', 'Red', 'Green'],'int_gone_bad':[1,0,np.nan,0,0,0],'taste': ['Sweet', 'Sweet','Sweet', 'Sour', 'Sweet','Sour'], 'size': [
        >>>                   'Big', 'Big', 'Small', 'Medium',np.nan, 'Small'], 'int_size': [7, 8, 2, 5,4, np.nan]})
        >>> X=df[['color','int_size','size','int_gone_bad']]
        >>> y=df['taste']
        >>> X


        color	int_size	size	int_gone_bad
        0	Red	7.0	Big	1.0
        1	Green	8.0	Big	0.0
        2	Green	2.0	Small	NaN
        3	Green	5.0	Medium	0.0
        4	Red	4.0	NaN	0.0
        5	Green	NaN	Small	0.0

        >>> X,y=cleanData(X,y)
        >>> X

        numColsimputeStrategy: mean
        catColsimputeStrategy: most_frequent
        encoderName: oe 
        handle_unknown: error    
          

        numColsX: ['int_size', 'int_gone_bad']
        catColsX: ['color', 'size']
          
        int_size	int_gone_bad	color	size
        0	7.0	1.0	1.0	0.0
        1	8.0	0.0	0.0	0.0
        2	2.0	0.2	0.0	2.0
        3	5.0	0.0	0.0	1.0
        4	4.0	0.0	1.0	0.0
        5	5.2	0.0	0.0	2.0


        >>> fitModel(train_test_split(X,y),cv=2)

        Found ytrain,ytest as np.ndarray type, reshaping -1 into pd.Series.

                Successfully create model: RandomForestRegressor
                
        y_pred:[0.84 0.64];
        y_true:
        0    1.0
        1    0.0
        dtype: float64
        (RandomForestRegressor(),
        {'score': 0.12959999999999994,
        'mae_score': 0.4,
        'cv_score': array([0.5  , 0.655])})
    ```
    '''
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
    model=model.fit(Xtrain, ytrain) # variation 1: model.fit(Xtrain, ytrain)  (without `model=`)
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
    return model, {'score': score, 'mae_score': mae_score, 'cv_score': cv_score}


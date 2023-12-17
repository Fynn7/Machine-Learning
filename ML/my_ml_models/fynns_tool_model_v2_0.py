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
import matplotlib.pyplot as plt

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
            raise TypeError(f"Expect y as a pandas Series. Got {type(y)}.")
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


class Model:  # empty class just for hinting (pylance)
    def fit(self, Xtrain, ytrain):
        pass

    def fit_transform(self, X, y):
        pass

    def predict(self, Xtest):
        pass

    def score(self, Xtest, ytest):
        pass


def fitModel(tts: tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series], modelName: str = 'RandomForestRegressor', cv: int = 5, printInfo: bool = False, plotPred: bool = False, worst_n_preds: int = 1, **modelArgs) -> tuple[Model, dict[float | np.ndarray]]:
    '''
    Args:
        tts: train_test_split(X,y)
        modelName: model name, default as 'RandomForestRegressor'. Support values: 'ARDRegression','AdaBoostClassifier','AdaBoostRegressor','BaggingClassifier','BaggingRegressor','BayesianRidge','BernoulliNB','CatBoostClassifier','CatBoostRegressor','ComplementNB','DecisionTreeClassifier','DecisionTreeRegressor','ElasticNet','ElasticNetCV','EllipticEnvelope','ExtraTreeRegressor','ExtraTreesClassifier','ExtraTreesRegressor','GammaRegressor','GaussianNB','GaussianProcessClassifier','GaussianProcessRegressor','GeneralizedLinearRegressor','GradientBoostingClassifier','GradientBoostingRegressor','HistGradientBoostingClassifier','HistGradientBoostingRegressor','HuberRegressor','IsolationForest','KNeighborsClassifier','KNeighborsRegressor','KernelRidge','LGBMClassifier','LGBMRegressor','LabelPropagation','LabelSpreading','Lasso','LassoCV','LinearDiscriminantAnalysis','LinearRegression','LinearSVR','LocalOutlierFactor','LogisticRegression','LogisticRegressionCV','MLPClassifier','MLPRegressor','MultinomialNB','NearestCentroid','NuSVR','OneClassSVM','OrthogonalMatchingPursuit','OrthogonalMatchingPursuitCV','PassiveAggressiveRegressor','Perceptron','PoissonRegressor','QuadraticDiscriminantAnalysis','RANSACRegressor','RadiusNeighborsClassifier','RadiusNeighborsRegressor','RandomForestRegressor','Ridge','RidgeCV','RidgeClassifier','RidgeClassifierCV','SGDClassifier','SGDRegressor','SVC','SVR','StackingClassifier','StackingRegressor','TheilSenRegressor','TweedieRegressor','VotingClassifier','VotingRegressor','XGBRegressor'
        cv: cross validation number, default as 5
        printInfo: print model info, default as False
        plotPred: plot prediction, default as False
        worst_n_preds: plot worst n predictions as red dots, default as 1. 计算最差预测的n个点(diff)，并把它们标红
        **modelArgs: model arguments, default as empty dict

    Returns:
        model: e.g. XGBRegressor(random_state=0, ...),
        scores: {'score': ..., 'mae_score': ..., 'cv_score': ...}
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

    Testing on Kaggel Learning Notebook: XGBoostRegressor exercise

    `https://www.kaggle.com/code/fynnndreas/exercise-xgboost/edit`

    ```
        >>> # import these functions from this model...
        >>> ...
        >>> X,y=cleanData(X,y)
        >>> tts=train_test_split(X,y)
        >>> fitModel(tts,modelName='XGBRegressor',random_state=0)


        Successfully create model: XGBRegressor

        y_pred:[163474.78  428607.44  308496.34  185267.88  100593.74   92855.5
        313477.62  124950.55  268077.66  130366.44  213444.05  286540.12
        126592.36  190979.36  115987.23  242833.58  202199.48  157096.12
        127254.43  276219.56  307527.8   117826.125 119704.766 336385.2
        123473.805 209168.8    93860.695 158531.6   275648.44  204912.05
        145630.7   124979.04  128936.26  130631.95  162074.2   193102.06
        177652.27  209937.1   132445.11  146672.42  149760.44  182842.64
        105352.336 441448.75  238618.94  184119.47  204500.11  110405.85
        131304.92  182359.86  216617.16  350420.06  164733.39  108138.05
        230927.61  329077.66  110421.195 126848.42  104774.766 142946.67
        134791.58  110396.55  287512.03  109769.06  346300.62  186412.14
        88398.62  265194.9   249650.28  263419.    125731.86  216773.03
        137959.5   167434.84  148983.56  369164.2   126297.195 143039.22
        71791.27  128441.42  166390.97  180610.12  164406.83  401244.5
        433329.88  143011.06  202599.53  290394.72  224780.75   96614.555
        184685.    171108.16  209171.89  138440.16  202645.3   158438.3
        186035.08  231860.44  107733.46  118103.21  135909.8   291254.84
        103506.65  160484.64  119355.98  167899.08  269860.2   260043.45
        339188.62  191111.84  132213.7   236110.47  209426.58  206007.34
        171439.86  224245.56  146202.55  142444.03  270782.34  163230.17
        136737.75  147315.44  165190.67  121604.29  114792.67  212299.22
        193936.53  193520.88  194700.9    76283.83   79985.86  320113.47
        206822.11  251928.22  262413.38  236901.16  116750.23  146275.6
        255204.2   155015.4   135390.84  189522.1   177408.78  142905.38
        299931.6   142057.08  182527.6   225116.66  143205.4   170683.77
        129269.7   186773.08  151741.23  246438.05  144460.25   70416.43
        180844.14  167469.48  198656.3   190412.02  171678.3   240555.7
        239730.97  165603.55  213138.69  156881.83   99826.39  288310.9
        154515.81   85267.51  192061.33  439743.06  361629.44  194004.97
        464590.38  397278.9   167989.9   262591.06  231077.97  202341.45
        146347.17  127428.05  217716.23  104729.08  172873.52   92116.83
        182549.9   159256.72  173320.34  150332.06  193657.05  319214.
        186386.42  176115.02  127380.52  195555.66  167125.33  171773.27
        104324.984 129484.16   66945.92  297295.16  187971.22  125665.1
        197493.61  119461.74  305431.56  133675.44   61906.54  495388.53
        137962.73  170565.25  160581.97  177788.88  159710.34  160840.23
        247163.11  150311.47   98360.836  67838.24  180688.31  200122.53
        148067.62  156935.92  121813.05  349530.16  184179.9   177662.28
        225522.98  214277.3   174033.83  113913.79  125829.53  187179.81
        296002.56   89944.266 138435.25  244120.94  130928.375 115145.03
        109697.74  248923.89  164644.05  200682.55  225680.9   149921.94
        100235.05  229735.23  135106.58   69442.72  306926.84   94995.94
        177785.08   73159.24  179657.47  235733.66  212566.61  271636.12
        133499.89  109850.766 162456.45  109994.9    89465.484 227743.75
        151545.52  106483.85  221213.45  168634.36  157814.58  208347.4
        122817.23  113215.03  302798.28  128154.29  114924.49  105126.33
        87350.4    82626.12  164596.48  130073.984 252140.77  133993.56
        193785.28  170719.75  126914.7   146788.86  267143.03  125557.805
        285013.53  158307.83  147688.19   96538.46  129536.03  110469.77
        161521.19  370038.44  170850.75  166472.86  208704.02  267060.72
        231894.61  219377.05   78762.44  188201.    131798.23  177731.2
        105161.06  265256.4   367514.25  183302.64  180170.03  207232.28
        222924.72  315908.2   142862.94  167732.86  143661.25  202554.28
        219199.97  173324.81  168019.22  211273.06   93589.66  200565.73
        112165.     81398.68  270405.12  169682.86  184617.9   234150.88
        163529.64  144043.58  132148.62  144591.3   131838.39  186202.33
        73622.17  188818.77  107513.14  112409.85  166598.89  120601.98
        189326.67  158710.4   192057.92  426210.75  130945.91  109903.89
        117532.06  136783.89  152306.89  147708.36  253116.89  113951.54
        149009.06  132682.8   146761.12  126666.21  272271.9    86223.14
        188356.77  184515.72  172498.78  140623.69  104480.195];
        y_true:
            SalePrice
        799    175000.0
        1044   278000.0
        640    274000.0
        632     82500.0
        459    110000.0
        ...         ...
        1111   205000.0
        248    180000.0
        922    169990.0
        416    149500.0
        413    115000.0

        [365 rows x 1 columns]
        (XGBRegressor(base_score=None, booster=None, callbacks=None,
                    colsample_bylevel=None, colsample_bynode=None,
                    colsample_bytree=None, device=None, early_stopping_rounds=None,
                    enable_categorical=False, eval_metric=None, feature_types=None,
                    gamma=None, grow_policy=None, importance_type=None,
                    interaction_constraints=None, learning_rate=None, max_bin=None,
                    max_cat_threshold=None, max_cat_to_onehot=None,
                    max_delta_step=None, max_depth=None, max_leaves=None,
                    min_child_weight=None, missing=nan, monotone_constraints=None,
                    multi_strategy=None, n_estimators=None, n_jobs=None,
                    num_parallel_tree=None, random_state=0, ...),
        {'score': 0.8607098591264862,
        'mae_score': 17200.305072773972,
        'cv_score': array([18132.2284532 , 20960.76239655, 17231.01519692, 17936.10213328,
                19043.40710616])})
    '''
    # if model is not given or it is given illegally
    Xtrain, Xtest, ytrain, ytest = tts
    if type(ytrain) == np.ndarray and type(ytest) == np.ndarray:
        print("Found ytrain,ytest both as np.ndarray type, reshaping -1 into pd.Series.")
        ytrain, ytest = pd.Series(
            ytrain.reshape(-1)), pd.Series(ytest.reshape(-1))
    try:
        model = eval(f"{modelName}(**modelArgs)")
    except NameError as e:  # `modelName` type incorrect or unknown model
        model = RandomForestRegressor(**modelArgs)
        print(
            f"Illegal model name or type. Model argument set to default as `RandomForestRegressor`.\nOriginal error message: {e}")
    except Exception as e:
        print(
            f"Unknow error occurs while building the model.\nOriginal error message: {e}")
    # variation 1:model = model.fit(Xtrain, ytrain)  (with `model=`)
    model.fit(Xtrain, ytrain)
    ytest=ytest[ytest.columns[0]] # convert to pd.Series for plotPred! (for abs(ytest-pred) to work)
    pred = pd.Series(model.predict(Xtest), index=ytest.index,name=ytest.name)
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
    if printInfo:
        print(f'''
------------------------------------------
------------------------------------------
    DATA INFO:
------------------------------------------
model:\n{model}
------------------------------------------
model arguments:\n{modelArgs}
------------------------------------------
    ''')
        attrs = list(fitModel.__code__.co_varnames)[:-2]
        print("function arguments: ", attrs)
        attrs.remove('e')
        for attr in attrs:
            print(attr, '=>\n', eval(attr))
        print(f'''
------------------------------------------
------------------------------------------
    RESULTS:        
------------------------------------------
y_pred:{type(pred)}\n{pred}
------------------------------------------
y_test:{type(ytest)}\n{ytest}
------------------------------------------
score: {score}
mae_score: {mae_score}
cv_score: {cv_score}
------------------------------------------
------------------------------------------
              ''')
    if plotPred:  # only fits for 1-dimensional y
        i=ytest.index
        if worst_n_preds>len(i): # if worst_n_preds out of range
            raise ValueError(f"Argument `worst_n_preds` out of range: {worst_n_preds} > {len(i)}.")
        diff = abs(ytest-pred)
        print("diff=\n",diff,type(diff))
        print(f"Worst {worst_n_preds} predictions:\n",diff.nlargest(worst_n_preds))
        plt.scatter(i, ytest, c='green', label='y_true')
        plt.scatter(i, pred, c='orange', label='y_pred')
        plt.scatter(diff.nlargest(worst_n_preds).index, pred[diff.nlargest(worst_n_preds).index], c='red', label='worst_n_preds')
        plt.show()
    return model, {'score': score, 'mae_score': mae_score, 'cv_score': cv_score}

import sys
import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
import matplotlib.pylab as plt
import seaborn as sns
from functools import wraps
from scipy import stats
from sklearn.preprocessing import MinMaxScaler, StandardScaler


def trace(func: object) -> object:
    '''
    Decorater that traces a function.

    Args:
    - func: Function that needs to be traced

    How To Use:

    @trace # with decorator sign @
    def func(ARGS,KWARGS): # first define the function
        ...
    func(ARGS,KWARGS) # then call the function

    '''
    debug_log = sys.stderr  # Note that: originally, it is a global variable
    if debug_log:
        def callf(*args, **kwargs):
            """
            Private

            A wrapper function.

            Args:
            *args: unused parameters without arguments' names
            *kwargs: unused parameters with arguments' names
            """
            separator()
            debug_log.write('Calling function: {}\n'.format(func.__name__))
            res = func(*args, **kwargs)
            debug_log.write('Return value: {}\n'.format(res))
            separator()
            return res
        return callf
    else:
        return func


def separator(symb: str | None = '-', l: int = 42, noprint: bool | None = False) -> None | str:
    '''
    Formatter that separates sections.

    Args:
    - l: length of separator
    - noprint: instead of directly printing, returns the separator string

    '''
    if not isinstance(symb, str) or not isinstance(l, int):
        raise TypeError(
            f"Argument Type Error in one of them: {type(symb)}==str? / {type(l)}==int?")
    sep = symb*l
    if noprint:
        return sep
    print(sep)


def spacer(n: int | None = 2, noprint: bool | None = False) -> None | str:
    '''
    Formatter that separates sections with multiple \\n.

    Args:
    - n: length of the empty space
    - noprint: instead of directly printing, it returns multiple lines of blanks
    '''
    if not isinstance(n, int):
        raise TypeError(f"Argument Type Error: {type(n)}==int?")
    sp = "\n"*(n-1)
    if noprint:
        return sp
    print(sp)


def dfChecker(df: pd.DataFrame | pd.Series) -> pd.DataFrame:
    '''
    for Developer

    Check if the given argument is dataframe. If not, it returns a dataframe of the input data instead of original inputted datatype

    Args:
    - df: dataframe-like dataset
    '''
    try:
        df = pd.DataFrame(df)
    except Exception as e:
        separator()
        raise TypeError(
            f'''\nPlease input dataframe as dataframe or at least dataframe-like, instead of {type(df)}.\nIf you are using dfChecker() as a public function, please input dataframe instead.\n
            Original Error Message:
            {e}
            ''')
    return df

@trace
class DataAnalysis:
    def __init__(self, df: pd.DataFrame | pd.Series) -> None:
        # print("Start initializing instance...")
        self._data = dfChecker(df)

    def getData(self) -> pd.DataFrame:
        '''
        Get the current dataframe.
        '''
        return self._data

    def dataIntro(self) -> None:
        '''
        Print all data information and description of current dataframe.
        '''
        df = self.getData()
        separator()
        print("Basic Information of dataframe")
        separator()
        df.info()
        spacer()
        print(df.describe())
        separator()
        print("end dataIntro")
        separator()

    def checkTypes(self) -> None:
        '''
        Print out all kinds of types of the current dataframe.

        How to use:

        `da=DataAnalysis()`\n
        `da.checkTypes()`

        '''
        df = self.getData()

        typelist = df.dtypes.unique()
        typestr = str(list(typelist)).strip('[]')
        print(f"{len(typelist)} types in this dataframe:")
        print("You can see more detailed information with df.info(), which is already displayed before.")
        print(typestr)

    def getNumCols(self,subset:str|list=None) -> pd.DataFrame:
        '''
        Auxillary method

        Return dataframe only contains type 'float64' and 'int64'.

        Args:
        - subset: if you want to return a part of columns and only contain numbers, use `subset` to input the wanted dataframe
        `da.getNumCols(subset=['col1','col2',...]])`
        
        '''
        df=self.getData()
        if subset==None:
            subset=df.columns
        return df.select_dtypes(include=['float64', 'int64'])

    def createCorrMat(self, corrType: str = 'pearson') -> pd.DataFrame:
        '''
        Auxillary method

        Return: corrmat as `pd.Dataframe`

        Args:
        - corrType: 'pearson','kendall', 'spearman'
        '''
        df = self.getData()

        # only analyse numbers
        nums = self.getNumCols()
        corrmat = nums.corr(method=corrType)
        return corrmat

    def findCorr(self, x: str, corrMin: float | None = 0.5, corrMax: float | None = 1, corrType: str = 'pearson', head: int | None = None) -> tuple:
        '''
        Find the correlation of the whole dataframe.
        If you want to analyse as well the string and other types of data,
        use OneHotEncoder or any other kind of encoder to convert them into numbers

        Return: 
        - pd.Series: top ? correlation comparing to feature "col".

        Args:
        - x: feature name (column name) as string
        - corrMin: lower threshold of the correlation choosing range
        - corrMax: upper bound of the correlation choosing range
        - corrType: 'pearson','kendall', 'spearman'
        '''
        corrmat = self.createCorrMat(corrType=corrType)
        topCorrsmat = corrmat[corrmat[corrmat < corrMax] > corrMin]
        xcorr = topCorrsmat[x].sort_values(ascending=False).dropna()
        if head != None:
            return xcorr[:head]
        return xcorr

    def showAllRel(self, col: str) -> None:
        '''
        show all correlation of all other columns comparing to the argument "col"

        Return: None

        Args:
        - col: column that should be compared with
        '''
        df = self.getData()

        nums = self.getNumCols()
        for i in range(0, len(nums.columns), 5):
            sns.pairplot(data=nums,
                         x_vars=nums.columns[i:i+5],
                         y_vars=[col])

    def showDistr(self, col: str, kde: bool = True) -> float:
        '''
        Auxillary method

        Show the data distribution on any column of the dataframe
        And return the skewness

        Args:
        - col: show distribution comparing to this/these column(s)
        - kde: show kde line or not, default true
        '''
        df = self.getData()

        untransformed = sns.distplot(df[col], kde=kde)
        skewness = df[col].skew()
        return skewness

    def fixDistr(self, col: str, trans: str = 'log', inplace: bool = False) -> tuple:
        '''
        Do transforms on malformed data distribution

        Return: transformed pd.Series, new skewness as float

        Args:
        - col: show distribution comparing to this/these column(s)
        - trans: 'log' for log transformation, 
                'sqrt' for square root transformation,
                'boxcox' for box cox transformation
        - inplace: whether replace the original dataframe to the transformed data or not
        '''
        df = self.getData()

        skewness = self.showDistr(col)
        transformed = pd.Series()
        if trans == 'log':
            transformed = np.log(df[col])
        elif trans == 'sqrt':
            transformed = np.sqrt(df[col])
        elif trans == 'boxcox':
            transformed, maxlog = stats.boxcox()
        new_skewness = transformed.skew()
        sns.displot(transformed)

        if abs(new_skewness) < abs(skewness):
            print(
                f"Skew problem is better. Before: {skewness}; After:{skewness}")
        if inplace:
            df[col] = transformed
        return transformed, new_skewness

    def handleDup(self, subset: str | list | None = None, inplace: bool = False) -> pd.DataFrame:
        '''
        Return duplicated rows in cer tain columns.

        Args:

        - subset: only duplicate in certain columns, it will be returned 
        - inplace: whether replace the original dataframe to the transformed data or not
        '''
        df = self.getData()

        if subset == None:
            subset = df.columns

        if inplace:
            self.data = df.drop_duplicates(subset=subset)
        return df[df.duplicated(subset=subset)]

    def showNaN(self, head: int = 20, how: str = 'precentage') -> None:
        '''
        displaying NaN datas in a histplot

        Args:
        - head: top "head" missing values (NaN) amount
        - how: 'precentage' | 'count'
        '''
        df = self.getData()
        total = df.isnull().sum().sort_values(ascending=False)
        total_select = total.head(head)
        if how == 'precentage':
            precentage = 100*total_select/len(df)
            precentage.plot(kind="bar", figsize=(8, 6), fontsize=10)
            plt.ylabel("Precentage %", fontsize=20)
            plt.title("Total Missing Values Precentage %", fontsize=20)
        elif how == 'count':
            total_select.plot(kind="bar", figsize=(8, 6), fontsize=10)
            plt.ylabel("Count", fontsize=20)
            plt.title("Total Missing Values", fontsize=20)
        else:
            raise AttributeError(f"Invalid 'how' argument as '{how}'.")
        plt.xlabel("Columns", fontsize=20)

    def handleNaN(self, subset: str | list, how: str = 'row', inplace: bool = False) -> pd.DataFrame:
        '''
        Handling NaN data.

        Args:
        
        - subset: handle missing values onto these columns
        - how:  'row': Only drop missing values on certain columns
                'col' | 'column': drop whole column
                'median': replace missing values to medians
                'zero': replace missing values to zeros
                'mean': replace missing values to means
        - inplace: whether replace the original dataframe to the transformed data or not
        
        '''
        df = self.getData()
        dropped = pd.DataFrame()
        # if subset == None:
        #     raise TypeError("Please input 'subset' argument when dropping.")
        if how == 'row':
            print("Option 1: Only drop missing values on certain columns")
            dropped = df.dropna(subset=subset)
        elif how == 'col' or how == 'column':
            print("Option 2: Drop the whole column ")
            dropped = df.drop(subset, axis=1)
        elif how == 'median':
            print("Option 3: Replace missing values to medians")
            # dont need to check "subset"'s type
            median = df[subset].median()
            dropped = df[subset].fillna(median)
        elif how == 'zero':
            print("Option 4: Replace missing values to zeros")
            dropped = df[subset].fillna(0)
        elif how == 'mean':
            print("Option 5: Replace missing values to means")
            mean = df[subset].mean()
            dropped = df[subset].fillna(mean)
        else:
            raise AttributeError(f"Invalid 'how' argument as '{how}'.")
        if inplace:
            self._data = dropped
        return dropped

    def transformData(self, subset: str | list=None,scaler:str='min-max')->np.ndarray:
        '''
        Feature Scaling step.
        Uses "getNumCols()" in order to analyse numbers.

        Return: transformed data

        Args:
        - scaler: 'min-max' as `MinMaxScaler()`; 'standard' as `StandardScaler()`
        '''
        df=self.getData()
        nums=self.getNumCols(subset=subset) # here will just input subset argument and check, if subset==None
        if scaler=='min-max':
            return MinMaxScaler().fit_transform(nums)
        elif scaler=='standard':
            return StandardScaler().fit_transform(nums)
        else:
            raise AttributeError(f"Invalid 'scaler' argument as '{scaler}'.")
    # main function
    def plotOutliers(self,col:str|list|tuple,zscore=True)->pd.DataFrame|None:
        '''
        Auxillary method of `delOutliers`

        Note that boxplot is for single feature outlier seeking
        the scatter plot is for bi-variante feature outlier seeking

        z-score mathematically identifies outliers.

        if z-score turned on, it returns zstats dataframe.
        '''
        df=self.getData()
        # x,y,*=col
        if type(col)==str:
            sns.boxplot(x=df[col])
        else:  
            try:
                x,y=col
            except Exception as e:
                raise TypeError(f"If you choose a bi-variante feature outlier to seek, aka scatter plot, you must pass the `plot` argument as a list or tuple contains EXACT 2 element.\nOriginal Error message:{e}")
            df.plot.scatter(x=x,y=y)
        if zscore:
            print("Here is the Z-score stats of column",col)
            zstats=stats.zscore(df[col])
            print(zstats.describe().round(2))
            zmin,zmax=zstats.min(),zstats.max()
            print("If z-score (above shown) fulfills: zscore<-3||zscore>3, then it will be identified as a outlier\n")
            try:
                if (zmin>3) or (zmin<-3):
                    print(f"Outliers found: zmin={zmin} (greater than 3 or less than -3). Please check the plot manuelly.")
                    print()
                elif (zmax>3) or (zmax<-3):
                    print(f"Outliers found: zmax={zmax} (greater than 3 or less than -3). Please check the plot manuelly.")
                else:
                    print("No Z-score Outliers found.")
            except ValueError as e:
                raise ValueError(f"For z-score outlier analysis, you should only input string as `col` argument, instead of {type(col)}\nOriginal Error message:{e}")
            return zstats
        
    def delOutliers(self,col:str,how='sort',ascending:bool=False,inplace:bool=False)->pd.DataFrame|str:
        '''
        Delete Outliers, but it should be highly customed.

        Return: only sort_values result for `pd.DataFrame`

        How to use:
        `da.delOutliers(col=COL_NAME).drop(df.index[[OUTLIER_INDICES]]])`

        Details:
        1. First using df.sort_values() to sort out outlier values you've seen in method `plotOutliers`
        because there is no standard to decide whether these are outliers or not
        

        2. Then using df.drop() to drop the outliers directly with their indices
        eg: `outliers_dropped = df.drop(df.index[[1499,2181]])`
        
        Note that Option 1 only supports str parameter, instead of passing a list
        '''

        df=self.getData()
        # zstats=self.plotOutliers(col=col)
        if how=='sort':
            _sorted=df.sort_values(by=col,ascending=ascending)
            print("Now the dataframe has sorted. Find out the outliers by index and drop them yourself!")
        else:
            raise AttributeError(f"Invalid 'how' argument as '{how}'.")
        if inplace:
            self._data=_sorted
        return _sorted
    def analyseData(self, col: str | None = None) -> None:
        '''
        Main Tool Function of Machine Learning. \n
        Now only for testing!!!

        Args:
        - df: `pd.Dataframe()`
        '''
        if col == None:
            self.dataIntro()
            self.checkTypes()
            # self.showAllRel(col=col)
            self.createCorrMat()
            # self.findCorr(col=col)
            # self.showDistr()
            # self.fixDistr()
            self.handleDup()
            self.showNaN()
            # self.handleNaN()
            print("scaled data:", self.transformData())
        else:
            self.dataIntro()
            self.checkTypes()
            self.showAllRel(col=col)
            self.createCorrMat()
            self.findCorr(col=col)
            self.fixDistr(col=col)
            self.handleDup()
            self.showNaN()
            self.handleNaN(subset=col)

# --------------------------------------------------------------------------------------------------------------'''
# Class funcs
    # def dataIntro(cls, df: pd.DataFrame | pd.Series) -> None:
    #     '''
    #     Print all data information and description.
    #     Class method
    #     '''
    #     df = dfChecker(df)
    #     separator()
    #     print("Basic Information of dataframe")
    #     separator()
    #     df.info()
    #     spacer()
    #     print(df.describe())
    #     separator()
    #     print("end dataIntro")
    #     separator()

    # def checkTypes(cls, df: pd.DataFrame | pd.Series) -> None:
    #     '''
    #     Analyse the types of the given dataframe.
    #     '''
    #     df = dfChecker(df)

    #     typelist = df.dtypes.unique()
    #     typestr = str(list(typelist)).strip('[]')
    #     print(f"{len(typelist)} types in this dataframe:")
    #     print("You can see more detailed information with df.info(), which is already displayed before.")
    #     print(typestr)

    # def analyseData(cls, df: pd.DataFrame | pd.Series | None = ...) -> None:
    #     '''
    #     Main Tool Function of Machine Learning

    #     Args:
    #     - df: `pd.Dataframe()`. Original dataframe.
    #     '''
    #     df = dfChecker(df)

    #     cls.dataIntro(df)
    #     cls.checkTypes(df)
    #     cls.findCorr(df)
    #     cls.handleNaN(df)

    # def createCorrMat(cls, df: pd.DataFrame | pd.Series, corrType: str = 'pearson') -> pd.DataFrame:
    #     '''
    #     Auxillary method

    #     Return: corrmat (pd.Dataframe)
    #     Args:
    #     - corrType: 'pearson','kendall', 'spearman'
    #     '''
    #     df = dfChecker(df)

    #     # only analyse numbers
    #     nums = cls.getNumCols(df)
    #     corrmat = nums.corr(method=corrType)
    #     return corrmat

    # def getNumCols(cls, df: pd.DataFrame | pd.Series) -> pd.DataFrame:
    #     '''
    #     Auxillary method
    #     '''
    #     df = dfChecker(df)

    #     return df.select_dtypes(include=['float64', 'int64'])

    # def findCorr(cls, df: pd.DataFrame | pd.Series, x: str, corrMin: float | None = 0.5, corrMax: float | None = 1, corrType: str = 'pearson', head: int = 10) -> tuple:
    #     '''
    #     Find the correlation of the whole dataframe.
    #     If you want to analyse as well the string and other types of data,
    #     use OneHotEncoder or any other kind of encoder to convert them into numbers

    #     Return:
    #     - pd.Series: top ? correlation comparing to feature x.

    #     Args:
    #     - x: feature name (column name) as string
    #     - corrMin: lower threshold of the correlation choosing range
    #     - corrMax: upper bound of the correlation choosing range
    #     - corrType: 'pearson','kendall', 'spearman'
    #     '''
    #     df = dfChecker(df)

    #     corrmat = cls.createCorrMat(df, corrType=corrType)
    #     topPosCorrsmat = corrmat[corrmat[corrmat < corrMax] > corrMin]
    #     xcorr = topPosCorrsmat[x].sort_values(ascending=False).dropna()
    #     return xcorr

    # def handleDup(cls, df: pd.DataFrame | pd.Series, subset: str | list | None = None, inplace=False) -> pd.DataFrame:
    #     '''
    #     Return duplicated rows in certain columns.

    #     Args:

    #     - cols: only duplicate in certain columns, it will be returned
    #     '''
    #     df = dfChecker(df)

    #     if subset == None:
    #         subset = df.columns

    #     if inplace:
    #         df = df.drop_duplicates(subset=subset)
    #     return df[df.duplicated(subset=subset)]

    # def handleNaN(cls, df: pd.DataFrame | pd.Series, subset: str | list, how: str = 'row', inplace: bool = False) -> pd.DataFrame:
    #     '''
    #     Handling NaN data.

    #     Args:
    #     - how:  'row': Only drop missing values on certain columns
    #             'col' | 'column': drop whole column
    #             'median': replace missing values to medians
    #             'zero': replace missing values to zeros
    #             'mean': replace missing values to means
    #     '''
    #     df = dfChecker(df)

    #     dropped = pd.DataFrame()
    #     if subset == None:
    #         raise TypeError("Please input 'subset' argument when dropping.")
    #     if how == 'row':
    #         print("Option 1: Only drop missing values on certain columns")
    #         dropped = df.dropna(subset=subset)
    #     elif how == 'col' | 'column':
    #         print("Option 2: Drop the whole column ")
    #         dropped = df.drop(subset, axis=1)
    #     elif how == 'median':
    #         print("Option 3: Replace missing values to medians")
    #         # dont need to check "subset"'s type
    #         median = df[subset].median()
    #         dropped = df[subset].fillna(median)
    #     elif how == 'zero':
    #         print("Option 4: Replace missing values to zeros")
    #         dropped = df[subset].fillna(0)
    #     elif how == 'mean':
    #         print("Option 5: Replace missing values to means")
    #         mean = df[subset].mean()
    #         dropped = df[subset].fillna(mean)
    #     else:
    #         raise AttributeError("'how' argument not given right.")
    #     if inplace:
    #         df = dropped
    #     return dropped


if __name__ == "__main__":
    pass

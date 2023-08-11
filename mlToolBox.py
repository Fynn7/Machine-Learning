import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from functools import wraps
import sys
from scipy import stats


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
            debug_log.write('Calling function: {}\n'.format(func.__name__))
            res = func(*args, **kwargs)
            debug_log.write('Return value: {}\n'.format(res))
            return res
        return callf
    else:
        return func


def separator(symb: str | None = '-', l: int = 42, noprint: bool | None = False) -> None | str:
    '''
    Formatter that separates sections.

    Args:
    - l: length of separator
    - noprint: instead of directly printing

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
    - noprint: instead of directly printing
    '''
    if not isinstance(n, int):
        raise TypeError(f"Argument Type Error: {type(n)}==int?")
    sp = "\n"*(n-1)
    if noprint:
        return sp
    print(sp)


def dfChecker(df: pd.DataFrame | pd.Series = ...) -> None:
    '''
    for Developer

    Check if the given argument is dataframe.

    Args:
    - df: data source
    '''
    if type(df) != pd.DataFrame:
        separator()
        raise TypeError(
            f"Please input dataframe as dataframe, instead of {type(df)}.\nIf you are using dfChecker() as a public function, please input dataframe instead.\n")


@trace
class DataAnalysis:
    def __init__(self, df: pd.DataFrame | pd.Series) -> None:
        dfChecker(df)
        self._data = df

    def getData(self) -> pd.DataFrame:
        return self._data

    def dataIntro(self) -> None:
        '''
        Print all data information and description.
        Instance method
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

    def dataIntro(cls, df: pd.DataFrame | pd.Series) -> None:
        '''
        Print all data information and description.
        Class method
        '''
        dfChecker(df)
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
        Analyse the types of the given dataframe.
        '''
        df = self.getData()

        typelist = df.dtypes.unique()
        typestr = str(list(typelist)).strip('[]')
        print(f"{len(typelist)} types in this dataframe:")
        print("You can see more detailed information with df.info(), which is already displayed before.")
        print(typestr)

    def checkTypes(cls, df: pd.DataFrame | pd.Series) -> None:
        '''
        Analyse the types of the given dataframe.
        '''
        dfChecker(df)
        typelist = df.dtypes.unique()
        typestr = str(list(typelist)).strip('[]')
        print(f"{len(typelist)} types in this dataframe:")
        print("You can see more detailed information with df.info(), which is already displayed before.")
        print(typestr)

    def getNumCols(self) -> pd.DataFrame:
        df = self.getData()

        return df.select_dtypes(include=['float64', 'int64'])

    def getNumCols(cls, df: pd.DataFrame | pd.Series) -> pd.DataFrame:
        return df.select_dtypes(include=['float64', 'int64'])

    def createCorrMat(self, corrType: str = 'pearson') -> pd.DataFrame:
        '''
        Return: corrmat (pd.Dataframe)

        Args:
        - corrType: 'pearson','kendall', 'spearman'
        '''
        df = self.getData()

        # only analyse numbers
        nums = self.getNumCols()
        corrmat = nums.corr(method=corrType)
        return corrmat

    def createCorrMat(cls, df: pd.DataFrame | pd.Series, corrType: str = 'pearson') -> pd.DataFrame:
        '''
        Return: corrmat (pd.Dataframe)

        Args:
        - corrType: 'pearson','kendall', 'spearman'
        '''
        dfChecker(df)
        # only analyse numbers
        nums = cls.getNumCols(df)
        corrmat = nums.corr(method=corrType)
        return corrmat

    def findCorr(self, col: str, corrMin: float | None = 0.5, corrMax: float | None = 1, corrType: str = 'pearson', head: int = 10) -> tuple:
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
        corrmat = self.createCorrMat()
        topPosCorrsmat = corrmat[corrmat[corrmat < corrMax] > corrMin]
        xcorr = topPosCorrsmat[col].sort_values(ascending=False).dropna()
        return xcorr

    def findCorr(cls, df: pd.DataFrame | pd.Series, x: str, corrMin: float | None = 0.5, corrMax: float | None = 1, corrType: str = 'pearson', head: int = 10) -> tuple:
        '''
        Find the correlation of the whole dataframe.
        If you want to analyse as well the string and other types of data,
        use OneHotEncoder or any other kind of encoder to convert them into numbers

        Return: 
        - pd.Series: top ? correlation comparing to feature x.

        Args:
        - x: feature name (column name) as string
        - corrMin: lower threshold of the correlation choosing range
        - corrMax: upper bound of the correlation choosing range
        - corrType: 'pearson','kendall', 'spearman'
        '''
        corrmat = cls.createCorrMat(df)
        topPosCorrsmat = corrmat[corrmat[corrmat < corrMax] > corrMin]
        xcorr = topPosCorrsmat[x].sort_values(ascending=False).dropna()
        return xcorr

    def showAllCorr(self, col: str) -> None:
        '''
        show all correlation of all other columns comparing to the argument "col"

        Return: None

        Args:
        - col: column that should be compared with
        '''
        df = self.getData()

        nums = self.getNumCols(df)
        for i in range(0, len(nums.columns), 5):
            sns.pairplot(data=nums,
                         x_vars=nums.columns[i:i+5],
                         y_vars=[col])

    def showDistr(self, col: str, kde: bool = True) -> float:
        '''
        Show the data distribution on any column of the dataframe
        And return the skewness
        '''
        df = self.getData()

        untransformed = sns.distplot(df[col], kde=kde)
        skewness = df[col].skew()
        return skewness

    def fixDistr(self, col: str, trans: str = 'log', inplace: bool = False) -> tuple:
        '''
        Do transforms on malformed data distribution

        Return: transformed pd.Series , new skewness as float

        Args:
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

        - cols: only duplicate in certain columns, it will be returned 
        '''
        df = self.getData()

        if subset == None:
            subset = df.columns

        if inplace:
            self.data = df.drop_duplicates(subset=subset)
        return df[df.duplicated(subset=subset)]

    def handleDup(cls, df: pd.DataFrame | pd.Series, subset: str | list | None = None, inplace=False) -> pd.DataFrame:
        '''
        Return duplicated rows in certain columns.

        Args:

        - cols: only duplicate in certain columns, it will be returned 
        '''
        if subset == None:
            subset = df.columns

        if inplace:
            df = df.drop_duplicates(subset=subset)
        return df[df.duplicated(subset=subset)]

    def showNaN(self, head: int = 20) -> None:
        '''
        displaying NaN datas in a histplot

        Args:
        - head: top "head" missing values (NaN) amount
        '''
        df = self.getData()
        total = df.isnull().sum().sort_values(ascending=False)
        total_select = total.head(head)
        total_select.plot(kind="bar", figsize=(8, 6), fontsize=10)

        plt.xlabel("Columns", fontsize=20)
        plt.ylabel("Count", fontsize=20)
        plt.title("Total Missing Values", fontsize=20)

    def handleNaN(self, subset: str | list, how: str = 'row', inplace: bool = False) -> pd.DataFrame:
        '''
        Handling NaN data.

        Args:
        - how:  'row': Only drop missing values on certain columns
                'col' | 'column': drop whole column
                'median': replace missing values to medians
                'zero': replace missing values to zeros
                'mean': replace missing values to means
        '''
        df = self.getData()
        dropped = pd.DataFrame()
        if subset == None:
            raise TypeError("Please input 'subset' argument when dropping.")
        if how == 'row':
            print("Option 1: Only drop missing values on certain columns")
            dropped = df.dropna(subset=subset)
        elif how == 'col' | 'column':
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
            raise AttributeError("'how' argument not given right.")
        if inplace:
            self._data = dropped
        return dropped

    def handleNaN(cls,df:pd.DataFrame|pd.Series, subset: str | list, how: str = 'row', inplace: bool = False) -> pd.DataFrame:
        '''
        Handling NaN data.

        Args:
        - how:  'row': Only drop missing values on certain columns
                'col' | 'column': drop whole column
                'median': replace missing values to medians
                'zero': replace missing values to zeros
                'mean': replace missing values to means
        '''
        dropped = pd.DataFrame()
        if subset == None:
            raise TypeError("Please input 'subset' argument when dropping.")
        if how == 'row':
            print("Option 1: Only drop missing values on certain columns")
            dropped = df.dropna(subset=subset)
        elif how == 'col' | 'column':
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
            raise AttributeError("'how' argument not given right.")
        if inplace:
            df=dropped
        return dropped

    # main function

    def analyseData(self) -> None:
        '''
        Main Tool Function of Machine Learning

        Args:
        - df: <code>pd.Dataframe()</code>. Original dataframe.
        '''
        df = self.getData()

        self.dataIntro(df)
        self.checkTypes(df)
        self.findCorr(df)
        self.handleNaN(df)

    def analyseData(cls, df: pd.DataFrame | pd.Series | None = ...) -> None:
        '''
        Main Tool Function of Machine Learning

        Args:
        - df: <code>pd.Dataframe()</code>. Original dataframe.
        '''
        dfChecker(df)
        cls.dataIntro(df)
        cls.checkTypes(df)
        cls.findCorr(df)
        cls.handleNaN(df)


if __name__ == "__main__":
    pass

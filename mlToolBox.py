import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
from functools import wraps
import sys
from threading import Event,Thread


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


def dfChecker(df: pd.DataFrame | pd.Series | None = ...) -> None:
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





class DataAnalysis:
    def __init__(self, df: pd.DataFrame | pd.Series) -> None:
        self.dfChecker(df)
        self.data = df

    def dataIntro(self) -> None:
        '''
        Print all data information and description.
        Instance method
        '''
        df=self.data
        self.dfChecker()
        separator()
        print("Basic Information of dataframe")
        separator()
        df.info()
        spacer()
        print(df.describe())
        separator()
        print("end dataIntro")
        separator()
    
    def dataIntro(cls,df:pd.DataFrame|pd.Series)->None:

        '''
        Print all data information and description.
        Class method
        '''
        cls.dfChecker()
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
        df=self.data
        self.dfChecker()
        typelist = df.dtypes.unique()
        typestr = str(list(typelist)).strip('[]')
        print(f"{len(typelist)} types in this dataframe:")
        print("You can see more detailed information with df.info(), which is already displayed before.")
        print(typestr)

    def checkTypes(cls,df:pd.DataFrame|pd.Series) -> None:
        '''
        Analyse the types of the given dataframe.
        '''
        cls.dfChecker()
        typelist = df.dtypes.unique()
        typestr = str(list(typelist)).strip('[]')
        print(f"{len(typelist)} types in this dataframe:")
        print("You can see more detailed information with df.info(), which is already displayed before.")
        print(typestr)

    def findCorr(self) -> None:
        '''
        Find the correlation of the whole dataframe.
        '''
        df=self.data
        ...
    def findCorr(cls,df:pd.DataFrame|pd.Series) -> None:
        '''
        Find the correlation of the whole dataframe.
        '''
        ...

    def handleNaN(self) -> None:
        '''
        Handling NaN data.
        '''
        df=self.data
        self.dfChecker()
        print("Option 1: Drop NaN columns")
    def handleNaN(cls, df: pd.DataFrame | pd.Series) -> None:
        '''
        Handling NaN data.
        '''
        cls.dfChecker()
        print("Option 1: Drop NaN columns")
        ...
    # main function
    def analyseData(self) -> None:
        '''
        Main Tool Function of Machine Learning

        Args:
        - df: <code>pd.Dataframe()</code>. Original dataframe.
        '''
        df=self.data
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
        cls.dataIntro(df)
        cls.checkTypes(df)
        cls.findCorr(df)
        cls.handleNaN(df)

if __name__ == "__main__":
    pass
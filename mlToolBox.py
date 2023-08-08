import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
from functools import wraps
import sys
import threading

def trace(func: object) -> object:
    '''
    Public

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
    Public

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
    Public

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
    Public for Developer

    Check if the given argument is dataframe.

    Args:
    - df: data source
    '''
    if type(df) != pd.DataFrame:
        separator()
        raise TypeError(
            f"Please input dataframe as dataframe, instead of {type(df)}.\nIf you are using dfChecker() as a public function, please input dataframe instead.\n")


class Timer:
    '''
    All timer methodes in 1.\n

    - timer.timeme(func)
    - Timer.timeit(func)
    '''

    def timeme(self, func, include_sleep: False):
        '''
        Decorator that calculates the process time of a program process.
        Instance method

        Args:
        - funcArgs: a single object or function / a list of objects
        - include_sleep: including processing time from sleep()

        How To Use:
        - As an instance method:

        timer=Timer()
        @timer.timeme
        def func():
            ...

        PS: Note that process_time() does not include the time through sleep() methode and perf_counter() does.
        Also: inside there's a generator.
        '''
        def gen(timefunc: object):
            '''
            private Generator

            - timefunc: Whole function itself. Either perf_counter() or process_time()
            '''
            start = timefunc()
            if callable(func):
                func()
            else:
                raise TypeError(f"Not a callable type({type(func)}):{func}")
            end = timefunc()
            print(f"Processing Time of {func.__name__}", end-start)
            return func

        @wraps(func)
        def wrapper(*args, **kwargs):
            if include_sleep:
                return gen(time.perf_counter)(*args, **kwargs)
            else:
                return gen(time.process_time)(*args, **kwargs)

        return wrapper

    @classmethod
    def timeit(cls, func, include_sleep: False):
        '''
        Decorator that calculates the process time of a program process.
        Class Method

        Args:
        - funcArgs: a single object or function / a list of objects
        - include_sleep: including processing time from sleep()

        How To Use:
        - As a class method:

        @Timer.timeit
        def func():
            ...

        PS: Note that process_time() does not include the time through sleep() methode and perf_counter() does.
        Also: inside there's a generator.
        '''
        def gen(timefunc: object):
            '''
            private Generator

            - timefunc: Whole function itself. Either perf_counter() or process_time()
            '''
            start = timefunc()
            if callable(func):
                func()
            else:
                raise TypeError(f"Not a callable type({type(func)}):{func}")
            end = timefunc()
            print(f"Processing Time of {func.__name__}", end-start)
            return func

        @wraps(func)
        def wrapper(*args, **kwargs):
            if include_sleep:
                return gen(time.perf_counter)(*args, **kwargs)
            else:
                return gen(time.process_time)(*args, **kwargs)

        return wrapper


class Loader(threading.Thread):
    '''
    Loading decoration class.
    Only used inside the script. For developers, not UI or on display layer.
    '''

    def __init__(self) -> None:
        threading.Thread.__init__(self)
        self.status: str = ''
        self.running: bool = False # also, threading.Thread has self.isAlive attribute
        self.typedict: dict = {
            'rotate': ['|', '/', '-', '\\'],
            'dot': ['.', '..', '...', '....', '.....'],
            0: ['|', '/', '-', '\\'],
            1: ['.', '..', '...', '....', '.....'],
        }  # default rotating
        self.type: str | int = 0
    def info(self):
        print(
            f'''
        status:{self.status}
        running:{self.running}
        loading type:{self.type}
'''
        )

    def stop(self) -> None:
        self.running = False

    def isRunning(self) -> bool:
        return self.running
    
    @trace
    def run(self,interval:float=0.5) -> None: # rewrite run() from threading.Thread, it directly creates a thread
        #把要执行的代码写到run函数里面 线程在创建后会直接运行run函数 
        '''
        Public

        Loading decorator

        Args:
        - interval: loading refresh time interval
        '''
        self.running = True
        print(self.status)
        while self.running: # optimized (include self.typedict and make it iterate, dont need if type==0|1)
            for e in self.typedict[self.type]:
                sys.stdout.write(f"\rLoading {e}")
                time.sleep(interval)


class DataAnalysis:
    def __init__(self, df: pd.DataFrame | pd.Series = ...) -> None:
        self.dfChecker(df)
        self.data = df

    def dataIntro(self, df: pd.DataFrame | pd.Series | None = ...) -> None:
        '''
        Public

        Print all data information and description.

        Args:
        - df: given data as pandas dataframe
        '''
        if df is None:  # if called without arguments but called as a methode
            df = self.data
        self.dfChecker()
        self.separator()
        print("Basic Information of dataframe")
        self.separator()
        df.info()
        self.spacer()
        print(df.describe())
        self.separator()
        print("end dataIntro")
        self.separator()

    def checkTypes(self, df: pd.DataFrame | pd.Series | None = ...) -> None:
        '''
        Public

        Analyse the types of the given dataframe.

        Args:
        - df: given data in pandas dataframe
        '''
        if df is None:  # if called without arguments but called as a methode
            df = self.data
        self.dfChecker()
        typelist = df.dtypes.unique()
        typestr = str(list(typelist)).strip('[]')
        print(f"{len(typelist)} types in this dataframe:")
        print("You can see more detailed information with df.info(), which is already displayed before.")
        print(typestr)

    def findCorr(self, df: pd.DataFrame | pd.Series | None = ...) -> None:
        '''
        Public

        Find the correlation of the whole dataframe.
        '''
        if df is None:  # if called without arguments but called as a methode
            print("you called FUNC without argumenbt")
            df = self.data
        print(df)
        ...

    def handleNaN(self, df: pd.DataFrame | pd.Series | None = ...) -> None:
        '''
        Public

        Handling NaN data.
        '''
        if df is None:  # if called without arguments but called as a methode
            df = self.data
        self.dfChecker()
        print("Option 1: Drop NaN columns")

    # main function
    def analyseData(self, df: pd.DataFrame | pd.Series | None = ...) -> None:
        '''
        Main Tool Function of Machine Learning

        Args:
        - df: <code>pd.Dataframe()</code>. Original dataframe.
        '''
        if df is None:  # if called without arguments but called as a methode
            df = self.data
        self.dataIntro(df)
        self.checkTypes(df)
        self.findCorr(df)
        self.handleNaN(df)


if __name__ == "__main__":
    loader=Loader()
    loader.start()
    loader.stop()
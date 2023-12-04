from threading import Event,Thread
from functools import wraps
import time

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

class Loader():
    '''
    Loading decoration class.
    Only used inside the script. For developers, not UI or on display layer.
    '''

    def __init__(self) -> None:
        '''
        Args:

        - _threads: 
        '''
        self._threads = dict()  # should finally into np.array
        self._events=dict()
        self._typedict: dict = {
            'rotate': ['|', '/', '-', '\\'],
            'dot': ['.', '..', '...', '....', '.....'],
            0: ['|', '/', '-', '\\'],
            1: ['.', '..', '...', '....', '.....'],
        }  # default rotating
        self._type: str | int = 0

    def readType(self) -> str | int:
        return self._type

    def createThread(self,taskfunc: object ,threadName: str | None=None,eventName: str | None=None,event:Event|None=None) -> list:
        '''
        Creating thread and event at the same time
        thread1 <-> event1
        threadn <-> event2

        - taskfunc: like run(), timer starter functions. The task of the thread
        - threadName: thread's name
        - event: default creates a new Event() object. Can be customized as: multiple threads fits into a same event:
            myevent=Event()
            createThread(...,...,myevent)
            startThread(...)

            createThread('Thread 2',myevent)
            ...

        '''
        if threadName == None:
            threadName = f"DefaultThread {len(self._threads)+1}"
        if eventName == None:
            eventName = f"DefaultThread {len(self._threads)+1}"
        if threadName in self._threads.keys():
            raise KeyError(f"Already exists {threadName}")
        threadInfo=(Thread(target=taskfunc,args=(event,),threadName=threadName),
                                                       'stopped',taskfunc) # self._thread[threadName]=(Thread it self, Thread Status,taskfunc)
        self._threads[threadName] = threadInfo
        

        if event == None: # user is not using an old event, instead we create a new event
            event=Event()

        eventInfo=(event,threadName)
        self._events[eventName]=eventInfo
        return [
            threadName,
            threadInfo,
            eventName,
            eventInfo,
        ]
    def displayAllThreads(self) -> None:
        for threadName,threadInfo in self._threads.items():
            print(f'''
                Thread: {threadInfo[0]}
                Thread Name: {threadName}
                Status: {threadInfo[1]}
                Task Function:{threadInfo[2]}
                ''')
    def displayAllEvents(self)->None:
        for eventName,eventInfo in self._events.items():
            print(f'''
                Event Name: {eventName}
                Event: {eventInfo[0]}
                Thread Name: {eventInfo[1]}
                ''')
    def getThreadInfo(self, threadName: str) -> tuple:
        # may occur index error of python dict
        threadInfo=self._threads[threadName]
        return threadInfo
    def getEventInfo(self,eventName:str)->tuple:
        eventInfo=self._events[eventName]
        return eventInfo
    def getThread(self,threadName:str)->Thread:
        return self._threads[threadName][0]
    def getEvent(self,eventName:str)->Event:
        return self._events[eventName][0]
    def delThread(self, threadName: str) -> None:
        del self._threads[threadName]
        print("Successfully removed thread:", threadName)
    def delEvent(self,eventName:str)->None:
        del self._events[eventName]

    def startThread(self, threadName: str) -> None:
        threadInfo=self._threads[threadName]
        threadInfo[0].start()
        threadInfo[1]='running'

    def isRunning(self,threadName:str)->bool:
        threadInfo=self._threads[threadName]
        return threadInfo[1]=='running'
    
    def stopThread(self,threadName:str)->None:
        event.set() # NOT WORKING AS IN RUN()
    @trace
    def run(self, interval: float = 0.5) -> None:
        '''
        Loading decorator, normally as target of a thread.

        Args:
        - interval: loading refresh time interval

        How to use:

        createThread(loader.run())
        startThread(createThread[0])
        '''
        while self.:
            if event.is_set(): # ！！！！！！！！！！HOW TO CALL THIS EVENT? SEARCH THE RUN() FUNCTION INSIDE SELF._THREAD[THREADNAME][2]?????
            for e in self._typedict[self._type]:
                sys.stdout.write(f"\rLoading {e}")
                time.sleep(interval)

    def stopThread(self) -> None:

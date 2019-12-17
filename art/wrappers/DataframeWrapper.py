import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)

class DataframeWrapper(np.ndarray):
    """
    Wrapper class for pandas dataframes and series. Note that the attacks will always return a numpy array even if this wrapper is used.
    """
    logger.warning("This is experimental. Use with caution")
    __copy_dataframe = True
    __dataframe = None #The underlying dataframe used to create the array. This can change due to changes in the 
                       #numpy array such as slicing. So long as the # of columns 
                       # remains consistent and the array is 2D, the dataframe should remain the same.

    def __new__(cls, dataframe, copy=True):
        if(len(np.shape(dataframe)) > 2):
            logging.error("Input must have no more than 2 dimensions")
            return

        if isinstance(dataframe, pd.DataFrame) or isinstance(dataframe, pd.Series):

            #If the input is a series, we have to change it into a row
            if(isinstance(dataframe, pd.Series)):
                dataframe = pd.DataFrame(dataframe.to_numpy()[np.newaxis, :], columns=dataframe.index.values)
            if(copy):
                dataframe_copy = dataframe.copy()
                obj = dataframe_copy.to_numpy().view(cls)
                obj.__dataframe = dataframe_copy
            else:
                obj = dataframe.to_numpy().view(cls)
                obj.__dataframe = dataframe
                
            return obj
        else:
            logger.error("Input must be a dataframe or a series")
            return None
   
    #the order input is ignored. It's just there for compatibility
    def copy(self, order='C'):
        if(len(self.shape) == 1):
            self = self[np.newaxis, :]
        if(self.__copy_dataframe):  #If true, then we can copy the dataframe with the array
            return self.__new__(DataframeWrapper, pd.DataFrame(self, columns=self.__dataframe.columns.values), copy=True)
        else:
            return np.ndarray.copy(self, order)

    def __array_finalize__(self, obj):
        if obj is None: 
            return

        self.__copy_dataframe = getattr(obj, '__copy_dataframe', True)
        self.__dataframe = getattr(obj, 'dataframe', None)
        
        if(self.__dataframe is not None):
            self_shape = np.shape(self)
            df_shape = np.shape(self.__dataframe)
            
            #If the numpy array has the same number of columns, redefine the dataframe to mirror the numpy array.
            if(len(self_shape) == 2 and self_shape[1] == df_shape[1]):
                self.__dataframe = pd.DataFrame(self, columns=self.__dataframe.columns.values)
            elif (len(self_shape) == 1 and self_shape[0] == df_shape[1]):
                self.__dataframe = pd.DataFrame(self[np.newaxis], columns=self.__dataframe.columns.values)
                
            else:
                logger.warning("Inconsisent mapping present. Dataframe shape: %s Array shape: %s", df_shape, self_shape)
                self.__copy_dataframe = False

    @property
    def dataframe(self):
        return self.__dataframe

    @property
    def copy_dataframe(self):
        return self.__copy_dataframe    
# MIT License
#
# Copyright (C) IBM Corporation 2018
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
# rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit
# persons to whom the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the
# Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
# WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""
This module includes a wrapper for pandas Dataframes
"""
import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

class DataframeWrapper(np.ndarray):
    """
    Wrapper class for pandas dataframes and series. Note that the attacks will always return a
    numpy array even if this wrapper is used.
    """
    logger.warning("This is experimental. Use with caution")
    __copy_dataframe = True
    __dataframe = None

    def __new__(cls, dataframe, copy=True):
        """
        This function is called before __init__ for numpy arrays. Makes __init optional
        :param cls: This is a DataframeWrapper type for view casting
        :type cls: DataframeWrapper
        :param dataframe: A pandas dataframe used to initilize the ndarray
        :type dataframe: pd.DataFrame
        :param copy: If True, the a new instance of the supplied dataframe will be created. This ensures that
                     modifications to the array will not effect the original dataframe used to create the
                     wrapper object
        :rtype: `DataframeWrapper`
        """   
        if len(np.shape(dataframe)) > 2:
            logging.error("Input must have no more than 2 dimensions")
            return

        if isinstance(dataframe, pd.DataFrame) or isinstance(dataframe, pd.Series):

            # If the input is a series, we have to change it into a row
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

    # The order input is ignored. It's just there for compatibility
    def copy(self, order='C'):
    """
    This function overrides ndarray copy in order to preserve the dataframe when the object is copied.
    :param order: This is ignored usually. It exists so the call signature remains consistent
    :type order: string
    :rtype: `DataframeWrapper`
    """   
        if len(self.shape) == 1:
            self = self[np.newaxis, :]
        if(self.__copy_dataframe):  # If true, then we can copy the dataframe with the array
            column_names = self.__dataframe.columns.values
            return self.__new__(DataframeWrapper, pd.DataFrame(self, columns=column_names), copy=True)
        else:
            return np.ndarray.copy(self, order)

    def __array_finalize__(self, obj):
        """
        This function is called whenever a new object is made. Mainly, the dataframe
        object is preserved when possible
        :param obj: This is the DataframeWrapper object that is being used to create a new instance
        """   
        if obj is None:
            return

        self.__copy_dataframe = getattr(obj, '__copy_dataframe', True)
        self.__dataframe = getattr(obj, 'dataframe', None)

        if self.__dataframe is not None:
            self_shape = np.shape(self)
            df_shape = np.shape(self.__dataframe)

            # If the numpy array has the same number of columns, redefine the dataframe to mirror the numpy array.
            if len(self_shape) == 2 and self_shape[1] == df_shape[1]:
                self.__dataframe = pd.DataFrame(self, columns=self.__dataframe.columns.values)
            elif len(self_shape) == 1 and self_shape[0] == df_shape[1]:
                self.__dataframe = pd.DataFrame(self[np.newaxis], columns=self.__dataframe.columns.values)

            else:
                logger.warning("Inconsisent mapping present. Dataframe shape: %s Array shape: %s", df_shape, self_shape)
                self.__copy_dataframe = False

    @property
    def dataframe(self):
        """
        An attribute containing the dataframe equivalent of the numpy array
        """   
        return self.__dataframe

    @property
    def copy_dataframe(self):
        """
        An attribute that tracks if the dataframe matches the numpy array
        """   
        return self.__copy_dataframe

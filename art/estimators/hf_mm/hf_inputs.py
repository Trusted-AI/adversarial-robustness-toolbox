from collections import UserDict
import torch


class ARTInput(UserDict):
    """
    Custom class to allow HF inputs which are in a dictionary to be compatible with ART.
    Allows certain array-like functionality to be performed directly onto the input such as
    some arithmetic operations (addition, subtraction), and python operations
    such as slicing, reshaping, etc to be performed on the correct components of the HF input.
    """
    dtype = object
    shape = (1, 3, 224, 224)
    ndim = 4

    def __setitem__(self, key, value):
        # print('key ', key)
        # print('value ', value)
        if isinstance(key, slice):
            pixel_values = UserDict.__getitem__(self, 'pixel_values')
            original_shape = pixel_values.shape
            with torch.no_grad():
                pixel_values[key] = value['pixel_values']
                super().__setitem__('pixel_values', pixel_values)
            assert self['pixel_values'].shape == original_shape

        if isinstance(key, str):
            super().__setitem__(key, value)
            if key == 'pixel_values':
                pixel_values = UserDict.__getitem__(self, 'pixel_values')
                self.shape = pixel_values.shape
                self.ndim = pixel_values.ndim

        if isinstance(key, int):
            pixel_values = UserDict.__getitem__(self, 'pixel_values')
            original_shape = pixel_values.shape
            with torch.no_grad():
                if isinstance(value, ARTInput):
                    pixel_values[key] = value['pixel_values']
                else:
                    pixel_values[key] = torch.tensor(value)
                self['pixel_values'] = pixel_values
            assert self['pixel_values'].shape == original_shape

    def __getitem__(self, item):
        if isinstance(item, slice):
            pixel_values = UserDict.__getitem__(self, 'pixel_values')
            pixel_values = pixel_values[item]
            output = ARTInput(**self)
            output['pixel_values'] = pixel_values
            return output
        elif isinstance(item, int):
            pixel_values = UserDict.__getitem__(self, 'pixel_values')
            pixel_values = pixel_values[item]
            output = ARTInput(**self)
            output['pixel_values'] = pixel_values
            return output

        elif item in self.keys():
            return UserDict.__getitem__(self, item)

    def __add__(self, other):
        pixel_values = UserDict.__getitem__(self, 'pixel_values')
        if isinstance(other, ARTInput):
            pixel_values = pixel_values + other['pixel_values']
        else:
            pixel_values = pixel_values + torch.tensor(other)
        output = ARTInput(**self)
        output['pixel_values'] = pixel_values
        return output

    def __sub__(self, other):
        if isinstance(other, ARTInput):
            pixel_values = UserDict.__getitem__(self, 'pixel_values')
            pixel_values = pixel_values - other['pixel_values']
            output = ARTInput(**self)
            output['pixel_values'] = pixel_values
        else:
            raise ValueError('Unsupported type for __sub__ in ARTInput')
        return output

    def reshape(self, new_shape):
        pixel_values = UserDict.__getitem__(self, 'pixel_values')
        if not isinstance(pixel_values, torch.Tensor):
            pixel_values = torch.tensor(pixel_values)
        output = ARTInput(**self)
        output['pixel_values'] = torch.reshape(pixel_values, new_shape)
        return output

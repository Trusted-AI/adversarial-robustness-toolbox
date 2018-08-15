from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import math

from art.attacks.model_theft import ModelTheft


class SamplingModelTheft(ModelTheft):
    """
    The model theft attack using a set of natural samples to the network.
    """
    attack_params = ModelTheft.attack_params + ['x', 'query_datagen', 'fit_datagen']

    def __init__(self, x, query_datagen=None, fit_datagen=None):
        """
        Create a sampling model theft attack instance.

        :param x: Input samples to a model that the attacker can use.
        :type x: `numpy.array`
        :param query_datagen: A function taking a set of input and generating new/similar inputs.
        :type query_datagen: `function`
        :param fit_datagen: A generator function taking a set of input and generating new/similar inputs.
        :type fit_datagen: `function`
        """
        self.query_datagen = None
        self.fit_datagen = None
        super(SamplingModelTheft, self).__init__()
        kwargs = {'x': x,
                  'query_datagen': query_datagen,
                  'fit_datagen': fit_datagen
                  }
        assert self.set_params(**kwargs)

    def steal(self, model, stolen_model, budget, epochs=20, batch_size=128):
        """
        Steal a model, and return the stolen model.
        :param model: A trained model to steal.
        :type model: `Classifier`
        :param model: An untrained model to update with stealing.
        :type model:` Classifier`
        :param budget: The number of samples the attacker can query the model.
        :type budget: `int`
        """
        x = self.x
        num_samples = x.shape[0]
        x = np.concatenate([x] * math.ceil(budget / num_samples), axis=0)[:budget]
        if self.query_datagen != None:
            x = self.query_datagen(x)
        y = model.predict(x)
        if self.fit_datagen != None and isinstance(stolen_model, KerasClassifier):
            stolen_model._model.fit_generator(query_datagen(x,y), batch_size=batch_size, epochs=epochs)
        else:
            stolen_model.fit(x,y, batch_size=batch_size, nb_epochs=epochs)
        return stolen_model


    def set_params(self, **kwargs):
        """Take in a dictionary of parameters and applies attack-specific checks before saving them as attributes.

        :param x: Input samples to a model that the attacker can use.
        :type x: `numpy.array`
        :param query_datagen: A function taking a set of input and generating new/similar inputs.
        :type query_datagen: `function`
        :param fit_datagen: A generator function taking a set of input and generating new/similar inputs.
        :type fit_datagen: `function`
        """
        # Save attack-specific parameters
        super(SamplingModelTheft, self).set_params(**kwargs)

        if type(self.x) is not np.ndarray:
            raise ValueError("Input x must be a numpy.ndarray.")
        if self.fit_datagen and not callable(self.fit_datagen):
            raise ValueError("Input fit_datagen should be a function from input to new input")
        if self.query_datagen and not callable(self.query_datagen):
            raise ValueError("Input query_datagen should be a query function from input to new input")

        return True

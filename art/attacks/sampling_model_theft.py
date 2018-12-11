from __future__ import absolute_import, division, print_function, unicode_literals

import logging

import numpy as np

from art.attacks.model_theft import ModelTheft

logger = logging.getLogger(__name__)


class SamplingModelTheft(ModelTheft):
    """
    The model theft attack using a set of natural samples to the network as used in https://arxiv.org/abs/1806.00054.
    """
    attack_params = ModelTheft.attack_params + ['x', 'query_datagen', 'fit_datagen']

    def __init__(self, x, query_datagen=None, fit_datagen=None):
        """
        Create a sampling model theft attack instance.

        :param x: Input samples to a model that the attacker can use.
        :type x: `numpy.array`
        :param query_datagen: A callable, applied before querying, taking a set of input `x` and generating a sequence
               of new/similar input `x'`. It should be able to generate an indefinite length sequence.
        :type query_datagen: `Callable`
        :param fit_datagen: A callable, applied after querying and during the training, taking a set of input and output
               (x, y) and generating a sequence of new/similar input and output (x', y). It should be able to generate
               an indefinite length sequence.
        :type fit_datagen: `Callable`
        """
        self.query_datagen = None
        self.fit_datagen = None
        super(SamplingModelTheft, self).__init__()
        kwargs = {'x': x,
                  'query_datagen': query_datagen,
                  'fit_datagen': fit_datagen
                  }
        assert self.set_params(**kwargs)

    @staticmethod
    def _identity_datagen(x, y):
        while True:
            yield x, y

    def steal(self, model, stolen_model, budget, batch_size=128, nb_epochs=20):
        """
        Steal a model, and return the stolen model.

        :param model: A trained model to steal.
        :type model: `Classifier`
        :param stolen_model: An untrained model to update with stealing.
        :type stolen_model:` Classifier`
        :param budget: The number of samples the attacker can query the model.
        :type budget: `int`
        :param nb_epochs: Number of epochs to use for trainings.
        :type nb_epochs: `int`
        :param batch_size: Size of batches.
        :type batch_size: `int`
        :return: A stolen model.
        :rtype: `Classifier`
        """
        x = self.x
        num_samples = 0

        if self.query_datagen is not None:
            xlist = []
            ylist = []
            for xi in self.query_datagen(x):
                yi = model.predict(xi)
                xlist.append(xi)
                ylist.append(yi)
                num_samples += xi.shape[0]
                if num_samples >= budget:
                    break
            x = np.concatenate(xlist, axis=0)
            y = np.concatenate(ylist, axis=0)
        else:
            y = model.predict(x)
        x = x[:budget]
        y = y[:budget]

        output_fit_datagen = self.fit_datagen(x, y)
        for _ in range(nb_epochs):
            xlist = []
            ylist = []
            num_samples = 0
            for xi, yi in output_fit_datagen:
                xlist.append(xi)
                ylist.append(yi)
                num_samples += xi.shape[0]
                if num_samples >= budget:
                    break
            xt = np.concatenate(xlist[:budget], axis=0)
            yt = np.concatenate(ylist[:budget], axis=0)
            stolen_model.fit(xt, yt, batch_size=batch_size, nb_epochs=1)
        return stolen_model

    def set_params(self, **kwargs):
        """Take in a dictionary of parameters and applies attack-specific checks before saving them as attributes.

        :param x: Input samples to a model that the attacker can use.
        :type x: `numpy.array`
        :param query_datagen: A callable, applied before querying, taking a set of input `x` and generating a sequence
               of new/similar input `x'`. It should be able to generate an indefinite length sequence.
        :type query_datagen: `Callable`
        :param fit_datagen: A callable, applied after querying and during the training, taking a set of input and output
               (x, y) and generating a sequence of new/similar input and output (x', y). It should be able to generate
               an indefinite length sequence.
        :type fit_datagen: `Callable`
        """
        # Save attack-specific parameters
        super(SamplingModelTheft, self).set_params(**kwargs)

        if not isinstance(self.x, np.ndarray):
            raise ValueError("Input x must be a numpy.ndarray.")

        if self.fit_datagen and not callable(self.fit_datagen):
            raise ValueError("Input fit_datagen should be a Callable from input to new input")

        if self.query_datagen and not callable(self.query_datagen):
            raise ValueError("Input query_datagen should be a Callable from input to new input")

        if self.fit_datagen is None:
            self.fit_datagen = SamplingModelTheft._identity_datagen

        return True

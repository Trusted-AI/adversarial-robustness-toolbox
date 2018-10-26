from __future__ import absolute_import, division, print_function, unicode_literals

import logging

import numpy as np

from art.attacks import Attack

logger = logging.getLogger(__name__)


class TextFGSM:
    """
    Fast gradient sign method (FGSM) for text to be used as transformation strategy in the configurable text attack.
    """
    def __init__(self, eps):
        """
        Create a :class:`TextFGSM` transformation instance.

        :param eps: Attack step size (input variation).
        :type eps: `float`
        """
        self.eps = eps

    @property
    def uses_embedding(self):
        return True

    def __call__(self, classifier, x, y):
        """
        Apply FGSM attack on each component of `x`.

        :param classifier: A trained text model.
        :type classifier: :class:`TextClassifier`
        :param x: Individual sample.
        :type x: `np.ndarray`
        :param y: Label for sample `x` in one-hot encoding.
        :type y: `np.ndarray`
        :return: The adversarial counterpart of `x`.
        :rtype: `np.ndarray`
        """
        batch_x = np.expand_dims(x, axis=0)
        x_embed = classifier.to_embedding(batch_x)
        x_embed_adv = x_embed + self.eps * classifier.loss_gradient(batch_x, np.expand_dims(y, axis=0))
        return x_embed_adv[0]


class TemporalHeadScore:
    """
    Compute the temporal head score as described in https://arxiv.org/pdf/1801.04354
    """
    @property
    def uses_embedding(self):
        return False

    def __call__(self, classifier, x, y, null_token=0):
        """
        Compute the temporal head score for each token in `x` and model `classifier`.

        :param classifier: A trained text model.
        :type classifier: :class:`TextClassifier`
        :param x: Individual sample.
        :type x: `np.ndarray`
        :param y: Label for sample `x` in one-hot encoding.
        :type y: `np.ndarray`
        :param null_token: The index of the null token.
        :type null_token: `int`
        :return: The combined score.
        :rtype: `float`
        """
        # Create modified input
        x_padding = null_token * np.ones(x.shape)
        scores = np.zeros(x.shape[0])

        # Treat first token separately
        x_padding[0] = x[0]
        pred_including = classifier.predict(np.expand_dims(x_padding, axis=0))[0]
        label = np.argmax(pred_including)
        scores[0] = pred_including[label]

        for index in range(1, x.shape[0]):
            x_padding[index] = x[index]

            # Use previous prediction for current word excluded
            pred_before = pred_including

            # Compute score when current word included
            pred_including = classifier.predict(np.expand_dims(x_padding, axis=0))[0]
            label = np.argmax(pred_including)

            # Only consider the change in prediction for the predicted label
            scores[index] = pred_including[label] - pred_before[label]

        return scores


class TemporalTailScore:
    """
    Compute the temporal tail score as described in https://arxiv.org/pdf/1801.04354
    """
    @property
    def uses_embedding(self):
        return False

    def __call__(self, classifier, x, y, null_token=0):
        """
        Compute the temporal tail score for each token in `x` and model `classifier`.

        :param classifier: A trained text model.
        :type classifier: :class:`TextClassifier`
        :param x: Individual sample.
        :type x: `np.ndarray`
        :param y: Label for sample `x` in one-hot encoding.
        :type y: `np.ndarray`
        :param null_token: The index of the null token.
        :type null_token: `int`
        :return: The combined score.
        :rtype: `float`
        """
        # Create modified input
        x_padding = null_token * np.ones(x.shape)
        scores = np.zeros(x.shape[0])

        # Treat last token separately
        x_padding[-1] = x[-1]
        pred_including = classifier.predict(np.expand_dims(x_padding, axis=0))[0]
        label = np.argmax(pred_including)
        scores[-1] = pred_including[label]

        for index in range(x.shape[0] - 2, -1, -1):
            x_padding[index] = x[index]

            # Use prediction from previous iteration for current word excluded
            pred_after = pred_including

            # Compute score when current word included
            pred_including = classifier.predict(np.expand_dims(x_padding, axis=0))[0]
            label = np.argmax(pred_including)

            # Only consider the change in prediction for the predicted label
            scores[index] = pred_including[label] - pred_after[label]

        return scores


class CombinedScore:
    """
    Compute the combined values of the temporal head and tail scores as described in https://arxiv.org/pdf/1801.04354
    """
    def __init__(self, lamb=1.):
        """
        Create a :class:`CombinedScore` instance.

        :param lamb: The weight of the tail score (considering the head score has weight 1).
        :type lamb: `float`
        """
        self.lamb = lamb
        self.head_score = TemporalHeadScore()
        self.tail_score = TemporalTailScore()

    @property
    def uses_embedding(self):
        return False

    def __call__(self, classifier, x, y, null_token=0):
        """
        Compute the combined temporal head and tail score for each token in `x` and model `classifier`.

        :param classifier: A trained text model.
        :type classifier: :class:`TextClassifier`
        :param x: Individual sample.
        :type x: `np.ndarray`
        :param y: Label for sample `x` in one-hot encoding.
        :type y: `np.ndarray`
        :param null_token: The index of the null token.
        :type null_token: `int`
        :return: The combined score.
        :rtype: `float`
        """
        return self.head_score(classifier, x, None, null_token) + \
               self.lamb * self.tail_score(classifier, x, None, null_token)


def loss_gradient_score(classifier, x, y):
    """
    Score the tokens in `x` with the values of the loss gradient.

    :param classifier: A trained text model.
    :type classifier: :class:`TextClassifier`
    :param x: Individual sample.
    :type x: `np.ndarray`
    :param y: Label for sample `x` in one-hot encoding.
    :type y: `np.ndarray`
    :return:
    """
    return classifier.word_gradient(np.expand_dims(x, axis=0), np.expand_dims(y, axis=0))[0]


def check_prediction_change(classifier, x, x_adv):
    """
    Compare two individual samples and return true if `classifier` provides different predictions.

    :param classifier: A trained text model.
    :type classifier: :class:`TextClassifier`
    :param x: Individual sample to compare.
    :type x: `np.ndarray`
    :param x_adv: A second individual sample to compare to the first one.
    :type x_adv: `np.ndarray`
    :return: `True` if the label prediction of `classifier` has changed between `x` and `x_adv`.
    :rtype: `bool`
    """
    pred = np.argmax(classifier.predict(np.expand_dims(x, axis=0)))
    pred_adv = np.argmax(classifier.predict(np.expand_dims(x_adv, axis=0)))
    return pred != pred_adv


class ConfigurableTextAttack(Attack):
    """
    This class represents a generic text attack strategy.
    """
    attack_params = Attack.attack_params + ['stop_condition', 'score', 'transform', 'nb_changes']

    def __init__(self, classifier, transform, score, stop_condition, nb_changes=1):
        """
        Create a :class:`ConfigurableTextAttack` instance.

        :param classifier: A trained text model to be attacked.
        :type classifier: :class:`TextClassifier`
        :param transform: A callable strategy for transforming tokens. This should have a property `uses_embedding` set
                          to true if the transformation is performed in the embedding space of the model.
        :type transform: `Callable`
        :param score: A callable strategy for scoring tokens. This order is subsequently used to determine the priority
                      for changing the tokens as part of the attack.
        :type score: `Callable`
        :param stop_condition: A callable returning true if the stopping condition of the attack has been fulfilled.
        :type stop_condition: `Callable`
        :param nb_changes: Number of maximum changes allowed for each input. Each change usually corresponds with the
                           displacement of one token.
        :type nb_changes: `int`
        """
        from art.classifiers import TextClassifier

        if not isinstance(classifier, TextClassifier):
            raise TypeError('This attack is only supported for text classifiers.')

        super(ConfigurableTextAttack, self).__init__(classifier)
        params = {'stop_condition': stop_condition, 'score': score, 'transform': transform, 'nb_changes': nb_changes}
        self.set_params(**params)

    def generate(self, x, **kwargs):
        """
         Generate adversarial samples and return them in an array.

        :param x: An array with the original inputs to be attacked.
        :type x: `np.ndarray`
        :param kwargs:
        :return: An array holding the adversarial examples of the same shape as input `x`.
        :rtype: `np.ndarray`
        """
        from art.utils import get_labels_np_array

        x_adv = np.copy(x)
        preds = get_labels_np_array(self.classifier.predict(x))

        for i, input_ in enumerate(x_adv):
            print('Attacking input %i' % i)
            logger.debug('Attacking input %i', i)
            scores = self.score(self.classifier, input_, preds[i])
            prioritized_tokens = np.flip(scores.argsort(), axis=0)

            if hasattr(self.transform, 'uses_embedding') and self.transform.uses_embedding:
                input_emb = self.classifier.to_embedding(np.expand_dims(input_, axis=0))[0]
            transform_values = self.transform(self.classifier, input_, preds[i])

            for j, token_pos in enumerate(prioritized_tokens):
                if hasattr(self.transform, 'uses_embedding') and self.transform.uses_embedding:
                    input_emb[token_pos, :] = transform_values[token_pos]
                    old_token = input_[token_pos]
                    input_ = self.classifier.to_id(np.expand_dims(input_emb, axis=0))[0]
                else:
                    input_[token_pos] = transform_values[token_pos]

                logger.debug('Changed word in position %i from ID %i to %i', token_pos, old_token, input_[token_pos])
                print('Changed word in position %i from ID %i to %i' % (token_pos, old_token, input_[token_pos]))

                if self.stop_condition(self.classifier, x[i], input_) or j >= self.nb_changes - 1:
                    break
            x_adv[i] = input_

        adv_preds = np.argmax(self.classifier.predict(x_adv), axis=1)
        rate = np.sum(adv_preds != np.argmax(preds, axis=1)) / x_adv.shape[0]
        print('Success rate of text attack: %.2f%%' % rate)
        logger.info('Success rate of text attack: %.2f%%', rate)

        return x_adv

    def set_params(self, **kwargs):
        """
        Take in a dictionary of parameters and applies attack-specific checks before saving them as attributes.

        :param transform: A callable strategy for transforming tokens. This should have a property `uses_embedding` set
                          to true if the transformation is performed in the embedding space of the model.
        :type transform: `Callable`
        :param score: A callable strategy for scoring tokens. This order is subsequently used to determine the priority
                      for changing the tokens as part of the attack.
        :type score: `Callable`
        :param stop_condition: A callable returning true if the stopping condition of the attack has been fulfilled.
        :type stop_condition: `Callable`
        :param nb_changes: Number of maximum changes allowed for each input. Each change usually corresponds with the
                           displacement of one token.
        :type nb_changes: `int`
        """
        # Save attack-specific parameters
        super(ConfigurableTextAttack, self).set_params(**kwargs)

        if not isinstance(self.nb_changes, (int, np.integer)) or self.nb_changes <= 0:
            raise ValueError('The number of allowed changes for the attack should be a positive integer.')

        if not callable(self.transform):
            raise ValueError('`transform` should be a callable transformation.')

        if not callable(self.score):
            raise ValueError('`score` should be a callable scoring function.')

        if not callable(self.stop_condition):
            raise ValueError('`stop_condition` should be a callable returning a boolean.')

from __future__ import absolute_import, division, print_function, unicode_literals

import logging

import numpy as np

from art.attacks import Attack

logger = logging.getLogger(__name__)


class TextFGSM:
    def __init__(self, eps):
        self.eps = eps

    @property
    def uses_embedding(self):
        return True

    def __call__(self, classifier, x, y):
        x_embed = classifier.to_embedding(x)
        x_embed_adv = x_embed + \
                      self.eps * classifier.loss_gradient(np.expand_dims(x, axis=0), np.expand_dims(y, axis=0))[0]
        return x_embed_adv


class TemporalHeadScore:
    @property
    def uses_embedding(self):
        return False

    def __call__(self, classifier, x, y, null_token=0):
        """

        :param classifier:
        :param x:
        :param null_token:
        :return:
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
    @property
    def uses_embedding(self):
        return False

    def __call__(self, classifier, x, y, null_token=0):
        """

        :param classifier:
        :param x:
        :param null_token:
        :return:
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

    """
    def __init__(self, lamb=1.):
        self.lamb = lamb
        self.head_score = TemporalHeadScore()
        self.tail_score = TemporalTailScore()

    @property
    def uses_embedding(self):
        return False

    def __call__(self, classifier, x, y, null_token=0):
        return self.head_score(classifier, x, None, null_token) + \
               self.lamb * self.tail_score(classifier, x, None, null_token)


def loss_gradient_score(classifier, x, y):
    """

    :param classifier:
    :param x:
    :param y:
    :return:
    """
    return classifier.word_gradient(np.expand_dims(x, axis=0), np.expand_dims(y, axis=0))[0]


def check_prediction_change(classifier, x, x_adv):
    """
    Compare two individual samples and return true if `classifier` provides different predictions.

    :param classifier:
    :param x:
    :param x_adv:
    :return:
    """
    pred = np.argmax(classifier.predict(np.expand_dims(x, axis=0)))
    pred_adv = np.argmax(classifier.predict(np.expand_dims(x_adv, axis=0)))
    return pred != pred_adv


class ConfigurableTextAttack(Attack):
    """
    TODO
    """
    attack_params = Attack.attack_params + ['stop_condition', 'score', 'transform', 'nb_changes']

    def __init__(self, classifier, transform, score, stop_condition, nb_changes=1):
        from art.classifiers import TextClassifier

        if not isinstance(classifier, TextClassifier):
            raise TypeError('This attack is only supported for text classifiers.')

        super(ConfigurableTextAttack, self).__init__(classifier)
        params = {'stop_condition': stop_condition, 'score': score, 'transform': transform, 'nb_changes': nb_changes}
        self.set_params(**params)

    def generate(self, x, **kwargs):
        """

        :param x:
        :param kwargs:
        :return:
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
                # TODO otherwise, detect automatically if the transform operates in the embedding space
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

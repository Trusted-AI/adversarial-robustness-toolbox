from __future__ import absolute_import, division, print_function

from config import config_dict

from keras.utils.np_utils import to_categorical
import tensorflow as tf

from cleverhans.attacks_tf import CarliniWagnerL2 as CWL2
from cleverhans.utils_keras import KerasModelWrapper

from src.attacks.attack import Attack


class CarliniL2Method(Attack):
    """
    The L_2 optimized attack of Carlini and Wagner (2016). This attack is the most efficient and should be used as
    the primary attack to evaluate potential defenses (wrt to the L_0 and L_inf attacks). This code is a wrapper on top
    of the code provided by Cleverhans, which reproduces the authors' original code
    (https://github.com/carlini/nn_robust_attacks). Paper link: https://arxiv.org/pdf/1608.04644.pdf
    """
    attack_params = ['batch_size', 'confidence', 'targeted', 'learning_rate', 'binary_search_steps', 'max_iterations',
                     'abort_early', 'initial_const', 'clip_min', 'clip_max']

    def __init__(self, classifier, sess, batch_size=10, confidence=2.3, targeted=True, learning_rate=5e-3,
                 binary_search_steps=5, max_iterations=1000, abort_early=True, initial_const=1e-2, clip_min=0,
                 clip_max=1):
        """
        Create a Carlini L_2 attack instance.
        :param classifier: A function that takes a symbolic input and returns the symbolic output for the classifier's
        predictions.
        :param sess: The tf session to run graphs in.
        :param batch_size: (optional integer) Number of attacks to run simultaneously.
        :param confidence: Confidence of adversarial examples: higher produces examples that are farther away, but more
               strongly classified as adversarial.
        :param targeted: (optional boolean) should the attack target one specific class
        :param learning_rate: The learning rate for the attack algorithm. Smaller values produce better results but are
               slower to converge.
        :param binary_search_steps: (optional positive integer) number of times to adjust the constant with binary search
        :param max_iterations: (integer) The maximum number of iterations.
        :param abort_early: (optional boolean) if we stop improving, abort gradient descent early
        :param initial_const: (optional float, positive) The initial trade-off constant c to use to tune the relative
               importance of distance and confidence. If binary_search_steps is large,
               the initial constant is not important.
        :param clip_min: (optional float) Minimum input component value.
        :param clip_max: (optional float) Maximum input component value.
        """
        super(CarliniL2Method, self).__init__(classifier, sess)

        kwargs = {'batch_size': batch_size,
                  'confidence': confidence,
                  'targeted': targeted,
                  'learning_rate': learning_rate,
                  'binary_search_steps': binary_search_steps,
                  'max_iterations': max_iterations,
                  'abort_early': abort_early,
                  'initial_const': initial_const,
                  'clip_min': clip_min,
                  'clip_max': clip_max
                  }
        self.set_params(**kwargs)

    def generate(self, x_val, **kwargs):
        """
        Generate adversarial samples and return them in a Numpy array.
        :param x_val:
        :param y_val: If self.targeted is true, then y_val represents the target labels. If self.targeted is false, then
                      targets are the original class labels.
        :return: A Numpy array holding the adversarial examples.
        """
        # Parse and save attack-specific parameters
        params_cpy = dict(kwargs)
        y_val = params_cpy.pop('y_val', None)
        assert self.set_params(**params_cpy)

        model = KerasModelWrapper(self.classifier.model)
        attack = CWL2(self.sess, model, self.batch_size, self.confidence, self.targeted, self.learning_rate,
                      self.binary_search_steps, self.max_iterations, self.abort_early, self.initial_const,
                      self.clip_min, self.clip_max, self.classifier.model.output_shape[1], x_val.shape[1:])

        if y_val is None:
            # No labels provided, use model prediction as correct class
            x = tf.placeholder(dtype=tf.float32, shape=self.classifier.model.get_input_shape_at(0))
            y_val = self.sess.run(tf.argmax(self.classifier.model(x), axis=1), {x: x_val})
            y_val = to_categorical(y_val, self.classifier.model.get_output_shape_at(-1)[-1])

        return attack.attack(x_val, y_val)

    def set_params(self, **kwargs):
        """Take in a dictionary of parameters and applies attack-specific checks before saving them as attributes.

        Attack-specific parameters:
        :param batch_size: (optional integer) Number of attacks to run simultaneously.
        :param confidence: Confidence of adversarial examples: higher produces examples that are farther away, but more
               strongly classified as adversarial.
        :param targeted: (optional boolean) should the attack target one specific class
        :param learning_rate: The learning rate for the attack algorithm. Smaller values produce better results but are
               slower to converge.
        :param binary_search_steps: (optional positive integer) number of times to adjust the constant with binary search
        :param max_iterations: (integer) The maximum number of iterations.
        :param abort_early: (optional boolean) if we stop improving, abort gradient descent early
        :param initial_const: (optional float, positive) The initial trade-off constant c to use to tune the relative
               importance of distance and confidence. If binary_search_steps is large,
               the initial constant is not important.
        :param clip_min: (optional float) Minimum input component value.
        :param clip_max: (optional float) Maximum input component value.
        """
        # Save attack-specific parameters
        super(CarliniL2Method, self).set_params(**kwargs)

        if type(self.max_iterations) is not int or self.max_iterations <= 0:
            raise ValueError("The number of iterations must be a positive integer.")

        return True
